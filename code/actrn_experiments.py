"""ACTRN v3 experiment protocol.

Runs the registered protocol in ``docs/PROTOCOL.md``:
equal-budget validation tuning for every model, component ablations, feature-parity
baselines, an alternative-saturation robustness benchmark, paired statistics over the five
v2 repeats and five fresh repeats, and a leakage-free rolling-origin field evaluation with a
look-ahead-free decision example.  The v2 protocol and runner are at git tag
``actrn-v2-registered``.

Usage (from the project root):
    .venv/bin/python code/actrn_experiments.py [--workers 40] [--out data/processed/actrn_v3]
    .venv/bin/python code/actrn_experiments.py --quick --out _tmp/actrn_v3_quick
    .venv/bin/python code/actrn_experiments.py --dev --out _tmp/actrn_v3_dev   # development seeds

Outputs are CSV/JSON files plus ``manifest.json`` (environment, git state, protocol hash,
seeds, timings and output hashes).  Registered v1 outputs are never overwritten.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import functools
import itertools
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from action_conditioned_decision import (
    FlatModel,
    _sha256,
    build_field_case_dataset,
    make_decision_sets,
    metrics,
    score_decision_sets,
    simulate_dataset,
)
from actrn_model import ACTRN

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "PROTOCOL.md"
BASELINES = ("Ridge", "Random forest", "HGB", "MLP")
FAMILIES = ("ACTRN",) + BASELINES + tuple(f"{b} (engineered)" for b in BASELINES)
ABLATIONS = {
    "ACTRN w/o contrastive learning": {"contrastive": False},
    "ACTRN w/o interaction": {"interaction": "none"},
    "ACTRN w/o response proxies": {"history_proxies": False},
    "ACTRN raw action input": {"action_features": "raw"},
    "ACTRN linear response": {"nonlinear": False},
    "ACTRN concatenation interaction": {"interaction": "concat"},
}
DATA_SEED_BASE = 20260716      # registered benchmark draws: base + 1009 * repeat
DEV_SEED_BASE = 910_000        # development draws used while designing v3 (never reported)
REGISTERED_REPEATS = tuple(range(5))     # the five v2 repeats
FRESH_REPEATS = tuple(range(5, 10))      # first run under v3
FIELD_GAP = 8          # = planning horizon: all fitting labels precede the next block
FIELD_VALIDATION = 30
FIELD_TEST = 60
FIELD_FOLD_ENDS = (220, 280, 340, 400)
CASE_LEVEL_M3 = 20000.0
BOOTSTRAP_SEED = 20261016
BOOTSTRAP_DRAWS = 10_000


def _grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, values)) for values in itertools.product(*axes.values())]


GRIDS = {
    "benchmark": {
        "ACTRN": _grid(latent_dim=[16, 32], learning_rate=[0.001, 0.002], l2=[1e-4, 1e-3]),
        "Ridge": _grid(alpha=[0.1, 0.3, 1, 3, 10, 30, 100, 300]),
        "Random forest": _grid(max_depth=[8, 14, 20, None], min_samples_leaf=[1, 3]),
        "HGB": _grid(learning_rate=[0.05, 0.1], max_leaf_nodes=[24, 48],
                     min_samples_leaf=[12, 30]),
        "MLP": _grid(hidden_layer_sizes=[(72, 48), (128, 64)], alpha=[2e-4, 1e-3],
                     learning_rate_init=[1e-3, 3e-4]),
    },
    "field": {
        "ACTRN": _grid(latent_dim=[8, 16], learning_rate=[5e-4, 1e-3], l2=[0.01, 0.1]),
        "Ridge": _grid(alpha=[1, 3, 10, 30, 100, 300, 1000, 3000]),
        "Random forest": _grid(max_depth=[4, 8, 14, None], min_samples_leaf=[3, 8]),
        "HGB": _grid(learning_rate=[0.05, 0.1], max_leaf_nodes=[8, 24],
                     min_samples_leaf=[5, 12]),
        "MLP": _grid(hidden_layer_sizes=[(32, 16), (72, 48)], alpha=[2e-4, 1e-2],
                     learning_rate_init=[1e-3, 3e-4]),
    },
}


def make_actrn(setting: str, params: dict, seed: int, **overrides) -> ACTRN:
    if setting == "benchmark":
        base = dict(max_epochs=900, patience=80,
                    level_alphas=(0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000))
    else:
        base = dict(max_epochs=1200, patience=100,
                    level_alphas=(1, 3, 10, 30, 100, 300, 1000, 3000))
    base.update(latent_dim=int(params["latent_dim"]), learning_rate=params["learning_rate"],
                l2=params["l2"], ensemble=5, action_alphas=(0.01, 0.1, 1, 10), seed=seed)
    base.update(overrides)
    return ACTRN(**base)


def family_kind(family: str) -> tuple[str, str]:
    """Return (model kind, baseline feature set) for a benchmark family name."""
    if family.startswith("ACTRN"):
        return "ACTRN", ""
    if family.endswith(" (engineered)"):
        return family.removesuffix(" (engineered)"), "engineered"
    return family, "raw"


# ----------------------------------------------------------------- benchmark
@functools.lru_cache(maxsize=4)
def benchmark_data(repeat: int, response: str, seed_base: int = DATA_SEED_BASE):
    data_seed = seed_base + repeat * 1009
    h, a, y, groups, contexts, context_ids = simulate_dataset(
        n_reservoirs=48, seed=data_seed, return_context_ids=True, response=response)
    split = {"train": groups < 32, "validation": (groups >= 32) & (groups < 40),
             "test": groups >= 40}
    test_contexts = [c for c in contexts if c.reservoir_id >= 40]
    decision_sets = make_decision_sets(test_contexts, np.random.default_rng(data_seed + 1776),
                                       horizon=a.shape[1], n_candidates=80)
    return data_seed, h, a, y, context_ids, split, decision_sets


def benchmark_job(spec: dict) -> dict:
    repeat, response, family = spec["repeat"], spec["response"], spec["family"]
    data_seed, h, a, y, context_ids, s, decision_sets = benchmark_data(
        repeat, response, spec.get("seed_base", DATA_SEED_BASE))
    kind, features = family_kind(family)
    seed = 42 + repeat
    t0 = time.perf_counter()
    if kind == "ACTRN":
        model = make_actrn("benchmark", spec["params"], seed, **ABLATIONS.get(family, {}))
        model.fit(h[s["train"]], a[s["train"]], y[s["train"]],
                  validation_data=(h[s["validation"]], a[s["validation"]], y[s["validation"]]),
                  context_ids=context_ids[s["train"]],
                  validation_context_ids=context_ids[s["validation"]])
        epochs = json.dumps(model.epochs_)
    else:
        model = FlatModel(kind, seed=seed, params=spec["params"], features=features)
        model.fit(h[s["train"]], a[s["train"]], y[s["train"]])
        epochs = None
    fit_seconds = time.perf_counter() - t0
    val = metrics(y[s["validation"]], model.predict(h[s["validation"]], a[s["validation"]]))
    row = {**{k: spec[k] for k in ("repeat", "response", "family", "config_id")},
           "params": json.dumps(spec["params"], default=list), "data_seed": data_seed,
           "model_seed": seed, "val_nmae_pct": val["nmae_pct"], "fit_seconds": fit_seconds,
           "epochs": epochs}
    row.update(metrics(y[s["test"]], model.predict(h[s["test"]], a[s["test"]])))
    summary, details = score_decision_sets(model, decision_sets)
    row.update(summary)
    return {"row": row, "details": details}


# ----------------------------------------------------------------- field
@functools.lru_cache(maxsize=1)
def field_data():
    hist, act, out, dates = build_field_case_dataset()
    level = np.median(hist[:, -4:, 2:7].sum(axis=2), axis=1)
    scale = level + 5000.0
    target = (out - level[:, None]) / scale[:, None]
    return hist, act, out, dates, level, scale, target


def field_blocks(test_start: int) -> dict:
    """Train | gap | validation | gap | test, gap = planning horizon."""
    val_end = test_start - FIELD_GAP
    val_start = val_end - FIELD_VALIDATION
    train_end = val_start - FIELD_GAP
    if train_end < 60:
        raise ValueError("field block leaves too few training windows")
    return {"train": np.arange(0, train_end), "validation": np.arange(val_start, val_end)}


def field_predict(model, idx, bounds):
    hist, act, out, dates, level, scale, target = field_data()
    normalized = np.clip(model.predict(hist[idx], act[idx]), -1.0, 3.0)
    return np.clip(level[idx, None] + scale[idx, None] * normalized, *bounds)


def field_fit(family: str, params: dict, seed: int, blocks: dict):
    hist, act, out, dates, level, scale, target = field_data()
    tr, va = blocks["train"], blocks["validation"]
    kind, features = family_kind(family)
    if kind == "ACTRN":
        model = make_actrn("field", params, seed)
        model.fit(hist[tr], act[tr], target[tr],
                  validation_data=(hist[va], act[va], target[va]))
    else:
        model = FlatModel(kind, seed=seed, params=params, features=features)
        model.fit(hist[tr], act[tr], target[tr])
    bounds = (float(np.quantile(out[tr], 0.025)), float(np.quantile(out[tr], 0.975)))
    return model, bounds


def field_job(spec: dict) -> dict:
    hist, act, out, dates, level, scale, target = field_data()
    test_start = spec["test_start"]
    blocks = field_blocks(test_start)
    seed = 242 + spec["fold"] if spec["fold"] != "case" else 747
    t0 = time.perf_counter()
    model, bounds = field_fit(spec["family"], spec["params"], seed, blocks)
    fit_seconds = time.perf_counter() - t0
    va = blocks["validation"]
    row = {"fold": spec["fold"], "family": spec["family"], "config_id": spec["config_id"],
           "params": json.dumps(spec["params"], default=list), "model_seed": seed,
           "train_windows": f"0-{int(blocks['train'][-1])}",
           "validation_windows": f"{int(va[0])}-{int(va[-1])}",
           "val_nmae_pct": metrics(out[va], field_predict(model, va, bounds))["nmae_pct"],
           "fit_seconds": fit_seconds}
    if spec["fold"] != "case":
        test = np.arange(test_start, spec["test_end"])
        row.update(metrics(out[test], field_predict(model, test, bounds)))
    return {"row": row}


def field_persistence(fold: int, test_start: int, test_end: int) -> dict:
    hist, act, out, dates, level, scale, target = field_data()
    test = np.arange(test_start, test_end)
    row = {"fold": fold, "family": "Persistence", "config_id": 0, "params": "{}"}
    row.update(metrics(out[test], np.repeat(level[test, None], out.shape[1], axis=1)))
    return row


def field_decision_example(case_params: dict, out_dir: Path) -> dict:
    """Look-ahead-free decision example at the latest active window."""
    hist, act, out, dates, level, scale, target = field_data()
    c = int(np.flatnonzero(level >= CASE_LEVEL_M3)[-1])
    model, bounds = field_fit("ACTRN", case_params, 747, field_blocks(c))
    complete = np.arange(0, c - act.shape[1] + 1)        # action blocks ending before date c
    totals = act[complete].sum(axis=(1, 2))
    budget = float(np.median(totals[-52:]))
    keep = complete[np.abs(totals - budget) / budget <= 0.05]
    plans = act[keep] * (budget / act[keep].sum(axis=(1, 2)))[:, None, None]
    history = np.repeat(hist[c][None], len(plans), axis=0)
    normalized = np.clip(model.predict(history, plans), -1.0, 3.0)
    pred = np.clip(level[c] + scale[c] * normalized, *bounds)
    at_bound = (np.isclose(pred, bounds[0]) | np.isclose(pred, bounds[1])).sum(axis=1)
    best = int(np.argmax(pred.sum(axis=1)))
    realized_pred = np.clip(
        level[c] + scale[c] * np.clip(model.predict(hist[c][None], act[c][None]), -1, 3),
        *bounds)[0]
    pd.DataFrame({
        "source_action_block_date": [str(dates[j].date()) for j in keep],
        "original_total_m3": act[keep].sum(axis=(1, 2)),
        "predicted_cumulative_oil_m3": pred.sum(axis=1),
        "steps_at_clip_bound": at_bound,
        "selected": np.arange(len(keep)) == best,
    }).to_csv(out_dir / "field_case_candidates.csv", index=False)
    pd.DataFrame({
        "step": np.arange(1, act.shape[1] + 1),
        "injector_1_m3": plans[best][:, 0], "injector_2_m3": plans[best][:, 1],
        "predicted_field_oil_m3": pred[best],
    }).to_csv(out_dir / "field_case_decision.csv", index=False)
    pd.DataFrame({
        "step": np.arange(1, act.shape[1] + 1),
        "implemented_injector_1_m3": act[c][:, 0], "implemented_injector_2_m3": act[c][:, 1],
        "observed_field_oil_m3": out[c], "predicted_under_implemented_m3": realized_pred,
        "persistence_m3": np.full(act.shape[1], level[c]),
    }).to_csv(out_dir / "field_case_backtest.csv", index=False)
    return {
        "case_decision_date": str(dates[c].date()), "case_index": c,
        "case_level_m3": float(level[c]), "latest_candidate_block": str(dates[complete[-1]].date()),
        "budget_m3": budget, "library_size": int(len(keep)),
        "selected_source_block": str(dates[keep[best]].date()),
        "selected_predicted_cumulative_m3": float(pred[best].sum()),
        "candidate_predicted_cumulative_range_m3": [float(pred.sum(axis=1).min()),
                                                     float(pred.sum(axis=1).max())],
        "candidates_with_any_clipped_step": int((at_bound > 0).sum()),
        "implemented_schedule_total_m3": float(act[c].sum()),
        "implemented_predicted_cumulative_m3": float(realized_pred.sum()),
        "implemented_observed_cumulative_m3": float(out[c].sum()),
        "clip_bounds_m3": list(bounds), "case_params": case_params,
    }


# ----------------------------------------------------------------- statistics
def select_best(tuning: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    order = tuning.sort_values(keys + ["val_nmae_pct", "config_id"])
    return order.groupby(keys, sort=False).head(1)


def _bootstrap_mean(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Means of BOOTSTRAP_DRAWS resamples that redraw contexts within each repeat."""
    total = np.zeros(BOOTSTRAP_DRAWS)
    for g in groups:
        total += g[rng.integers(0, len(g), (BOOTSTRAP_DRAWS, len(g)))].sum(axis=1)
    return total / sum(len(g) for g in groups)


def _compare(rep: pd.DataFrame, ctx: pd.DataFrame, other: str, rng) -> dict:
    entry = {}
    for metric in ("nmae_pct", "decision_regret_pct", "optimal_selection_pct",
                   "top_decile_selection_pct"):
        diff = (rep[metric]["ACTRN"] - rep[metric][other]).to_numpy()
        entry[metric] = {
            "mean_actrn": float(rep[metric]["ACTRN"].mean()),
            "mean_other": float(rep[metric][other].mean()),
            "mean_diff": float(diff.mean()),
            "repeats_actrn_lower": int(np.sum(diff < 0)),
            "wilcoxon_exact_p": (float(stats.wilcoxon(diff, method="exact").pvalue)
                                 if np.any(diff != 0) else 1.0),
        }
    d = (ctx["ACTRN"] - ctx[other]).dropna()
    boot = _bootstrap_mean([d.xs(r, level="repeat").to_numpy()
                            for r in sorted(d.index.unique("repeat"))], rng)
    entry["context_regret_pct"] = {
        "n": int(len(d)), "actrn_better": int((d < 0).sum()),
        "other_better": int((d > 0).sum()), "ties": int((d == 0).sum()),
        "mean_diff": float(d.mean()),
        "bootstrap95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "wilcoxon_p": (float(stats.wilcoxon(d.to_numpy()).pvalue)
                       if np.any(d.to_numpy() != 0) else 1.0),
    }
    return entry


def paired_statistics(runs: pd.DataFrame, decisions: pd.DataFrame) -> dict:
    """ACTRN versus every other family, for all repeats and for the v2 / fresh subsets."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    present = set(runs["repeat"])
    subsets = {"all": sorted(present), "registered": [r for r in REGISTERED_REPEATS if r in present],
               "fresh": [r for r in FRESH_REPEATS if r in present]}
    result = {}
    for response, rr in runs.groupby("response"):
        dd = decisions[decisions["response"] == response]
        result[response] = {}
        for subset, repeats in subsets.items():
            if not repeats or (subset != "all" and len(repeats) == len(present)):
                continue
            r_sub = rr[rr["repeat"].isin(repeats)]
            ctx = dd[dd["repeat"].isin(repeats)].pivot_table(
                index=["repeat", "reservoir_id", "decision_age"], columns="family",
                values="decision_regret_pct")
            rep = r_sub.pivot(index="repeat", columns="family")
            result[response][subset] = {
                other: _compare(rep, ctx, other, rng)
                for other in r_sub["family"].unique() if other != "ACTRN"}
    return result


def summarize(runs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    cols = ["nmae_pct", "nrmse_pct", "r2", "decision_regret", "decision_regret_pct",
            "optimal_selection_pct", "top_decile_selection_pct"]
    cols = [c for c in cols if c in runs]
    agg = runs.groupby(keys, sort=False)[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


# ----------------------------------------------------------------- driver
def run_parallel(fn, specs, workers):
    specs = sorted(specs, key=lambda s: 0 if str(s.get("family", "")).startswith("ACTRN") else 1)
    if workers <= 1:
        return [fn(s) for s in specs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, specs, chunksize=1))


def benchmark_specs(repeats, responses, families, grid_limit):
    specs = []
    for repeat, response, family in itertools.product(repeats, responses, families):
        kind, _ = family_kind(family)
        for cid, params in enumerate(GRIDS["benchmark"][kind][:grid_limit]):
            specs.append({"repeat": repeat, "response": response, "family": family,
                          "config_id": cid, "params": params})
    return specs


def run_benchmark(out_dir, workers, repeats, grid_limit, responses, seed_base):
    specs = benchmark_specs(repeats, responses, FAMILIES, grid_limit)
    for spec in specs:
        spec["seed_base"] = seed_base
    results = run_parallel(benchmark_job, specs, workers)
    tuning = pd.DataFrame([r["row"] for r in results])
    selected = select_best(tuning, ["repeat", "response", "family"])
    # ablations reuse the full ACTRN configuration selected in the same repeat
    full = selected[(selected["family"] == "ACTRN") & (selected["response"] == "exp")]
    abl_specs = [{"repeat": int(r.repeat), "response": "exp", "family": name,
                  "config_id": int(r.config_id), "params": json.loads(r.params),
                  "seed_base": seed_base}
                 for r in full.itertuples() for name in ABLATIONS]
    abl = run_parallel(benchmark_job, abl_specs, workers)
    results += abl
    abl_rows = pd.DataFrame([r["row"] for r in abl])
    tuning = pd.concat([tuning, abl_rows], ignore_index=True)
    runs = pd.concat([selected, abl_rows], ignore_index=True)
    chosen = {(r["repeat"], r["response"], r["family"], r["config_id"])
              for r in runs[["repeat", "response", "family", "config_id"]].to_dict("records")}
    detail_rows = []
    for res in results:
        key = tuple(res["row"][k] for k in ("repeat", "response", "family", "config_id"))
        if key in chosen:
            for d in res["details"]:
                detail_rows.append({**{k: res["row"][k] for k in
                                       ("repeat", "response", "family")}, **d})
    decisions = pd.DataFrame(detail_rows)
    tuning["selected"] = tuning.set_index(["repeat", "response", "family", "config_id"]).index.isin(
        list(chosen))
    tuning.to_csv(out_dir / "benchmark_tuning.csv", index=False)
    runs.to_csv(out_dir / "benchmark_runs.csv", index=False)
    decisions.to_csv(out_dir / "benchmark_decisions.csv", index=False)
    summarize(runs, ["response", "family"]).to_csv(out_dir / "benchmark_summary.csv", index=False)
    stats_out = paired_statistics(runs, decisions)
    (out_dir / "benchmark_stats.json").write_text(json.dumps(stats_out, indent=2) + "\n")
    return {"jobs": len(results), "fit_seconds_total": float(tuning["fit_seconds"].sum())}


def run_field(out_dir, workers, grid_limit, folds):
    hist, act, out, dates, level, scale, target = field_data()
    specs, persistence = [], []
    for fold, end in enumerate(FIELD_FOLD_ENDS):
        if fold not in folds:
            continue
        start = end - FIELD_TEST
        persistence.append(field_persistence(fold, start, end))
        for family in FAMILIES:
            for cid, params in enumerate(GRIDS["field"][family_kind(family)[0]][:grid_limit]):
                specs.append({"fold": fold, "family": family, "config_id": cid, "params": params,
                              "test_start": start, "test_end": end})
    c = int(np.flatnonzero(level >= CASE_LEVEL_M3)[-1])
    for cid, params in enumerate(GRIDS["field"]["ACTRN"][:grid_limit]):
        specs.append({"fold": "case", "family": "ACTRN", "config_id": cid, "params": params,
                      "test_start": c})
    rows = [r["row"] for r in run_parallel(field_job, specs, workers)]
    tuning = pd.DataFrame(rows)
    rolling = tuning[tuning["fold"] != "case"]
    selected = select_best(rolling, ["fold", "family"])
    selected = pd.concat([pd.DataFrame(persistence), selected], ignore_index=True)
    selected["test_start"] = selected["fold"].map(
        lambda f: str(dates[FIELD_FOLD_ENDS[f] - FIELD_TEST].date()))
    selected["test_end"] = selected["fold"].map(lambda f: str(dates[FIELD_FOLD_ENDS[f] - 1].date()))
    tuning["selected"] = tuning.index.isin(select_best(rolling, ["fold", "family"]).index)
    case_rows = tuning[tuning["fold"] == "case"].sort_values(["val_nmae_pct", "config_id"])
    tuning.loc[case_rows.index[:1], "selected"] = True
    tuning.to_csv(out_dir / "field_tuning.csv", index=False)
    selected.to_csv(out_dir / "field_rolling_metrics.csv", index=False)
    summarize(selected, ["family"]).to_csv(out_dir / "field_summary.csv", index=False)
    case = field_decision_example(json.loads(case_rows.iloc[0]["params"]), out_dir)
    (out_dir / "field_case_summary.json").write_text(json.dumps(case, indent=2) + "\n")
    return {"jobs": len(specs), "case": case}


def environment() -> dict:
    pkgs = ["numpy", "pandas", "scikit-learn", "scipy", "pyarrow"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "code/"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return {"python": platform.python_version(),
            "packages": {p: metadata.version(p) for p in pkgs},
            "git_commit": commit, "code_dirty": bool(dirty),
            "code_dirty_paths": dirty.splitlines()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "actrn_v3"))
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--quick", action="store_true",
                    help="smoke run: 1 repeat, 2 configs, exp response, fold 0")
    ap.add_argument("--dev", action="store_true",
                    help="benchmark on the development draws only (no field run)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    quick = args.quick
    t0 = time.perf_counter()
    seed_base = DEV_SEED_BASE if args.dev else DATA_SEED_BASE
    bench = run_benchmark(out_dir, args.workers,
                          repeats=[0] if quick else REGISTERED_REPEATS + FRESH_REPEATS,
                          grid_limit=2 if quick else 8,
                          responses=["exp"] if quick else ["exp", "hill"], seed_base=seed_base)
    field = None if args.dev else run_field(out_dir, args.workers, grid_limit=2 if quick else 8,
                                            folds={0} if quick else {0, 1, 2, 3})
    files = sorted(p for p in out_dir.glob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": _sha256(PROTOCOL),
        "quick": quick, "dev_seeds": args.dev, "environment": environment(),
        "workers": args.workers,
        "wall_seconds": time.perf_counter() - t0, "benchmark": bench, "field": field,
        "seeds": {"data": f"{seed_base} + 1009*repeat", "model": "42 + repeat",
                  "ensemble_members": "model seed + 1000*k, k = 0..4",
                  "decision_candidates": "data_seed + 1776", "field_model": "242 + fold",
                  "field_case_model": 747, "bootstrap": BOOTSTRAP_SEED},
        "outputs": {p.name: _sha256(p) for p in files},
        "command": " ".join([Path(sys.executable).name] + sys.argv),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({k: manifest[k] for k in ("wall_seconds", "benchmark", "field")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
