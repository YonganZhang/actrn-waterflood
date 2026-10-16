"""ACTRN v4 experiment protocol.

Runs the registered protocol in ``docs/PROTOCOL.md``:
known-truth benchmarks on two reservoir populations (``base`` and ``channelised``) and an
alternative saturation response, twenty previously unused repeats, the same candidate menu for
every model family (eight configurations and their average, chosen on validation decision
libraries), component ablations, a model-free measure of how much the best schedule depends on
the reservoir state, and the leakage-free rolling-origin field evaluation with a look-ahead-free
decision example.  Earlier protocols and runners are at git tags ``actrn-v1-registered``,
``actrn-v2-registered`` and ``actrn-v3-registered``.

Usage (from the project root):
    .venv/bin/python code/actrn_experiments.py [--workers 44] [--out data/processed/actrn_v4]
    .venv/bin/python code/actrn_experiments.py --quick --out _tmp/actrn_v4_quick
    .venv/bin/python code/actrn_experiments.py --dev --out _tmp/actrn_v4_dev   # development seeds

Outputs are CSV/JSON files plus ``manifest.json`` (environment, git state, protocol hash,
seeds, timings and output hashes).  Registered outputs of earlier protocols are never touched.
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

from action_conditioned_decision import (
    FlatModel,
    _sha256,
    build_field_case_dataset,
    candidate_predictions,
    make_decision_sets,
    metrics,
    rollout,
    score_predictions,
    simulate_dataset,
)
from actrn_model import ACTRN
from actrn_sequence_baselines import SEQUENCE_KINDS, SequenceModel
from actrn_stats import BOOTSTRAP_SEED, paired_statistics, summarize

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "PROTOCOL.md"
BASELINES = ("Ridge", "Random forest", "HGB", "MLP")
FAMILIES = ("ACTRN",) + BASELINES + tuple(f"{b} (engineered)" for b in BASELINES)
ABLATIONS = {
    "ACTRN w/o contrastive learning": {"contrastive": False},
    "ACTRN w/o interaction": {"interaction": "none"},
    "ACTRN w/o response proxies": {"history_proxies": "none"},
    "ACTRN correlation proxies only": {"history_proxies": "correlation"},
    "ACTRN raw action input": {"action_features": "raw"},
    "ACTRN linear response": {"nonlinear": False},
    "ACTRN concatenation interaction": {"interaction": "concat"},
}
SETTINGS = (("base", "exp"), ("base", "hill"), ("channelised", "exp"))
ABLATION_SETTINGS = (("base", "exp"), ("channelised", "exp"))
# training-data reductions (reservoirs, states per reservoir, schedules per state); full = 32x18x6
LEARNING_CURVE = {"16x18x6": (16, 18, 6), "32x18x3": (32, 18, 3), "8x18x6": (8, 18, 6)}
LC_FAMILIES = ("ACTRN", "ACTRN linear response", "Ridge (engineered)", "HGB (engineered)")
DATA_SEED_BASE = 20260716      # benchmark draws: base + 1009 * repeat
DEV_SEED_BASE = 910_000        # development draws (never reported)
REPEATS = tuple(range(10, 30))  # repeats 0-9 were used by earlier protocols
ENSEMBLE = 8                   # candidate id: average of the eight configurations
FIELD_GAP = 8          # = planning horizon: all fitting labels precede the next block
FIELD_VALIDATION = 30
FIELD_TEST = 60
FIELD_FOLD_ENDS = (220, 280, 340, 400)
CASE_LEVEL_M3 = 20000.0


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
        **{k: _grid(d_model=[32, 64], learning_rate=[1e-3, 3e-3], dropout=[0.0, 0.1])
           for k in SEQUENCE_KINDS},
    },
    "field": {
        "ACTRN": _grid(latent_dim=[8, 16], learning_rate=[5e-4, 1e-3], l2=[0.01, 0.1]),
        "Ridge": _grid(alpha=[10, 30, 100, 300, 1000, 3000, 10000, 30000]),
        "Random forest": _grid(max_depth=[4, 8, 14, None], min_samples_leaf=[3, 8]),
        "HGB": _grid(learning_rate=[0.05, 0.1], max_leaf_nodes=[8, 24],
                     min_samples_leaf=[5, 12]),
        "MLP": _grid(hidden_layer_sizes=[(32, 16), (72, 48)], alpha=[2e-4, 1e-2],
                     learning_rate_init=[1e-3, 3e-4]),
        **{k: _grid(d_model=[16, 32], learning_rate=[1e-3, 3e-3], dropout=[0.1, 0.3])
           for k in SEQUENCE_KINDS},
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
    """Return (model kind, baseline feature set) for a family name."""
    if family.startswith("ACTRN"):
        return "ACTRN", ""
    if family in SEQUENCE_KINDS:
        return family, "sequence"
    if family.endswith(" (engineered)"):
        return family.removesuffix(" (engineered)"), "engineered"
    return family, "raw"


class AveragedModel:
    """Average prediction of several fitted models (the ``ENSEMBLE`` candidate)."""

    def __init__(self, models):
        self.models = list(models)

    def predict(self, histories, actions):
        return np.mean([m.predict(histories, actions) for m in self.models], axis=0)


def _params_label(grid, candidate):
    if candidate == ENSEMBLE:
        return "average of all configurations"
    return json.dumps(grid[candidate], default=list)


def _mean_sets(per_config):
    return [np.mean(parts, axis=0) for parts in zip(*per_config)]


# ----------------------------------------------------------------- benchmark
@functools.lru_cache(maxsize=3)
def benchmark_data(repeat: int, population: str = "base", response: str = "exp",
                   seed_base: int = DATA_SEED_BASE):
    data_seed = seed_base + repeat * 1009
    h, a, y, groups, contexts, context_ids = simulate_dataset(
        n_reservoirs=48, seed=data_seed, return_context_ids=True, response=response,
        population=population)
    split = {"train": groups < 32, "validation": (groups >= 32) & (groups < 40),
             "test": groups >= 40}
    horizon = a.shape[1]
    test_sets = make_decision_sets([c for c in contexts if c.reservoir_id >= 40],
                                   np.random.default_rng(data_seed + 1776), horizon, 80)
    val_sets = make_decision_sets([c for c in contexts if 32 <= c.reservoir_id < 40],
                                  np.random.default_rng(data_seed + 2999), horizon, 80)
    return data_seed, h, a, y, context_ids, split, val_sets, test_sets


def training_subset(context_ids: np.ndarray, train: np.ndarray, subset: str | None,
                    states_per_reservoir: int = 18) -> np.ndarray:
    """Training rows kept by a learning-curve subset (nested in the full training set)."""
    if subset is None:
        return train
    n_res, n_states, n_plans = LEARNING_CURVE[subset]
    reservoir = context_ids // states_per_reservoir
    state = context_ids % states_per_reservoir
    plan = np.zeros(len(context_ids), int)
    for c in np.unique(context_ids):
        idx = np.flatnonzero(context_ids == c)
        plan[idx] = np.arange(len(idx))
    keep = ((reservoir < n_res) & (state % (states_per_reservoir // n_states) == 0)
            & (plan < n_plans))
    return train & keep


def _fit_benchmark(kind, features, params, seed, overrides, h, a, y, cid, s):
    tr, va = s["train"], s["validation"]
    if kind == "ACTRN":
        model = make_actrn("benchmark", params, seed, **overrides)
        return model.fit(h[tr], a[tr], y[tr], validation_data=(h[va], a[va], y[va]),
                         context_ids=cid[tr], validation_context_ids=cid[va])
    if features == "sequence":
        return SequenceModel(kind, seed=seed, params=params).fit(
            h[tr], a[tr], y[tr], validation_data=(h[va], a[va], y[va]))
    return FlatModel(kind, seed=seed, params=params, features=features).fit(h[tr], a[tr], y[tr])


def benchmark_job(spec: dict) -> dict:
    """Fit every configuration of one family and score the nine-candidate menu."""
    repeat, population, response, family = (spec["repeat"], spec["population"],
                                             spec["response"], spec["family"])
    data_seed, h, a, y, cid, s, val_sets, test_sets = benchmark_data(
        repeat, population, response, spec["seed_base"])
    kind, features = family_kind(family)
    s = {**s, "train": training_subset(cid, s["train"], spec.get("subset"))}
    if "params" in spec:          # a single fixed configuration (sensitivity analysis)
        grid, wanted = [spec["params"]], [0]
    else:
        grid = GRIDS["benchmark"][kind][: spec["grid_limit"]]
        wanted = spec.get("candidates") or list(range(len(grid))) + [ENSEMBLE]
    fit_ids = list(range(len(grid))) if ENSEMBLE in wanted else list(wanted)
    seed = 42 + repeat
    va, te = s["validation"], s["test"]
    preds = {}
    t0 = time.perf_counter()
    for c in fit_ids:
        overrides = {**ABLATIONS.get(family, {}), **spec.get("overrides", {})}
        model = _fit_benchmark(kind, features, grid[c], seed, overrides, h, a, y, cid, s)
        preds[c] = {"val": model.predict(h[va], a[va]), "test": model.predict(h[te], a[te]),
                    "val_sets": candidate_predictions(model, val_sets),
                    "test_sets": candidate_predictions(model, test_sets)}
    fit_seconds = time.perf_counter() - t0
    if ENSEMBLE in wanted:
        members = [preds[c] for c in fit_ids]
        preds[ENSEMBLE] = {"val": np.mean([m["val"] for m in members], axis=0),
                           "test": np.mean([m["test"] for m in members], axis=0),
                           "val_sets": _mean_sets([m["val_sets"] for m in members]),
                           "test_sets": _mean_sets([m["test_sets"] for m in members])}
    rows, details = [], {}
    for c in wanted:
        p = preds[c]
        val_summary, _ = score_predictions(val_sets, p["val_sets"])
        test_summary, test_details = score_predictions(test_sets, p["test_sets"])
        row = {"repeat": repeat, "population": population, "response": response,
               "family": family, "config_id": c, "params": _params_label(grid, c),
               "train_subset": spec.get("subset") or "32x18x6",
               "train_rows": int(s["train"].sum()),
               "data_seed": data_seed, "model_seed": seed,
               "val_nmae_pct": metrics(y[va], p["val"])["nmae_pct"],
               "val_regret_pct": val_summary["decision_regret_pct"],
               "val_gain_capture_pct": val_summary["gain_capture_pct"],
               "fit_seconds_all_configs": fit_seconds}
        row.update(metrics(y[te], p["test"]))
        row.update(test_summary)
        rows.append(row)
        details[c] = test_details
    return {"rows": rows, "details": details}


def state_dependence_job(spec: dict) -> list[dict]:
    """Model-free: regret of a random candidate and of the best state-blind rule.

    The state-blind rule ranks a context's candidates by their mean true value over all other
    test contexts of the repeat; its regret bounds what any selector whose ranking ignores the
    reservoir state could achieve with perfect knowledge of the average response."""
    repeat, population, response = spec["repeat"], spec["population"], spec["response"]
    sets = benchmark_data(repeat, population, response, spec["seed_base"])[-1]
    rows = []
    for i, item in enumerate(sets):
        own = item["truth"].sum(axis=1)
        others = np.array([[rollout(o["context"].reservoir, o["context"].pressure,
                                    o["context"].water, plan, o["context"].age, response).sum()
                            for plan in item["plans"]]
                           for j, o in enumerate(sets) if j != i])
        blind = int(np.argmax(others.mean(axis=0)))
        rows.append({"repeat": repeat, "population": population, "response": response,
                     "reservoir_id": item["context"].reservoir_id,
                     "decision_age": item["context"].age,
                     "random_regret_pct": 100 * (own.max() - own.mean()) / own.max(),
                     "state_blind_regret_pct": 100 * (own.max() - own[blind]) / own.max()})
    return rows


def select_best(tuning: pd.DataFrame, keys: list[str], rule: str) -> pd.DataFrame:
    order = ["val_regret_pct", "val_nmae_pct"] if rule == "regret" else ["val_nmae_pct"]
    ranked = tuning.sort_values(keys + order + ["config_id"])
    return ranked.groupby(keys, sort=False).head(1)


def run_parallel(fn, specs, workers):
    specs = sorted(specs, key=lambda s: 0 if "Random" in s.get("family", "")
                   or "HGB" in s.get("family", "") else 1)
    if workers <= 1:
        return [fn(s) for s in specs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, specs, chunksize=1))


def run_benchmark(out_dir, workers, repeats, settings, seed_base, grid_limit):
    keys = ["population", "response", "repeat", "family"]
    specs = [{"repeat": r, "population": p, "response": resp, "family": f,
              "seed_base": seed_base, "grid_limit": grid_limit}
             for (p, resp), r, f in itertools.product(settings, repeats, FAMILIES)]
    results = run_parallel(benchmark_job, specs, workers)
    tuning = pd.DataFrame([row for res in results for row in res["rows"]])
    selected = select_best(tuning, keys, "regret")
    full = selected[selected["family"] == "ACTRN"]
    abl_specs = [{"repeat": int(r.repeat), "population": r.population, "response": r.response,
                  "family": name, "seed_base": seed_base, "grid_limit": grid_limit,
                  "candidates": [int(r.config_id)]}
                 for r in full.itertuples() if (r.population, r.response) in settings
                 and (r.population, r.response) in ABLATION_SETTINGS for name in ABLATIONS]
    abl = run_parallel(benchmark_job, abl_specs, workers)
    abl_rows = pd.DataFrame([row for res in abl for row in res["rows"]])
    dep_specs = [{"repeat": r, "population": p, "response": resp, "seed_base": seed_base}
                 for (p, resp), r in itertools.product(settings, repeats)]
    dependence = pd.DataFrame([row for rows in run_parallel(state_dependence_job, dep_specs,
                                                            workers) for row in rows])
    runs = pd.concat([selected, abl_rows], ignore_index=True)
    chosen = set(map(tuple, runs[keys + ["config_id"]].astype(str).to_numpy()))
    detail_rows = []
    for res in results + abl:
        for row in res["rows"]:
            if tuple(str(row[k]) for k in keys + ["config_id"]) in chosen:
                detail_rows += [{**{k: row[k] for k in keys}, **d}
                                for d in res["details"][row["config_id"]]]
    decisions = pd.DataFrame(detail_rows)
    tuning["selected"] = [tuple(map(str, t)) in chosen
                          for t in tuning[keys + ["config_id"]].itertuples(index=False)]
    nmae_selected = select_best(tuning, keys, "nmae")
    tuning.to_csv(out_dir / "benchmark_tuning.csv", index=False)
    runs.to_csv(out_dir / "benchmark_runs.csv", index=False)
    decisions.to_csv(out_dir / "benchmark_decisions.csv", index=False)
    dependence.to_csv(out_dir / "benchmark_state_dependence.csv", index=False)
    summarize(runs, ["population", "response", "family"]).to_csv(
        out_dir / "benchmark_summary.csv", index=False)
    summarize(nmae_selected, ["population", "response", "family"]).to_csv(
        out_dir / "benchmark_summary_nmae_selection.csv", index=False)
    stats_out = paired_statistics(runs, decisions)
    (out_dir / "benchmark_stats.json").write_text(json.dumps(stats_out, indent=2) + "\n")
    lc_specs = [{"repeat": r, "population": p, "response": resp, "family": f,
                 "seed_base": seed_base, "grid_limit": grid_limit, "subset": sub}
                for (p, resp), r, f, sub in itertools.product(
                    [st for st in settings if st in ABLATION_SETTINGS], repeats, LC_FAMILIES,
                    LEARNING_CURVE)]
    lc = run_parallel(benchmark_job, lc_specs, workers)
    lc_rows = pd.DataFrame([row for res in lc for row in res["rows"]])
    full_rows = runs[runs["family"].isin(LC_FAMILIES)
                     & runs[["population", "response"]].apply(tuple, axis=1).isin(
                         ABLATION_SETTINGS)]
    curve = pd.concat([full_rows, select_best(lc_rows, keys + ["train_subset"], "regret")],
                      ignore_index=True)
    curve.to_csv(out_dir / "benchmark_learning_curve.csv", index=False)
    return {"jobs": len(results) + len(abl) + len(lc),
            "fit_seconds_total": float(sum(r["rows"][0]["fit_seconds_all_configs"]
                                           for r in results + abl + lc))}


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


def field_predict(model, idx, bounds, histories=None, actions=None):
    hist, act, out, dates, level, scale, target = field_data()
    h = hist[idx] if histories is None else histories
    a = act[idx] if actions is None else actions
    normalized = np.clip(model.predict(h, a), -1.0, 3.0)
    return np.clip(level[idx, None] + scale[idx, None] * normalized, *bounds)


def field_fit_family(family: str, seed: int, blocks: dict, grid_limit: int):
    """Fit every configuration; return the models and the training-label clipping bounds."""
    hist, act, out, dates, level, scale, target = field_data()
    tr, va = blocks["train"], blocks["validation"]
    kind, features = family_kind(family)
    models = []
    for params in GRIDS["field"][kind][:grid_limit]:
        if kind == "ACTRN":
            model = make_actrn("field", params, seed)
            model.fit(hist[tr], act[tr], target[tr],
                      validation_data=(hist[va], act[va], target[va]))
        elif features == "sequence":
            model = SequenceModel(kind, seed=seed, params=params, relative_oil=True)
            model.fit(hist[tr], act[tr], target[tr],
                      validation_data=(hist[va], act[va], target[va]))
        else:
            model = FlatModel(kind, seed=seed, params=params, features=features)
            model.fit(hist[tr], act[tr], target[tr])
        models.append(model)
    bounds = (float(np.quantile(out[tr], 0.025)), float(np.quantile(out[tr], 0.975)))
    return models, bounds


def field_job(spec: dict) -> list[dict]:
    hist, act, out, dates, level, scale, target = field_data()
    blocks = field_blocks(spec["test_start"])
    seed = 242 + spec["fold"]
    t0 = time.perf_counter()
    models, bounds = field_fit_family(spec["family"], seed, blocks, spec["grid_limit"])
    fit_seconds = time.perf_counter() - t0
    grid = GRIDS["field"][family_kind(spec["family"])[0]]
    va, test = blocks["validation"], np.arange(spec["test_start"], spec["test_end"])
    rows = []
    for c, model in list(enumerate(models)) + [(ENSEMBLE, AveragedModel(models))]:
        row = {"fold": spec["fold"], "family": spec["family"], "config_id": c,
               "params": _params_label(grid, c), "model_seed": seed,
               "train_windows": f"0-{int(blocks['train'][-1])}",
               "validation_windows": f"{int(va[0])}-{int(va[-1])}",
               "val_nmae_pct": metrics(out[va], field_predict(model, va, bounds))["nmae_pct"],
               "fit_seconds_all_configs": fit_seconds}
        row.update(metrics(out[test], field_predict(model, test, bounds)))
        rows.append(row)
    return rows


def field_persistence(fold: int, test_start: int, test_end: int) -> dict:
    hist, act, out, dates, level, scale, target = field_data()
    test = np.arange(test_start, test_end)
    row = {"fold": fold, "family": "Persistence", "config_id": 0, "params": "{}"}
    row.update(metrics(out[test], np.repeat(level[test, None], out.shape[1], axis=1)))
    return row


def field_decision_example(out_dir: Path, grid_limit: int) -> dict:
    """Look-ahead-free decision example at the latest active window."""
    hist, act, out, dates, level, scale, target = field_data()
    c = int(np.flatnonzero(level >= CASE_LEVEL_M3)[-1])
    blocks = field_blocks(c)
    models, bounds = field_fit_family("ACTRN", 747, blocks, grid_limit)
    va = blocks["validation"]
    menu = list(enumerate(models)) + [(ENSEMBLE, AveragedModel(models))]
    scores = [(metrics(out[va], field_predict(m, va, bounds))["nmae_pct"], cid, m)
              for cid, m in menu]
    val_nmae, chosen, model = min(scores, key=lambda x: (x[0], x[1]))
    complete = np.arange(0, c - act.shape[1] + 1)        # action blocks ending before date c
    totals = act[complete].sum(axis=(1, 2))
    budget = float(np.median(totals[-52:]))
    keep = complete[np.abs(totals - budget) / budget <= 0.05]
    plans = act[keep] * (budget / act[keep].sum(axis=(1, 2)))[:, None, None]
    idx = np.full(len(plans), c)
    pred = field_predict(model, idx, bounds, np.repeat(hist[c][None], len(plans), axis=0), plans)
    at_bound = (np.isclose(pred, bounds[0]) | np.isclose(pred, bounds[1])).sum(axis=1)
    best = int(np.argmax(pred.sum(axis=1)))
    realized_pred = field_predict(model, np.array([c]), bounds)[0]
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
        "clip_bounds_m3": list(bounds),
        "case_candidate": _params_label(GRIDS["field"]["ACTRN"], chosen),
        "case_validation_nmae_pct": val_nmae,
    }


def run_field(out_dir, workers, grid_limit, folds):
    specs, persistence = [], []
    for fold, end in enumerate(FIELD_FOLD_ENDS):
        if fold not in folds:
            continue
        start = end - FIELD_TEST
        persistence.append(field_persistence(fold, start, end))
        specs += [{"fold": fold, "family": f, "test_start": start, "test_end": end,
                   "grid_limit": grid_limit} for f in FAMILIES]
    _, _, _, dates, *_ = field_data()
    tuning = pd.DataFrame([row for rows in run_parallel(field_job, specs, workers)
                           for row in rows])
    selected = select_best(tuning, ["fold", "family"], "nmae")
    tuning["selected"] = tuning.index.isin(selected.index)
    selected = pd.concat([pd.DataFrame(persistence), selected], ignore_index=True)
    selected["test_start"] = selected["fold"].map(
        lambda f: str(dates[FIELD_FOLD_ENDS[f] - FIELD_TEST].date()))
    selected["test_end"] = selected["fold"].map(
        lambda f: str(dates[FIELD_FOLD_ENDS[f] - 1].date()))
    tuning.to_csv(out_dir / "field_tuning.csv", index=False)
    selected.to_csv(out_dir / "field_rolling_metrics.csv", index=False)
    summarize(selected, ["family"]).to_csv(out_dir / "field_summary.csv", index=False)
    case = field_decision_example(out_dir, grid_limit)
    (out_dir / "field_case_summary.json").write_text(json.dumps(case, indent=2) + "\n")
    return {"jobs": len(specs), "case": case}


# ----------------------------------------------------------------- driver
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
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "actrn_v4"))
    ap.add_argument("--workers", type=int, default=44)
    ap.add_argument("--quick", action="store_true",
                    help="smoke run: 1 repeat, 2 configurations, base/exp, fold 0")
    ap.add_argument("--dev", action="store_true",
                    help="benchmark on the development draws only (no field run)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    quick = args.quick
    t0 = time.perf_counter()
    seed_base = DEV_SEED_BASE if args.dev else DATA_SEED_BASE
    repeats = tuple(range(10)) if args.dev else REPEATS
    grid_limit = 2 if quick else 8
    bench = run_benchmark(out_dir, args.workers, repeats=repeats[:1] if quick else repeats,
                          settings=SETTINGS[:1] if quick else SETTINGS, seed_base=seed_base,
                          grid_limit=grid_limit)
    field = None if args.dev else run_field(out_dir, args.workers, grid_limit,
                                            folds={0} if quick else {0, 1, 2, 3})
    files = sorted(p for p in out_dir.glob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": _sha256(PROTOCOL),
        "quick": quick, "dev_seeds": args.dev, "environment": environment(),
        "workers": args.workers, "wall_seconds": time.perf_counter() - t0,
        "benchmark": bench, "field": field,
        "seeds": {"data": f"{seed_base} + 1009*repeat", "repeats": list(repeats),
                  "model": "42 + repeat", "ensemble_members": "model seed + 1000*k, k = 0..4",
                  "test_candidates": "data_seed + 1776",
                  "validation_candidates": "data_seed + 2999",
                  "field_model": "242 + fold", "field_case_model": 747,
                  "bootstrap": BOOTSTRAP_SEED},
        "outputs": {p.name: _sha256(p) for p in files},
        "command": " ".join([Path(sys.executable).name] + sys.argv),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({k: manifest[k] for k in ("wall_seconds", "benchmark", "field")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
