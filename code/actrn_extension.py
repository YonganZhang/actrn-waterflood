"""ACTRN v4 extension: deep sequence baselines and hyperparameter sensitivity.

Runs the protocol in ``docs/PROTOCOL.md`` (extension section) on the
registered v4 repeats without refitting or changing any v4 model:

* deep sequence baselines (Elman RNN, LSTM, Transformer, Autoformer, Reformer) on every benchmark
  setting and in the field case, with the same nine-candidate menu and selection rules as v4;
* one-at-a-time sensitivity of ACTRN to latent dimension, l2 penalty, ensemble size and learning
  rate around the default configuration, on the base and channelised populations;
* paired statistics of the registered v4 ACTRN against each sequence baseline.

Usage (from the project root):
    .venv/bin/python code/actrn_extension.py [--workers 44] [--out data/processed/actrn_v4_ext]
    .venv/bin/python code/actrn_extension.py --quick --out _tmp/actrn_v4_ext_quick
    # corrected Autoformer only (ADR-0006), without the sensitivity runs
    .venv/bin/python code/actrn_extension.py --kinds Autoformer --skip-sensitivity \
        --protocol docs/PROTOCOL.md \
        --out data/processed/actrn_v4_ext_autoformer
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from action_conditioned_decision import _sha256
from actrn_experiments import (ABLATION_SETTINGS, DATA_SEED_BASE, DEV_SEED_BASE, FIELD_FOLD_ENDS,
                               FIELD_TEST, REPEATS, SETTINGS, benchmark_job, environment,
                               field_job, select_best)
from actrn_sequence_baselines import SEQUENCE_KINDS
from actrn_stats import BOOTSTRAP_SEED, paired_statistics, summarize

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "PROTOCOL.md"
V4 = ROOT / "data" / "processed" / "actrn_v4"
SENSITIVITY_DEFAULT = {"latent_dim": 16, "learning_rate": 0.002, "l2": 1e-3, "ensemble": 5}
SENSITIVITY = {"latent_dim": [4, 8, 16, 32, 64], "l2": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
               "ensemble": [1, 2, 5, 10], "learning_rate": [5e-4, 1e-3, 2e-3, 5e-3]}
COST = {"Reformer": 0, "Autoformer": 1, "Transformer": 2, "LSTM": 3, "RNN": 4}
KEYS = ["population", "response", "repeat", "family"]


def run_parallel(fn, specs, workers):
    specs = sorted(specs, key=lambda s: COST.get(s.get("family", ""), 9))
    if workers <= 1:
        return [fn(s) for s in specs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, specs, chunksize=1))


def sensitivity_configs():
    seen, out = set(), []
    for axis, values in SENSITIVITY.items():
        for v in values:
            cfg = {**SENSITIVITY_DEFAULT, axis: v}
            key = tuple(sorted(cfg.items()))
            if key not in seen:
                seen.add(key)
                out.append(cfg)
    return out


def sensitivity_job(spec):
    cfg = spec["config"]
    params = {k: cfg[k] for k in ("latent_dim", "learning_rate", "l2")}
    res = benchmark_job({**spec, "family": "ACTRN", "params": params,
                         "overrides": {"ensemble": cfg["ensemble"]}})
    row = res["rows"][0]
    row.update({f"cfg_{k}": v for k, v in cfg.items()})
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "actrn_v4_ext"))
    ap.add_argument("--workers", type=int, default=44)
    ap.add_argument("--quick", action="store_true",
                    help="smoke run on one development repeat with two configurations")
    ap.add_argument("--kinds", nargs="+", choices=SEQUENCE_KINDS, default=list(SEQUENCE_KINDS),
                    help="sequence models to run (default: all)")
    ap.add_argument("--skip-sensitivity", action="store_true",
                    help="do not run the ACTRN hyperparameter sensitivity")
    ap.add_argument("--protocol", default=str(PROTOCOL.relative_to(ROOT)),
                    help="protocol file recorded in the manifest (relative to the project root)")
    args = ap.parse_args()
    protocol = ROOT / args.protocol
    kinds = [k for k in SEQUENCE_KINDS if k in args.kinds]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    quick = args.quick
    seed_base = DEV_SEED_BASE if quick else DATA_SEED_BASE
    repeats = [0] if quick else list(REPEATS)
    settings = SETTINGS[:1] if quick else SETTINGS
    grid_limit = 2 if quick else 8

    # 1. deep sequence baselines on the benchmark
    specs = [{"repeat": r, "population": p, "response": resp, "family": f,
              "seed_base": seed_base, "grid_limit": grid_limit}
             for (p, resp), r, f in itertools.product(settings, repeats, kinds)]
    results = run_parallel(benchmark_job, specs, args.workers)
    tuning = pd.DataFrame([row for res in results for row in res["rows"]])
    selected = select_best(tuning, KEYS, "regret")
    chosen = set(map(tuple, selected[KEYS + ["config_id"]].astype(str).to_numpy()))
    decisions = pd.DataFrame([{**{k: row[k] for k in KEYS}, **d}
                              for res in results for row in res["rows"]
                              if tuple(str(row[k]) for k in KEYS + ["config_id"]) in chosen
                              for d in res["details"][row["config_id"]]])
    tuning["selected"] = [tuple(map(str, t)) in chosen
                          for t in tuning[KEYS + ["config_id"]].itertuples(index=False)]
    tuning.to_csv(out / "sequence_tuning.csv", index=False)
    selected.to_csv(out / "sequence_runs.csv", index=False)
    decisions.to_csv(out / "sequence_decisions.csv", index=False)
    summarize(selected, ["population", "response", "family"]).to_csv(
        out / "sequence_summary.csv", index=False)
    summarize(select_best(tuning, KEYS, "nmae"), ["population", "response", "family"]).to_csv(
        out / "sequence_summary_nmae_selection.csv", index=False)

    # 2. ACTRN (registered v4) against the sequence baselines
    if not quick:
        v4_runs = pd.read_csv(V4 / "benchmark_runs.csv")
        v4_dec = pd.read_csv(V4 / "benchmark_decisions.csv")
        runs = pd.concat([v4_runs[v4_runs.family == "ACTRN"], selected], ignore_index=True)
        dec = pd.concat([v4_dec[v4_dec.family == "ACTRN"], decisions], ignore_index=True)
        (out / "sequence_stats.json").write_text(
            json.dumps(paired_statistics(runs, dec), indent=2) + "\n")

    # 3. ACTRN hyperparameter sensitivity (fixed configurations, no selection)
    sens_specs = [] if args.skip_sensitivity else [
        {"repeat": r, "population": p, "response": resp, "seed_base": seed_base,
         "grid_limit": grid_limit, "config": cfg}
        for (p, resp), r, cfg in itertools.product(
            [st for st in settings if st in ABLATION_SETTINGS], repeats,
            sensitivity_configs()[:2] if quick else sensitivity_configs())]
    if sens_specs:
        sens = pd.DataFrame(run_parallel(sensitivity_job, sens_specs, args.workers))
        sens.to_csv(out / "sensitivity_hyperparameters.csv", index=False)

    # 4. field case
    field = None
    if not quick:
        fspecs = [{"fold": fold, "family": f, "test_start": end - FIELD_TEST, "test_end": end,
                   "grid_limit": grid_limit}
                  for fold, end in enumerate(FIELD_FOLD_ENDS) for f in kinds]
        ftuning = pd.DataFrame([row for rows in run_parallel(field_job, fspecs, args.workers)
                                for row in rows])
        fsel = select_best(ftuning, ["fold", "family"], "nmae")
        ftuning["selected"] = ftuning.index.isin(fsel.index)
        ftuning.to_csv(out / "sequence_field_tuning.csv", index=False)
        fsel.to_csv(out / "sequence_field_metrics.csv", index=False)
        field = {"jobs": len(fspecs)}

    files = sorted(p for p in out.glob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "protocol": args.protocol if protocol.exists() else None,
        "protocol_sha256": _sha256(protocol) if protocol.exists() else None,
        "kinds": kinds,
        "quick": quick, "environment": environment(), "workers": args.workers,
        "wall_seconds": time.perf_counter() - t0,
        "jobs": {"sequence": len(specs), "sensitivity": len(sens_specs), "field": field},
        "seeds": {"data": f"{seed_base} + 1009*repeat", "repeats": repeats,
                  "model": "42 + repeat", "bootstrap": BOOTSTRAP_SEED},
        "outputs": {p.name: _sha256(p) for p in files},
        "command": " ".join([Path(sys.executable).name] + sys.argv),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({k: manifest[k] for k in ("wall_seconds", "jobs")}, indent=2))


if __name__ == "__main__":
    main()
