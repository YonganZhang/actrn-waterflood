"""Exploratory (not pre-registered) field analysis added after the v2 protocol run.

The pre-registered field comparison used raw-input baselines.  Because the benchmark
ablations identified the multiscale action representation as the main contributor, this
script evaluates the same four baselines on ACTRN's engineered features under the
identical leakage-free blocks, grids and selection rule.  Results are labelled
exploratory in the manuscript.

Usage (from the project root):
    .venv/bin/python code/actrn_field_exploratory.py [--out data/processed/actrn_v2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from action_conditioned_decision import FlatModel, metrics
from actrn_experiments import (BASELINES, FIELD_FOLD_ENDS, FIELD_TEST, GRIDS, ROOT,
                               field_blocks, field_data, field_predict, run_parallel,
                               select_best)


def job(spec: dict) -> dict:
    hist, act, out, dates, level, scale, target = field_data()
    blocks = field_blocks(spec["test_start"])
    tr, va = blocks["train"], blocks["validation"]
    model = FlatModel(spec["family"], seed=242 + spec["fold"], params=spec["params"],
                      features="engineered").fit(hist[tr], act[tr], target[tr])
    bounds = (float(np.quantile(out[tr], 0.025)), float(np.quantile(out[tr], 0.975)))
    test = np.arange(spec["test_start"], spec["test_end"])
    row = {"fold": spec["fold"], "family": f"{spec['family']} (engineered)",
           "config_id": spec["config_id"], "params": json.dumps(spec["params"], default=list),
           "val_nmae_pct": metrics(out[va], field_predict(model, va, bounds))["nmae_pct"]}
    row.update(metrics(out[test], field_predict(model, test, bounds)))
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "actrn_v2"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    specs = [{"fold": f, "family": fam, "config_id": cid, "params": p,
              "test_start": end - FIELD_TEST, "test_end": end}
             for f, end in enumerate(FIELD_FOLD_ENDS) for fam in BASELINES
             for cid, p in enumerate(GRIDS["field"][fam])]
    tuning = pd.DataFrame(run_parallel(job, specs, args.workers))
    selected = select_best(tuning, ["fold", "family"])
    out = Path(args.out)
    selected.to_csv(out / "exploratory_field_engineered_metrics.csv", index=False)
    print(selected.groupby("family")[["nmae_pct", "nrmse_pct", "r2"]].mean().round(3))


if __name__ == "__main__":
    main()
