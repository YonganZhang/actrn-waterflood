"""Paired statistics for the ACTRN benchmark (ACTRN against every other model family).

Decision contexts are nested in test reservoirs, which are nested in independent repeats.  The
primary test is the exact Wilcoxon signed-rank test over repeat means; uncertainty intervals come
from a hierarchical bootstrap that resamples repeats and then reservoirs within each repeat
(the four decision ages of a reservoir are averaged first).  A context-level Wilcoxon test is
reported as a secondary, less conservative summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

BOOTSTRAP_SEED = 20261016
BOOTSTRAP_DRAWS = 10_000
DECISION_METRICS = ("decision_regret_pct", "gain_capture_pct")
HIGHER_IS_BETTER = {"gain_capture_pct", "optimal_selection_pct", "top_decile_selection_pct", "r2"}
SUMMARY_METRICS = ("nmae_pct", "nrmse_pct", "r2", "decision_regret", "decision_regret_pct",
                   "gain_capture_pct", "optimal_selection_pct", "top_decile_selection_pct")


def hierarchical_bootstrap(reservoir_means: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap means of a (repeats x reservoirs) array: resample repeats, then reservoirs."""
    n_rep, n_res = reservoir_means.shape
    rep = rng.integers(0, n_rep, (BOOTSTRAP_DRAWS, n_rep))
    res = rng.integers(0, n_res, (BOOTSTRAP_DRAWS, n_rep, n_res))
    return reservoir_means[rep[:, :, None], res].mean(axis=(1, 2))


def _exact_wilcoxon(diff: np.ndarray) -> float:
    if not np.any(diff != 0):
        return 1.0
    return float(stats.wilcoxon(diff, method="exact" if len(diff) <= 25 else "auto").pvalue)


def _better(diff, metric):
    """Count of ACTRN-favourable differences (ACTRN minus other)."""
    return int(np.sum(diff > 0) if metric in HIGHER_IS_BETTER else np.sum(diff < 0))


def compare(decisions: pd.DataFrame, runs: pd.DataFrame, other: str, rng) -> dict:
    """ACTRN minus ``other`` for one benchmark setting."""
    entry = {}
    rep = runs.pivot(index="repeat", columns="family")
    for metric in ("nmae_pct",) + DECISION_METRICS + ("optimal_selection_pct",):
        diff = (rep[metric]["ACTRN"] - rep[metric][other]).dropna().to_numpy()
        entry[metric] = {"mean_actrn": float(rep[metric]["ACTRN"].mean()),
                         "mean_other": float(rep[metric][other].mean()),
                         "mean_diff": float(diff.mean()),
                         "repeats_actrn_better": _better(diff, metric),
                         "n_repeats": int(len(diff)),
                         "wilcoxon_exact_p": _exact_wilcoxon(diff)}
    for metric in DECISION_METRICS:
        ctx = decisions.pivot_table(index=["repeat", "reservoir_id", "decision_age"],
                                    columns="family", values=metric)
        d = (ctx["ACTRN"] - ctx[other]).dropna()
        by_res = d.groupby(level=["repeat", "reservoir_id"]).mean().unstack("reservoir_id")
        boot = hierarchical_bootstrap(by_res.to_numpy(), rng)
        values = d.to_numpy()
        entry[metric]["hierarchical95"] = [float(np.quantile(boot, 0.025)),
                                           float(np.quantile(boot, 0.975))]
        entry[metric]["context"] = {
            "n": int(len(values)), "actrn_better": _better(values, metric),
            "other_better": int(len(values) - _better(values, metric) - np.sum(values == 0)),
            "ties": int(np.sum(values == 0)),
            "wilcoxon_p": float(stats.wilcoxon(values).pvalue) if np.any(values != 0) else 1.0}
    return entry


def paired_statistics(runs: pd.DataFrame, decisions: pd.DataFrame,
                      keys=("population", "response")) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {}
    for setting, rr in runs.groupby(list(keys)):
        name = "/".join(setting)
        dd = decisions.merge(rr[list(keys) + ["repeat", "family"]].drop_duplicates(),
                             on=list(keys) + ["repeat", "family"])
        out[name] = {other: compare(dd, rr, other, rng)
                     for other in rr["family"].unique() if other != "ACTRN"}
    return out


def summarize(runs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    cols = [c for c in SUMMARY_METRICS if c in runs]
    agg = runs.groupby(keys, sort=False)[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()
