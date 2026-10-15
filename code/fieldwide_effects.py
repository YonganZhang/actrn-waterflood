"""P17 fieldwide dynamic effects of F-4 injection.

This module replaces the invalid pairwise F4->F12 DML and permanent-shutdown
calculation.  The treatment is weekly F-4 injected volume.  Outcomes are
strictly future, additive production volumes from all producers.  Nuisance
models use only measurements available by t-1 plus concurrent F-5 injection,
and are cross-fitted with contiguous test blocks and an overlap purge.

The reported coefficient is an orthogonal partially linear dynamic projection
of cumulative field oil over weeks t+1..t+8 on F-4 injection in week t. A
causal reading additionally requires sequential exchangeability and diagnostic
stability; it is never a permanent-shutdown effect.
"""
from __future__ import annotations

import os

# This project runs on a many-core host.  For these small weekly folds, allowing
# every nuisance fit to create hundreds of OpenMP/BLAS threads is more than two
# orders of magnitude slower than a single native thread and can make a rerun
# appear hung.  Set defaults before importing NumPy/scikit-learn; callers may
# still override them explicitly.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

F4 = "15/9-F-4"
F5_INJ = "15/9-F-5_inj"
PRODUCERS = [
    "15/9-F-1 C",
    "15/9-F-11",
    "15/9-F-12",
    "15/9-F-14",
    "15/9-F-15 D",
]
WELL_SLUG = {
    "15/9-F-1 C": "F1C",
    "15/9-F-11": "F11",
    "15/9-F-12": "F12",
    "15/9-F-14": "F14",
    "15/9-F-15 D": "F15D",
    "15/9-F-5_op": "F5OP",
}
VOLUME_MAP = {
    "oil": "BORE_OIL_VOL",
    "water": "BORE_WAT_VOL",
    "gas": "BORE_GAS_VOL",
}
HISTORY_LAGS = (1, 2, 4, 8)
MAX_HORIZON = 8
N_SPLITS = 5
PURGE_WEEKS = 16
HAC_LAGS = 12
MAX_FEATURE_MISSING_RATE = 0.50


@dataclass(frozen=True)
class Design:
    weekly: pd.DataFrame
    X: pd.DataFrame
    treatment: pd.Series
    outcomes: dict[str, pd.DataFrame]
    dates: pd.DatetimeIndex
    metadata: dict


def _week_end(date: pd.Timestamp) -> pd.Timestamp:
    return date.to_period("W-SUN").end_time.normalize()


def _daily_wide(
    panel: pd.DataFrame,
    roles: Iterable[str],
    value: str,
    wells: list[str],
    daily_index: pd.DatetimeIndex,
    fill_volume_zero: bool,
) -> pd.DataFrame:
    sub = panel[panel["role"].isin(list(roles))]
    wide = sub.pivot_table(
        index="DATEPRD", columns="well_id", values=value, aggfunc="first"
    )
    wide = wide.reindex(index=daily_index, columns=wells)
    if fill_volume_zero:
        # The preprocessing layer represents off-stream daily volumes as NaN.
        # A field volume contract needs those physical volumes to be zero.
        wide = wide.fillna(0.0)
    return wide


def build_weekly_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    panel = panel.copy()
    panel["DATEPRD"] = pd.to_datetime(panel["DATEPRD"])
    daily_index = pd.date_range(
        panel["DATEPRD"].min().normalize(), panel["DATEPRD"].max().normalize(), freq="D"
    )

    inj = _daily_wide(
        panel, ["injector"], "BORE_WI_VOL", [F4, F5_INJ], daily_index, True
    ).resample("W-SUN").sum()
    inj = inj.rename(columns={F4: "inj_F4_m3", F5_INJ: "inj_F5_m3"})

    pieces = [inj]

    inj_hours = _daily_wide(
        panel, ["injector"], "ON_STREAM_HRS", [F4, F5_INJ], daily_index, True
    ).resample("W-SUN").sum()
    inj_hours = inj_hours.rename(
        columns={F4: "inj_hours_F4", F5_INJ: "inj_hours_F5"}
    )
    pieces.append(inj_hours)

    inj_dp_choke = _daily_wide(
        panel, ["injector"], "DP_CHOKE_SIZE", [F4, F5_INJ], daily_index, False
    ).resample("W-SUN").mean()
    inj_dp_choke = inj_dp_choke.rename(
        columns={F4: "inj_dp_choke_F4", F5_INJ: "inj_dp_choke_F5"}
    )
    pieces.append(inj_dp_choke)
    for fluid, source in VOLUME_MAP.items():
        wide = _daily_wide(
            panel,
            ["producer", "producer_minor"],
            source,
            PRODUCERS,
            daily_index,
            True,
        ).resample("W-SUN").sum()
        wide.columns = [f"{fluid}__{WELL_SLUG[w]}" for w in wide.columns]
        pieces.append(wide)

    hours = _daily_wide(
        panel,
        ["producer", "producer_minor"],
        "ON_STREAM_HRS",
        PRODUCERS,
        daily_index,
        True,
    ).resample("W-SUN").sum()
    hours.columns = [f"hours__{WELL_SLUG[w]}" for w in hours.columns]
    pieces.append(hours)

    pressure = _daily_wide(
        panel,
        ["producer", "producer_minor"],
        "AVG_DOWNHOLE_PRESSURE",
        PRODUCERS,
        daily_index,
        False,
    ).resample("W-SUN").mean()
    pressure.columns = [f"pressure__{WELL_SLUG[w]}" for w in pressure.columns]
    pieces.append(pressure)

    operator_sensor_map = {
        "choke": "AVG_CHOKE_SIZE_P",
        "whp": "AVG_WHP_P",
        "dp_tubing": "AVG_DP_TUBING",
        "dp_choke": "DP_CHOKE_SIZE",
    }
    for short, source in operator_sensor_map.items():
        sensor = _daily_wide(
            panel,
            ["producer", "producer_minor"],
            source,
            PRODUCERS,
            daily_index,
            False,
        ).resample("W-SUN").mean()
        sensor.columns = [f"{short}__{WELL_SLUG[w]}" for w in sensor.columns]
        pieces.append(sensor)

    weekly = pd.concat(pieces, axis=1).sort_index()
    for fluid in VOLUME_MAP:
        cols = [c for c in weekly if c.startswith(f"{fluid}__")]
        weekly[f"field_{fluid}_m3"] = weekly[cols].sum(axis=1)

    f4_rows = panel[(panel["role"] == "injector") & (panel["well_id"] == F4)]
    f4_positive = f4_rows[f4_rows["BORE_WI_VOL"].fillna(0.0) > 0.0]
    f4_start = _week_end(f4_positive["DATEPRD"].min())
    f4_end = _week_end(f4_positive["DATEPRD"].max())
    reconstruction = {
        "daily_date_range": [str(daily_index.min().date()), str(daily_index.max().date())],
        "f4_positive_injection_window": [str(f4_start.date()), str(f4_end.date())],
        "volume_missing_contract": "off-stream or absent well-day -> 0 m3",
        "sensor_missing_contract": "retain NaN; impute within training fold",
        "producers": PRODUCERS,
        "f5_role_change": "F5 changes from injector to producer in 2016; primary outcomes must not cross that regime boundary",
    }
    return weekly, reconstruction


def build_design(max_horizon: int = MAX_HORIZON) -> Design:
    panel = pd.read_parquet(PROC / "volve_panel.parquet")
    weekly, reconstruction = build_weekly_panel(panel)

    feature_parts: dict[str, pd.Series] = {
        "inj_F5_m3__t0_cotreatment": weekly["inj_F5_m3"]
    }
    history_sources = [
        c
        for c in weekly.columns
        if c.startswith(
            (
                "inj_",
                "oil__",
                "water__",
                "gas__",
                "hours__",
                "pressure__",
                "choke__",
                "whp__",
                "dp_tubing__",
                "dp_choke__",
            )
        )
    ]
    for lag in HISTORY_LAGS:
        for col in history_sources:
            feature_parts[f"{col}__lag{lag}"] = weekly[col].shift(lag)

    trend = np.linspace(0.0, 1.0, len(weekly))
    feature_parts["time_trend"] = pd.Series(trend, index=weekly.index)
    feature_parts["time_trend_sq"] = pd.Series(trend**2, index=weekly.index)
    feature_parts["calendar_sin"] = pd.Series(
        np.sin(2 * np.pi * weekly.index.dayofyear / 365.25), index=weekly.index
    )
    feature_parts["calendar_cos"] = pd.Series(
        np.cos(2 * np.pi * weekly.index.dayofyear / 365.25), index=weekly.index
    )
    X_all = pd.DataFrame(feature_parts, index=weekly.index)

    f4_rows = panel[(panel["role"] == "injector") & (panel["well_id"] == F4)]
    f4_positive = f4_rows[f4_rows["BORE_WI_VOL"].fillna(0.0) > 0.0]
    first_treatment_week = _week_end(f4_positive["DATEPRD"].min())
    last_treatment_week = _week_end(f4_positive["DATEPRD"].max())
    f5op_rows = panel[
        (panel["role"] == "producer_minor") & (panel["well_id"] == "15/9-F-5_op")
    ]
    first_f5_production_week = _week_end(f5op_rows["DATEPRD"].min())
    start_pos = weekly.index.get_indexer([first_treatment_week], method="nearest")[0]
    end_pos = weekly.index.get_indexer([last_treatment_week], method="nearest")[0]
    f5_role_change_pos = weekly.index.get_indexer(
        [first_f5_production_week], method="nearest"
    )[0]
    # Keep the co-treatment regime stable: the last outcome week must precede
    # F5's switch from injector to producer.
    stable_regime_end_pos = f5_role_change_pos - max_horizon - 1
    analysis_pos = np.arange(
        start_pos + max(HISTORY_LAGS),
        min(end_pos, stable_regime_end_pos, len(weekly) - max_horizon - 1) + 1,
    )
    dates = pd.DatetimeIndex(weekly.index[analysis_pos], name="date")

    X_candidate = X_all.loc[dates].copy()
    missing_rate = X_candidate.isna().mean()
    all_missing = set(missing_rate[missing_rate == 1.0].index)
    mostly_missing = set(
        missing_rate[missing_rate > MAX_FEATURE_MISSING_RATE].index
    )
    constant = set(
        col
        for col in X_candidate.columns
        if X_candidate[col].dropna().nunique() <= 1
    )
    excluded_features = all_missing | mostly_missing | constant
    X = X_candidate.drop(columns=sorted(excluded_features)).copy()
    treatment = weekly.loc[dates, "inj_F4_m3"].rename("A_F4_m3")

    outcomes: dict[str, pd.DataFrame] = {}
    for fluid in VOLUME_MAP:
        cols = [f"{fluid}__{WELL_SLUG[w]}" for w in PRODUCERS]
        base = weekly[cols]
        for horizon in range(1, max_horizon + 1):
            cumulative = sum(base.shift(-k) for k in range(1, horizon + 1))
            cumulative = cumulative.loc[dates].copy()
            cumulative.columns = [WELL_SLUG[w] for w in PRODUCERS]
            outcomes[f"{fluid}_cum_h{horizon}"] = cumulative

    metadata = {
        **reconstruction,
        "analysis_date_range": [str(dates.min().date()), str(dates.max().date())],
        "stable_cotreatment_regime": (
            "F4 weekly treatment with F5 in its injector role; all outcome weeks precede F5 production"
        ),
        "first_excluded_f5_production_week": str(first_f5_production_week.date()),
        "n_weeks": int(len(dates)),
        "n_features": int(X.shape[1]),
        "feature_screening": {
            "raw_feature_count": int(X_candidate.shape[1]),
            "retained_feature_count": int(X.shape[1]),
            "maximum_allowed_missing_rate": MAX_FEATURE_MISSING_RATE,
            "all_missing_features": sorted(all_missing),
            "mostly_missing_features": sorted(mostly_missing - all_missing),
            "constant_features": sorted(constant - all_missing),
            "retained_missing_rate_max": float(X.isna().mean().max()),
            "retained_missing_rate_median": float(X.isna().mean().median()),
        },
        "history_lags_weeks": list(HISTORY_LAGS),
        "max_feature_time_offset_weeks": -1,
        "outcome_time_offsets_weeks": [1, max_horizon],
        "treatment_unit": "m3/week",
        "outcome_unit": "m3 cumulative volume",
        "primary_endpoint": f"field_oil_cumulative_h{max_horizon}",
        "estimand_class": (
            "orthogonal partially linear dynamic projection; under heterogeneous effects, "
            "a residual-treatment-variance-weighted projection, not an unweighted average derivative"
        ),
        "raw_sample_window_offsets_weeks": [-max(HISTORY_LAGS), max_horizon],
        "required_minimum_train_test_center_distance_weeks": max(HISTORY_LAGS)
        + max_horizon
        + 1,
    }
    return Design(weekly, X, treatment, outcomes, dates, metadata)


def make_purged_block_folds(
    n: int, n_splits: int = N_SPLITS, purge: int = PURGE_WEEKS
) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n)
    tests = np.array_split(indices, n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for test in tests:
        lo = max(0, int(test[0]) - purge)
        hi = min(n - 1, int(test[-1]) + purge)
        train = indices[(indices < lo) | (indices > hi)]
        if len(train) == 0:
            raise ValueError("Purge leaves an empty training fold")
        folds.append((train, test))
    return folds


def make_learner(kind: str = "hgb", max_iter: int = 300):
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.04,
            max_iter=max_iter,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=10.0,
            early_stopping=False,
            random_state=42,
        )
    if kind == "ridge":
        x_model = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True
                    ),
                ),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=10.0)),
            ]
        )
        # Scaling the outcome prevents the fixed alpha from depending on whether
        # the target is gas, water, or oil volume.
        return TransformedTargetRegressor(
            regressor=x_model, transformer=StandardScaler()
        )
    if kind == "ols":
        x_model = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True
                    ),
                ),
                ("scale", StandardScaler()),
                ("ols", LinearRegression()),
            ]
        )
        return TransformedTargetRegressor(
            regressor=x_model, transformer=StandardScaler()
        )
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=15,
            max_features=0.7,
            n_jobs=1,
            random_state=42,
        )
    raise ValueError(f"Unknown learner: {kind}")


def learner_spec(kind: str, max_iter: int = 300) -> dict:
    """Return the fixed, publication-facing nuisance-model specification."""
    if kind == "hgb":
        return {
            "kind": "HistGradientBoostingRegressor",
            "learning_rate": 0.04,
            "max_iter": max_iter,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 10,
            "l2_regularization": 10.0,
            "early_stopping": False,
            "random_state": 42,
        }
    if kind == "ridge":
        return {
            "kind": "median-imputed standardized Ridge",
            "alpha": 10.0,
            "standardized_target": True,
        }
    if kind == "ols":
        return {
            "kind": "median-imputed standardized LinearRegression",
            "standardized_target": True,
        }
    if kind == "rf":
        return {
            "kind": "RandomForestRegressor",
            "n_estimators": 300,
            "max_depth": 6,
            "min_samples_leaf": 15,
            "max_features": 0.7,
            "random_state": 42,
        }
    raise ValueError(kind)


def crossfit_predict(
    X: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    learner,
) -> np.ndarray:
    pred = np.full(len(y), np.nan, dtype=float)
    for train, test in folds:
        model = clone(learner)
        model.fit(X.iloc[train], y[train])
        pred[test] = model.predict(X.iloc[test])
    if not np.isfinite(pred).all():
        raise RuntimeError("Cross-fitting did not produce finite predictions for every row")
    return pred


def hac_partialling_out(
    treatment_resid: np.ndarray,
    outcome_resid: np.ndarray,
    max_lag: int = HAC_LAGS,
) -> dict:
    v = np.asarray(treatment_resid, dtype=float)
    u = np.asarray(outcome_resid, dtype=float)
    denominator = float(v @ v)
    if denominator <= 0:
        raise ValueError("Treatment residual has no variation")
    theta = float(v @ u / denominator)
    score = v * (u - theta * v)
    n = len(v)
    bandwidth = min(max_lag, n - 1)
    long_run = float(np.dot(score, score) / n)
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        gamma = float(np.dot(score[lag:], score[:-lag]) / n)
        long_run += 2.0 * weight * gamma
    q = float(np.mean(v**2))
    se = float(np.sqrt(max(long_run, 0.0) / n) / q)
    z = theta / se if se > 0 else np.nan
    p = float(2.0 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return {
        "theta": theta,
        "se_hac": se,
        "ci95_low": theta - 1.96 * se,
        "ci95_high": theta + 1.96 * se,
        "z": z,
        "p_value": p,
        "hac_lags": int(bandwidth),
        "n": int(n),
        "score_long_run_variance": long_run,
        "residual_treatment_second_moment": q,
    }


def fold_effects(
    v: np.ndarray, u: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]
) -> list[float]:
    values = []
    for _, test in folds:
        den = float(v[test] @ v[test])
        values.append(float(v[test] @ u[test] / den) if den > 0 else np.nan)
    return values


def fold_r2_values(
    observed: np.ndarray,
    predicted: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    values = []
    for _, test in folds:
        if np.var(observed[test]) <= 0:
            values.append(np.nan)
        else:
            values.append(float(r2_score(observed[test], predicted[test])))
    return values


def center_within_test_folds(
    values: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    centered = np.asarray(values, dtype=float).copy()
    for _, test in folds:
        centered[test] -= np.mean(centered[test])
    return centered


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    if not mask.any():
        return out
    vals = p[mask]
    order = np.argsort(vals)
    ranked = vals[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    out[np.where(mask)[0]] = restored
    return out


def _estimate_oil_residuals(
    design: Design,
    horizon: int,
    folds: list[tuple[np.ndarray, np.ndarray]],
    learner,
    treatment_pred: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    observed: dict[str, np.ndarray] = {}
    residuals: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    frame = design.outcomes[f"oil_cum_h{horizon}"]
    for well in frame.columns:
        y = frame[well].to_numpy(dtype=float)
        pred = crossfit_predict(design.X, y, folds, learner)
        observed[well] = y
        predictions[well] = pred
        residuals[well] = y - pred
    field_observed = np.sum(np.vstack(list(observed.values())), axis=0)
    field_prediction = np.sum(np.vstack(list(predictions.values())), axis=0)
    return observed, residuals, field_observed, field_prediction


def _effect_row(
    outcome: str,
    fluid: str,
    scope: str,
    well: str,
    horizon: int,
    result: dict,
    folds_values: list[float],
    learner_kind: str,
) -> dict:
    clean = {k: v for k, v in result.items() if not k.startswith("_")}
    return {
        "outcome": outcome,
        "fluid": fluid,
        "scope": scope,
        "well": well,
        "horizon_weeks": horizon,
        "estimand": (
            "orthogonal_partially_linear_projection_of_cumulative_weeks_"
            "t_plus_1_to_h_on_F4_at_t"
        ),
        "learner": learner_kind,
        "fold_theta_min": float(np.nanmin(folds_values)),
        "fold_theta_max": float(np.nanmax(folds_values)),
        "fold_theta_sd": float(np.nanstd(folds_values, ddof=1)),
        **clean,
    }


def estimate_all_effects(
    design: Design,
    learner_kind: str = "hgb",
    n_splits: int = N_SPLITS,
    purge: int = PURGE_WEEKS,
    max_iter: int = 300,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    folds = make_purged_block_folds(len(design.dates), n_splits, purge)
    learner = make_learner(learner_kind, max_iter=max_iter)
    treatment = design.treatment.to_numpy(dtype=float)
    treatment_pred = crossfit_predict(design.X, treatment, folds, learner)
    v = treatment - treatment_pred

    rows: list[dict] = []
    primary_payload: dict = {}
    cumulative_field_oil: dict[int, float] = {}
    for horizon in range(1, MAX_HORIZON + 1):
        print(f"  fitting horizon {horizon}/{MAX_HORIZON}", flush=True)
        observed, oil_resid, field_y, field_pred = _estimate_oil_residuals(
            design, horizon, folds, learner, treatment_pred
        )
        field_resid = field_y - field_pred
        for well, y_resid in oil_resid.items():
            res = hac_partialling_out(v, y_resid)
            rows.append(
                _effect_row(
                    f"oil_{well}",
                    "oil",
                    "producer",
                    well,
                    horizon,
                    res,
                    fold_effects(v, y_resid, folds),
                    learner_kind,
                )
            )
        field_res = hac_partialling_out(v, field_resid)
        cumulative_field_oil[horizon] = field_res["theta"]
        rows.append(
            _effect_row(
                "field_oil",
                "oil",
                "field",
                "ALL",
                horizon,
                field_res,
                fold_effects(v, field_resid, folds),
                learner_kind,
            )
        )
        if horizon == MAX_HORIZON:
            primary_payload = {
                "field_y": field_y,
                "field_pred": field_pred,
                "field_resid": field_resid,
                "oil_observed": observed,
                "oil_residuals": oil_resid,
                "result": field_res,
            }

        for fluid in ("water", "gas"):
            frame = design.outcomes[f"{fluid}_cum_h{horizon}"]
            y = frame.sum(axis=1).to_numpy(dtype=float)
            pred = crossfit_predict(design.X, y, folds, learner)
            y_resid = y - pred
            res = hac_partialling_out(v, y_resid)
            rows.append(
                _effect_row(
                    f"field_{fluid}",
                    fluid,
                    "field",
                    "ALL",
                    horizon,
                    res,
                    fold_effects(v, y_resid, folds),
                    learner_kind,
                )
            )

    effects = pd.DataFrame(rows)
    effects["q_value_bh"] = bh_adjust(effects["p_value"].to_numpy())

    impulse_rows = []
    previous = 0.0
    for horizon in range(1, MAX_HORIZON + 1):
        current = cumulative_field_oil[horizon]
        impulse_rows.append(
            {
                "fluid": "oil",
                "scope": "field",
                "exact_week": horizon,
                "theta_incremental": current - previous,
                "derivation": "difference_of_cumulative_orthogonal_local_projections",
            }
        )
        previous = current
    impulse = pd.DataFrame(impulse_rows)

    fold_id = np.full(len(design.dates), -1, dtype=int)
    fold_rows = []
    for k, (train, test) in enumerate(folds):
        fold_id[test] = k
        fold_rows.append(
            {
                "fold": k,
                "test_start": str(design.dates[test[0]].date()),
                "test_end": str(design.dates[test[-1]].date()),
                "n_test": int(len(test)),
                "n_train": int(len(train)),
                "purge_weeks": int(purge),
                "minimum_index_distance": int(
                    np.min(np.abs(train[:, None] - test[None, :]))
                ),
            }
        )
    folds_frame = pd.DataFrame(
        {"date": design.dates, "fold": fold_id, "A_F4_m3": treatment}
    )

    treatment_r2 = float(r2_score(treatment, treatment_pred))
    residual_q = np.quantile(v, [0.01, 0.05, 0.5, 0.95, 0.99])
    shifted = 0.9 * treatment - treatment_pred
    shift_coverage = float(np.mean((shifted >= residual_q[0]) & (shifted <= residual_q[-1])))
    fold_shift_support = []
    for k, (_, test) in enumerate(folds):
        fold_q = np.quantile(v[test], [0.01, 0.99])
        coverage = float(
            np.mean((shifted[test] >= fold_q[0]) & (shifted[test] <= fold_q[1]))
        )
        fold_shift_support.append(
            {
                "fold": int(k),
                "residual_1pct_m3": float(fold_q[0]),
                "residual_99pct_m3": float(fold_q[1]),
                "shift_coverage": coverage,
                "passes_90pct_gate": bool(coverage >= 0.90),
            }
        )
    balance = []
    for col in design.X.columns:
        x = design.X[col].to_numpy(dtype=float)
        keep = np.isfinite(x)
        if keep.sum() < 10 or np.nanstd(x[keep]) == 0:
            continue
        corr = float(np.corrcoef(v[keep], x[keep])[0, 1])
        if np.isfinite(corr):
            balance.append((col, corr))
    balance.sort(key=lambda item: abs(item[1]), reverse=True)

    max_abs_balance = float(abs(balance[0][1])) if balance else np.nan
    primary_folds = fold_effects(v, primary_payload["field_resid"], folds)
    treatment_fold_r2 = fold_r2_values(treatment, treatment_pred, folds)
    outcome_fold_r2 = fold_r2_values(
        primary_payload["field_y"], primary_payload["field_pred"], folds
    )
    centered_result = hac_partialling_out(
        center_within_test_folds(v, folds),
        center_within_test_folds(primary_payload["field_resid"], folds),
    )

    placebo_y = (
        design.weekly["field_oil_m3"].shift(1).loc[design.dates].to_numpy(dtype=float)
    )
    placebo_pred = crossfit_predict(design.X, placebo_y, folds, learner)
    placebo_result = hac_partialling_out(v, placebo_y - placebo_pred)
    distant_placebo_y = (
        design.weekly["field_oil_m3"].shift(12).loc[design.dates].to_numpy(dtype=float)
    )
    distant_placebo_pred = crossfit_predict(
        design.X, distant_placebo_y, folds, learner
    )
    distant_placebo_result = hac_partialling_out(
        v, distant_placebo_y - distant_placebo_pred
    )
    hac_bandwidth_sensitivity = {
        str(lag): hac_partialling_out(
            v, primary_payload["field_resid"], max_lag=lag
        )
        for lag in (4, 8, 12, 20, 24)
    }
    fold_denominator = np.asarray(
        [float(v[test] @ v[test]) for _, test in folds], dtype=float
    )
    fold_numerator = np.asarray(
        [
            float(v[test] @ primary_payload["field_resid"][test])
            for _, test in folds
        ],
        dtype=float,
    )
    positive_numerator = np.clip(fold_numerator, 0.0, None)
    fold_sign_stable = bool(
        all(value > 0 for value in primary_folds)
        or all(value < 0 for value in primary_folds)
    )
    credibility_checks = {
        "treatment_crossfit_r2_nonnegative": bool(treatment_r2 >= 0.0),
        "residual_effective_sample_size_at_least_50": bool(
            np.sum(v**2) ** 2 / np.sum(v**4) >= 50.0
        ),
        "max_abs_residual_feature_correlation_at_most_0_2": bool(
            max_abs_balance <= 0.2
        ),
        "primary_fold_sign_stable": fold_sign_stable,
        "all_treatment_fold_r2_nonnegative": bool(
            all(np.isfinite(value) and value >= 0.0 for value in treatment_fold_r2)
        ),
        "all_primary_outcome_fold_r2_nonnegative": bool(
            all(np.isfinite(value) and value >= 0.0 for value in outcome_fold_r2)
        ),
        "fold_centered_second_stage_same_sign": bool(
            np.sign(centered_result["theta"])
            == np.sign(primary_payload["result"]["theta"])
        ),
        "pretreatment_placebo_ci_contains_zero": bool(
            placebo_result["ci95_low"] <= 0.0 <= placebo_result["ci95_high"]
        ),
        "distant_pretreatment_placebo_ci_contains_zero": bool(
            distant_placebo_result["ci95_low"]
            <= 0.0
            <= distant_placebo_result["ci95_high"]
        ),
        "all_fold_shift_support_at_least_90pct": bool(
            all(row["passes_90pct_gate"] for row in fold_shift_support)
        ),
    }
    diagnostics = {
        "interpretation_status": "OBSERVED_PROJECTION_DIAGNOSTIC_ONLY",
        "learner": learner_kind,
        "nuisance_model_spec": learner_spec(learner_kind, max_iter=max_iter),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_splits": n_splits,
        "purge_weeks": purge,
        "folds": fold_rows,
        "treatment": {
            "observed_min_m3": float(np.min(treatment)),
            "observed_max_m3": float(np.max(treatment)),
            "observed_zero_weeks": int(np.sum(treatment == 0)),
            "observed_quantiles_m3": {
                str(q): float(np.quantile(treatment, q))
                for q in (0.01, 0.05, 0.5, 0.95, 0.99)
            },
            "crossfit_r2": treatment_r2,
            "residual_sd_m3": float(np.std(v, ddof=1)),
            "residual_quantiles_m3": {
                str(q): float(value)
                for q, value in zip((0.01, 0.05, 0.5, 0.95, 0.99), residual_q)
            },
            "residual_effective_sample_size": float(
                np.sum(v**2) ** 2 / np.sum(v**4)
            ),
        },
        "modified_policy_10pct": {
            "residual_support_1pct_99pct_coverage": shift_coverage,
            "passes_90pct_gate": bool(shift_coverage >= 0.90),
            "policy_interpretation": "one-week local shift only; repeated policy is not identified",
            "fold_conditional_support": fold_shift_support,
        },
        "residual_balance_top10": [
            {"feature": col, "corr_with_treatment_residual": corr}
            for col, corr in balance[:10]
        ],
        "max_abs_residual_feature_correlation": max_abs_balance,
        "fold_nuisance_diagnostics": [
            {
                "fold": int(k),
                "treatment_r2": treatment_fold_r2[k],
                "primary_outcome_r2": outcome_fold_r2[k],
                "primary_theta": primary_folds[k],
                "residual_treatment_squared_weight": float(
                    fold_denominator[k] / fold_denominator.sum()
                ),
                "second_stage_numerator": float(fold_numerator[k]),
                "positive_numerator_weight": float(
                    positive_numerator[k] / positive_numerator.sum()
                )
                if positive_numerator.sum() > 0
                else None,
            }
            for k in range(len(folds))
        ],
        "fold_centered_second_stage": {
            k: value for k, value in centered_result.items() if not k.startswith("_")
        },
        "pretreatment_placebo_field_oil_lag1": {
            k: value for k, value in placebo_result.items() if not k.startswith("_")
        },
        "distant_pretreatment_placebo_field_oil_lag12": {
            k: value
            for k, value in distant_placebo_result.items()
            if not k.startswith("_")
        },
        "primary_hac_bandwidth_sensitivity": hac_bandwidth_sensitivity,
        "credibility_checks": credibility_checks,
        "primary_reporting_status": (
            "SUPPORTED_PROJECTION_ESTIMATE"
            if all(credibility_checks.values())
            else "UNSTABLE_EXPLORATORY_ONLY"
        ),
        "primary_outcome_crossfit_r2": float(
            r2_score(primary_payload["field_y"], primary_payload["field_pred"])
        ),
        "primary_field_effect": {
            k: v
            for k, v in primary_payload["result"].items()
            if not k.startswith("_")
        },
        "permanent_shutdown_supported": False,
        "permanent_shutdown_reason": (
            "The estimand is a one-week local perturbation. A permanent intervention would "
            "recursively change future state and policy and is not identified by these local projections."
        ),
    }

    predictions = pd.DataFrame(
        {
            "date": design.dates,
            "fold": fold_id,
            "A_F4_m3": treatment,
            "A_F4_pred": treatment_pred,
            "A_F4_resid": v,
            "field_oil_cum_h8_m3": primary_payload["field_y"],
            "field_oil_pred": primary_payload["field_pred"],
            "field_oil_resid": primary_payload["field_resid"],
        }
    )
    effects["interpretation_status"] = diagnostics["primary_reporting_status"]
    impulse["interpretation_status"] = diagnostics["primary_reporting_status"]
    return effects, impulse, diagnostics, {
        "folds": folds_frame,
        "predictions": predictions,
        "treatment_pred": treatment_pred,
        "treatment_resid": v,
    }


def estimate_primary_sensitivity(
    design: Design, learner_kind: str, purge: int, max_iter: int = 300
) -> dict:
    folds = make_purged_block_folds(len(design.dates), N_SPLITS, purge)
    learner = make_learner(learner_kind, max_iter=max_iter)
    treatment = design.treatment.to_numpy(dtype=float)
    treatment_pred = crossfit_predict(design.X, treatment, folds, learner)
    v = treatment - treatment_pred
    _, _, field_y, field_pred = _estimate_oil_residuals(
        design, MAX_HORIZON, folds, learner, treatment_pred
    )
    result = hac_partialling_out(v, field_y - field_pred)
    return {
        "learner": learner_kind,
        "purge_weeks": purge,
        **{k: val for k, val in result.items() if not k.startswith("_")},
        "treatment_crossfit_r2": float(r2_score(treatment, treatment_pred)),
        "outcome_crossfit_r2": float(r2_score(field_y, field_pred)),
        "fold_theta": fold_effects(v, field_y - field_pred, folds),
    }


def economics_gate(reporting_status: str) -> pd.DataFrame:
    """Emit a machine-readable stop record, never an unsupported cash table.

    The production projection lets future F4/F5 operation follow its natural
    policy, whereas a current-week injection saving accounts for only one cost
    item. Those paths are not economically commensurate even if the production
    projection were statistically stable.
    """
    return pd.DataFrame(
        [
            {
                "interpretation_status": "WITHHELD_CANNOT_USE_FOR_DECISION",
                "projection_status": reporting_status,
                "deterministic_economic_table_emitted": False,
                "reason": (
                    "Observed production projection failed credibility gates, and its "
                    "natural future-injection policy is not aligned with a one-week-only "
                    "injection-cost change. No operating-margin or NPV claim is identified."
                ),
            }
        ]
    )


def write_outputs(
    design: Design,
    effects: pd.DataFrame,
    impulse: pd.DataFrame,
    diagnostics: dict,
    auxiliary: dict,
    sensitivity: list[dict],
) -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    design.weekly.to_parquet(PROC / "fieldwide_weekly.parquet")
    effects.to_csv(PROC / "fieldwide_effects.csv", index=False)
    impulse.to_csv(PROC / "fieldwide_impulse.csv", index=False)
    auxiliary["folds"].to_csv(PROC / "fieldwide_folds.csv", index=False)
    auxiliary["predictions"].to_csv(PROC / "fieldwide_oof_predictions.csv", index=False)
    economics_gate(diagnostics["primary_reporting_status"]).to_csv(
        PROC / "fieldwide_economics.csv", index=False
    )

    design_meta = {
        "interpretation_status": "OBSERVED_PROJECTION_DIAGNOSTIC_ONLY",
        "estimand": (
            "orthogonal partially linear projection of fieldwide cumulative production "
            "in t+1..t+h on F4 injection at t, conditional on pretreatment history and "
            "concurrent observed F5 injection"
        ),
        "feature_columns": list(design.X.columns),
        "forbidden_feature_offsets": "t or later",
        **design.metadata,
    }
    with open(PROC / "fieldwide_design.json", "w", encoding="utf-8") as f:
        json.dump(design_meta, f, indent=2, ensure_ascii=False)
    diagnostics["sensitivity"] = sensitivity
    with open(PROC / "fieldwide_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learner", choices=("hgb", "rf", "ridge", "ols"), default="hgb"
    )
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--purge", type=int, default=PURGE_WEEKS)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()

    design = build_design()
    print(
        f"Design: n={len(design.dates)} weeks, p={design.X.shape[1]}, "
        f"{design.dates.min().date()} to {design.dates.max().date()}"
    )
    effects, impulse, diagnostics, auxiliary = estimate_all_effects(
        design,
        learner_kind=args.learner,
        n_splits=args.n_splits,
        purge=args.purge,
        max_iter=args.max_iter,
    )

    sensitivity: list[dict] = []
    if not args.skip_sensitivity:
        for learner_kind, purge in (
            ("rf", 16),
            ("ridge", 16),
            ("hgb", 20),
            ("hgb", 24),
        ):
            sensitivity.append(
                estimate_primary_sensitivity(
                    design, learner_kind, purge, max_iter=args.max_iter
                )
            )
    write_outputs(design, effects, impulse, diagnostics, auxiliary, sensitivity)

    primary = effects[
        (effects["outcome"] == "field_oil")
        & (effects["horizon_weeks"] == MAX_HORIZON)
    ].iloc[0]
    print("\nPrimary field-oil effect, cumulative weeks 1-8:")
    print(
        f"  theta={primary.theta:+.6f} m3 oil / m3 F4 "
        f"(HAC 95% CI {primary.ci95_low:+.6f}, {primary.ci95_high:+.6f}; "
        f"p={primary.p_value:.4g})"
    )
    print(
        f"  10% one-week shift support coverage="
        f"{diagnostics['modified_policy_10pct']['residual_support_1pct_99pct_coverage']:.3f}"
    )
    print(f"  reporting status={diagnostics['primary_reporting_status']}")
    print("Saved fieldwide design/effects/diagnostics, an economics stop record, and OOF audit files.")


if __name__ == "__main__":
    main()
