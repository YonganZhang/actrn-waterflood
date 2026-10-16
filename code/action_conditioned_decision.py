"""Action-conditioned temporal response learning and injection decision search.

This module implements the input representations of the Action-Conditioned
Temporal Response Network (ACTRN; the model itself is in ``actrn_model.py``), a
matched set of conventional baselines, a reservoir-inspired known-truth
benchmark, and a field-case forecasting dataset builder.  The
academic task is general: learn future production under alternative injection
sequences and select the sequence that maximizes the stated production
objective.  Field identifiers and forecast horizons are experiment settings,
not definitions of the method.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODEL_NAMES = ("Ridge", "Random forest", "HGB", "MLP", "ACTRN")


@dataclass(frozen=True)
class Reservoir:
    connectivity: np.ndarray
    retention: np.ndarray
    capacity: np.ndarray
    decline: np.ndarray
    breakthrough: np.ndarray
    injector_interaction: np.ndarray


@dataclass(frozen=True)
class DecisionContext:
    reservoir_id: int
    history: np.ndarray
    pressure: np.ndarray
    water: np.ndarray
    age: int
    budget: float
    reservoir: Reservoir
    response: str = "exp"


def injection_effect(action: np.ndarray, response: str = "exp") -> np.ndarray:
    """Saturating pressure-support response of the known-truth benchmark.

    ``exp`` is the registered benchmark; ``hill`` is the robustness variant whose
    curvature differs from the exponential action feature used by ACTRN.
    """
    if response == "exp":
        return 1.0 - np.exp(-1.9 * action)
    if response == "hill":
        return action / (action + 0.55)
    raise ValueError(f"unknown benchmark response {response!r}")


def make_reservoir(rng: np.random.Generator, population: str = "base") -> Reservoir:
    """Reservoir parameters.  ``channelised``: each producer is connected to one injector by a
    high-conductivity channel and only weakly to the other, so the best allocation of a fixed
    injection volume differs strongly between reservoirs."""
    if population == "base":
        connectivity = rng.uniform(0.08, 0.65, size=(3, 2))
        connectivity *= rng.uniform(0.65, 1.45, size=(1, 2))
    elif population == "channelised":
        connectivity = rng.uniform(0.02, 0.15, size=(3, 2))
        connectivity[np.arange(3), rng.integers(0, 2, size=3)] += rng.uniform(0.55, 0.85, size=3)
    else:
        raise ValueError(f"unknown reservoir population {population!r}")
    return Reservoir(
        connectivity=connectivity,
        retention=rng.uniform(0.58, 0.86, size=3),
        capacity=rng.uniform(0.75, 1.55, size=3),
        decline=rng.uniform(0.004, 0.016, size=3),
        breakthrough=rng.uniform(0.035, 0.095, size=3),
        injector_interaction=rng.uniform(-0.18, 0.38, size=3),
    )


def advance(
    reservoir: Reservoir,
    pressure: np.ndarray,
    water: np.ndarray,
    action: np.ndarray,
    age: int,
    response: str = "exp",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action = np.clip(action, 0.0, None)
    action_effect = injection_effect(action, response)
    low_pressure_gate = 1.0 / (1.0 + np.exp(5.0 * (pressure - 0.62)))
    injector_interaction = action_effect[0] * action_effect[1]
    support = (
        (reservoir.connectivity @ action_effect)
        * (0.52 + 0.96 * low_pressure_gate)
        + reservoir.injector_interaction * injector_interaction
    )
    support = np.clip(support, -0.20, 1.80)
    pressure_next = (
        reservoir.retention * pressure
        + (1.0 - reservoir.retention) * 0.18
        + 0.68 * support
        - 0.055 * water
    )
    pressure_next = np.clip(pressure_next, 0.02, 1.8)
    water_next = np.clip(
        0.91 * water
        + reservoir.breakthrough * (reservoir.connectivity @ action**1.75),
        0.0,
        0.92,
    )
    pressure_factor = 1.0 / (1.0 + np.exp(-4.2 * (pressure_next - 0.42)))
    action_state_interaction = 0.38 * support * pressure_factor
    oil = (
        reservoir.capacity
        * np.exp(-reservoir.decline * age)
        * (0.18 + 0.84 * pressure_factor + action_state_interaction)
        * (1.0 - 0.78 * water_next)
    )
    return pressure_next, water_next, oil


def rollout(
    reservoir: Reservoir,
    pressure: np.ndarray,
    water: np.ndarray,
    plan: np.ndarray,
    age: int,
    response: str = "exp",
) -> np.ndarray:
    result = []
    p = pressure.copy()
    w = water.copy()
    for k, action in enumerate(plan):
        p, w, oil = advance(reservoir, p, w, action, age + k, response)
        result.append(float(oil.sum()))
    return np.asarray(result)


def sample_plan(
    rng: np.random.Generator,
    horizon: int,
    budget: float | None = None,
) -> np.ndarray:
    logits = rng.normal(size=(horizon, 2))
    logits = 0.72 * logits + 0.28 * np.cumsum(logits, axis=0) / np.sqrt(
        np.arange(1, horizon + 1)[:, None]
    )
    weights = np.exp(logits - logits.max())
    if budget is None:
        budget = float(rng.uniform(0.60, 1.12) * horizon * 2)
    return allocate_budget(weights, budget)


ACTION_LOWER = 0.02
ACTION_UPPER = 1.75


def allocate_budget(weights: np.ndarray, budget: float) -> np.ndarray:
    """Scale non-negative weights into a bounded plan whose total equals the budget."""
    weights = np.asarray(weights, dtype=float)
    lower = ACTION_LOWER
    upper = ACTION_UPPER
    budget = float(np.clip(budget, lower * weights.size, upper * weights.size))
    lo = 0.0
    hi = upper * weights.size / max(float(weights.min()), 1e-12)
    for _ in range(80):
        scale = 0.5 * (lo + hi)
        total = float(np.clip(weights * scale, lower, upper).sum())
        if total < budget:
            lo = scale
        else:
            hi = scale
    plan = np.clip(weights * (0.5 * (lo + hi)), lower, upper)
    plan.flat[-1] += budget - float(plan.sum())
    return plan


def simulate_dataset(
    n_reservoirs: int = 48,
    samples_per_reservoir: int = 108,
    plans_per_context: int = 6,
    history_length: int = 10,
    horizon: int = 8,
    seed: int = 20260716,
    return_context_ids: bool = False,
    response: str = "exp",
    training_design: str = "variable",
    population: str = "base",
) -> tuple:
    """Known-truth dataset: several simulated schedules restarted from each sampled state.

    ``training_design`` sets how the schedules of one state are drawn: ``variable`` gives each
    its own total injection; ``matched`` gives all of them one common total, so their
    differences are pure reallocation and retiming as in the fixed-budget decision; ``paired``
    uses two totals per state, half of the schedules each.  The schedules are drawn from a
    separate random stream, so geology, histories and decision libraries do not depend on it.
    """
    if training_design not in ("variable", "matched", "paired"):
        raise ValueError(f"unknown training design {training_design!r}")
    rng = np.random.default_rng(seed)
    design_rng = np.random.default_rng([seed, 1])
    if samples_per_reservoir % plans_per_context:
        raise ValueError(
            "samples_per_reservoir must be divisible by plans_per_context"
        )
    histories = []
    actions = []
    outcomes = []
    groups = []
    context_ids = []
    contexts: list[DecisionContext] = []
    for reservoir_id in range(n_reservoirs):
        reservoir = make_reservoir(rng, population)
        pressure = rng.uniform(0.20, 0.72, size=3)
        water = rng.uniform(0.02, 0.20, size=3)
        last_action = rng.uniform(0.35, 1.10, size=2)
        rows = []
        states = []
        waters = []
        ages = []
        for t in range(85):
            policy = np.clip(
                0.66 * last_action
                + 0.22 * (0.72 - pressure.mean())
                + rng.normal(0.0, 0.19, size=2),
                0.03,
                1.65,
            )
            states.append(pressure.copy())
            waters.append(water.copy())
            ages.append(t)
            pressure, water, oil = advance(
                reservoir, pressure, water, policy, t, response
            )
            rows.append(np.r_[policy, oil, water])
            last_action = policy
        rows = np.asarray(rows)
        n_contexts = samples_per_reservoir // plans_per_context
        eligible = np.linspace(
            history_length,
            len(rows) - horizon - 1,
            n_contexts,
            dtype=int,
        )
        for context_no, t in enumerate(eligible):
            history = rows[t - history_length : t]
            budgets = design_rng.uniform(0.60, 1.12, size=2) * horizon * 2
            for plan_no in range(plans_per_context):
                plan = sample_plan(rng, horizon)    # always drawn: keeps the main stream fixed
                if training_design == "matched":
                    plan = sample_plan(design_rng, horizon, budgets[0])
                elif training_design == "paired":
                    plan = sample_plan(design_rng, horizon, budgets[plan_no % 2])
                truth = rollout(
                    reservoir, states[t], waters[t], plan, ages[t], response
                )
                histories.append(history)
                actions.append(plan)
                outcomes.append(truth)
                groups.append(reservoir_id)
                context_ids.append(reservoir_id * n_contexts + context_no)
            decision_context_positions = {
                max(0, min(n_contexts - 1, int(round(n_contexts * fraction))))
                for fraction in (0.2, 0.4, 0.6, 0.8)
            }
            if context_no in decision_context_positions:
                contexts.append(
                    DecisionContext(
                        reservoir_id=reservoir_id,
                        history=history,
                        pressure=states[t],
                        water=waters[t],
                        age=int(ages[t]),
                        budget=float(horizon * rng.uniform(1.40, 1.85)),
                        reservoir=reservoir,
                        response=response,
                    )
                )
    result = (
        np.asarray(histories),
        np.asarray(actions),
        np.asarray(outcomes),
        np.asarray(groups),
        contexts,
    )
    if return_context_ids:
        return (*result, np.asarray(context_ids))
    return result


def encode_history(histories: np.ndarray) -> np.ndarray:
    """x_H = [vec(H), H_L, mean, sd, least-squares slope] along the history index."""
    histories = np.asarray(histories)
    weights = np.linspace(-1.0, 1.0, histories.shape[1])
    weights /= np.sum(weights**2)
    slope = np.einsum("t,ntc->nc", weights, histories)
    return np.c_[
        histories.reshape(len(histories), -1),
        histories[:, -1],
        histories.mean(axis=1),
        histories.std(axis=1),
        slope,
    ]


def response_proxies(histories: np.ndarray, n_injectors: int = 2,
                     lags: tuple[int, ...] = (0, 1)) -> np.ndarray:
    """Within-window correlation of each injector rate with later increments of every other
    channel (lag 0 and 1), an interwell-connectivity proxy in the spirit of rank-correlation
    and capacitance-resistance analyses.  History channels must start with the injectors."""
    histories = np.asarray(histories, dtype=float)
    rates = histories[:, :, :n_injectors]
    increments = np.diff(histories[:, :, n_injectors:], axis=1)
    increments = increments - increments.mean(axis=1, keepdims=True)
    features = []
    for lag in lags:
        lagged = rates[:, 1 - lag: histories.shape[1] - lag]
        lagged = lagged - lagged.mean(axis=1, keepdims=True)
        covariance = np.einsum("ntj,ntk->njk", lagged, increments)
        norm = np.sqrt(np.einsum("ntj,ntj->nj", lagged, lagged)[:, :, None]
                       * np.einsum("ntk,ntk->nk", increments, increments)[:, None, :])
        features.append((covariance / (norm + 1e-9)).reshape(len(histories), -1))
    return np.concatenate(features, axis=1)


def regression_proxies(histories: np.ndarray, n_injectors: int = 2,
                       ridge: float = 1.0) -> np.ndarray:
    """Joint response descriptors: within the history window, ridge coefficients of every other
    channel's increments on the current and previous rates of all injectors (standardised per
    window).  Unlike the marginal correlations they separate the injectors' contributions."""
    histories = np.asarray(histories, dtype=float)
    rates = histories[:, :, :n_injectors]
    x = np.concatenate([rates[:, 1:], rates[:, :-1]], axis=2)
    y = np.diff(histories[:, :, n_injectors:], axis=1)
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-9)
    y = (y - y.mean(axis=1, keepdims=True)) / (y.std(axis=1, keepdims=True) + 1e-9)
    gram = np.einsum("ntp,ntq->npq", x, x) + ridge * np.eye(x.shape[2])[None]
    return np.linalg.solve(gram, np.einsum("ntp,ntk->npk", x, y)).reshape(len(histories), -1)


PROXY_SETS = ("full", "correlation", "none")


def history_features(histories: np.ndarray, proxies: str = "full") -> np.ndarray:
    """ACTRN history representation x_H: statistics plus response proxies rho(H_t).

    ``full`` = correlation and regression proxies; ``correlation`` = correlations only;
    ``none`` = statistics only."""
    if proxies not in PROXY_SETS:
        raise ValueError(f"unknown proxy set {proxies!r}")
    parts = [encode_history(histories)]
    if proxies in ("full", "correlation"):
        parts.append(response_proxies(histories))
    if proxies == "full":
        parts.append(regression_proxies(histories))
    return np.concatenate(parts, axis=1)


def saturation_scale(actions: np.ndarray) -> np.ndarray:
    """Per-injector mean positive injection, estimated from training actions only."""
    actions = np.asarray(actions, dtype=float)
    flat = actions.reshape(-1, actions.shape[-1])
    scale = np.ones(flat.shape[1])
    for j in range(flat.shape[1]):
        positive = flat[:, j][flat[:, j] > 0]
        if positive.size:
            scale[j] = float(positive.mean())
    return scale


def encode_actions(
    actions: np.ndarray,
    scale: np.ndarray,
    mode: str = "multiscale",
) -> np.ndarray:
    """x_U = [vec(U), vec(cumsum U), vec(dU), vec(1 - exp(-U/s))] or raw vec(U)."""
    actions = np.asarray(actions, dtype=float)
    flat = actions.reshape(len(actions), -1)
    if mode == "raw":
        return flat
    if mode != "multiscale":
        raise ValueError(f"unknown action feature mode {mode!r}")
    cumulative = np.cumsum(actions, axis=1)
    increments = np.diff(actions, axis=1, prepend=actions[:, :1])
    saturation = 1.0 - np.exp(-np.clip(actions, 0.0, None) / scale[None, None, :])
    return np.c_[
        flat,
        cumulative.reshape(len(actions), -1),
        increments.reshape(len(actions), -1),
        saturation.reshape(len(actions), -1),
    ]


BASELINE_DEFAULTS = {
    "Ridge": {"alpha": 10.0},
    "Random forest": {"max_depth": 14, "min_samples_leaf": 3},
    "HGB": {"learning_rate": 0.05, "max_leaf_nodes": 24, "min_samples_leaf": 12},
    "MLP": {"hidden_layer_sizes": (72, 48), "alpha": 2e-4, "learning_rate_init": 0.001},
}


class FlatModel:
    """Direct baselines on raw flattened inputs or on ACTRN's input features [x_H, x_U]."""

    def __init__(self, kind: str, seed: int = 42, params: dict | None = None,
                 features: str = "raw"):
        self.kind = kind
        self.seed = seed
        self.params = {**BASELINE_DEFAULTS[kind], **(params or {})}
        self.features = features

    def _design(self, histories, actions):
        if self.features == "raw":
            return np.c_[histories.reshape(len(histories), -1),
                         actions.reshape(len(actions), -1)]
        if self.features == "engineered":
            return np.c_[history_features(histories),
                         encode_actions(actions, self.saturation_scale_)]
        raise ValueError(f"unknown baseline feature set {self.features!r}")

    def _build(self):
        p = self.params
        if self.kind == "Ridge":
            return Ridge(alpha=p["alpha"])
        if self.kind == "Random forest":
            return RandomForestRegressor(
                n_estimators=240,
                max_depth=p["max_depth"],
                min_samples_leaf=p["min_samples_leaf"],
                max_features=0.8,
                n_jobs=1,
                random_state=self.seed,
            )
        if self.kind == "HGB":
            return MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    learning_rate=p["learning_rate"],
                    max_iter=180,
                    max_leaf_nodes=p["max_leaf_nodes"],
                    min_samples_leaf=p["min_samples_leaf"],
                    l2_regularization=1.0,
                    early_stopping=False,
                    random_state=self.seed,
                ),
                n_jobs=1,
            )
        if self.kind == "MLP":
            return MLPRegressor(
                hidden_layer_sizes=p["hidden_layer_sizes"],
                alpha=p["alpha"],
                learning_rate_init=p["learning_rate_init"],
                max_iter=700,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=35,
                random_state=self.seed,
            )
        raise ValueError(self.kind)

    def fit(self, histories, actions, truth):
        self.saturation_scale_ = saturation_scale(actions)
        x = self._design(histories, actions)
        self.x_scaler = StandardScaler().fit(x)
        self.y_scaler = StandardScaler().fit(truth)
        self.model = self._build()
        self.model.fit(self.x_scaler.transform(x), self.y_scaler.transform(truth))
        return self

    def predict(self, histories, actions):
        x = self._design(histories, actions)
        pred = self.model.predict(self.x_scaler.transform(x))
        return self.y_scaler.inverse_transform(pred)


def metrics(truth, pred):
    scale = float(np.mean(np.abs(truth)))
    rmse = float(np.sqrt(np.mean((np.asarray(truth) - np.asarray(pred)) ** 2)))
    return {
        "mae": float(mean_absolute_error(truth, pred)),
        "nmae_pct": float(100 * mean_absolute_error(truth, pred) / scale),
        "rmse": rmse,
        "nrmse_pct": float(100 * rmse / scale),
        "r2": float(r2_score(truth, pred, multioutput="variance_weighted")),
    }


def make_decision_sets(
    contexts: list[DecisionContext],
    rng: np.random.Generator,
    horizon: int = 8,
    n_candidates: int = 80,
) -> list[dict]:
    """Create fixed-budget candidate sets and their known production truth."""
    decision_sets = []
    for context in contexts:
        plans = np.asarray(
            [sample_plan(rng, horizon, context.budget) for _ in range(n_candidates)]
        )
        truth = np.asarray(
            [
                rollout(
                    context.reservoir,
                    context.pressure,
                    context.water,
                    plan,
                    context.age,
                    context.response,
                )
                for plan in plans
            ]
        )
        decision_sets.append({"context": context, "plans": plans, "truth": truth})
    return decision_sets


def candidate_predictions(model, decision_sets: list[dict]) -> list[np.ndarray]:
    """Predicted trajectories of every candidate in every decision set."""
    out = []
    for item in decision_sets:
        histories = np.repeat(item["context"].history[None, ...], len(item["plans"]), axis=0)
        out.append(model.predict(histories, item["plans"]))
    return out


def score_decision_sets(model, decision_sets: list[dict]) -> tuple[dict, list[dict]]:
    """Score model-selected decisions against the same candidate-set truth."""
    return score_predictions(decision_sets, candidate_predictions(model, decision_sets))


def score_predictions(decision_sets: list[dict],
                      predictions: list[np.ndarray]) -> tuple[dict, list[dict]]:
    """Score the candidates selected by predicted cumulative oil.

    Besides regret relative to the best candidate, ``gain_capture_pct`` is the share of the
    attainable gain over the average candidate, 100 (V_sel - V_mean) / (V_best - V_mean)."""
    details = []
    for item, predicted in zip(decision_sets, predictions):
        context = item["context"]
        truth_value = item["truth"].sum(axis=1)
        predicted_value = np.asarray(predicted).sum(axis=1)
        n_candidates = len(truth_value)
        best = int(np.argmax(truth_value))
        selected = int(np.argmax(predicted_value))
        regret = float(truth_value[best] - truth_value[selected])
        mean_value = float(truth_value.mean())
        rank = int(np.sum(truth_value > truth_value[selected]) + 1)
        details.append(
            {
                "reservoir_id": context.reservoir_id,
                "decision_age": context.age,
                "candidate_count": n_candidates,
                "fixed_budget": context.budget,
                "true_best_value": float(truth_value[best]),
                "candidate_mean_value": mean_value,
                "selected_true_value": float(truth_value[selected]),
                "selected_predicted_value": float(predicted_value[selected]),
                "decision_regret": regret,
                "decision_regret_pct": 100 * regret / max(abs(truth_value[best]), 1e-8),
                "gain_capture_pct": 100 * (truth_value[selected] - mean_value)
                / max(truth_value[best] - mean_value, 1e-12),
                "selected_truth_rank": rank,
                "optimal_selection": bool(best == selected),
                "top_decile_selection": bool(rank <= max(1, int(np.ceil(0.10 * n_candidates)))),
            }
        )
    frame = pd.DataFrame(details)
    summary = {
        "decision_regret": float(frame["decision_regret"].mean()),
        "decision_regret_pct": float(frame["decision_regret_pct"].mean()),
        "gain_capture_pct": float(frame["gain_capture_pct"].mean()),
        "optimal_selection_pct": float(100 * frame["optimal_selection"].mean()),
        "top_decile_selection_pct": float(100 * frame["top_decile_selection"].mean()),
    }
    return summary, details


def decision_metrics(model, contexts, rng, horizon=8, n_candidates=80):
    decision_sets = make_decision_sets(contexts, rng, horizon, n_candidates)
    summary, _ = score_decision_sets(model, decision_sets)
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


FIELD_HISTORY_COLUMNS = [
    "inj_F4_m3",
    "inj_F5_m3",
    "oil__F1C",
    "oil__F11",
    "oil__F12",
    "oil__F14",
    "oil__F15D",
    "water__F1C",
    "water__F11",
    "water__F12",
    "water__F14",
    "water__F15D",
]


def build_field_case_dataset(
    history_length: int = 10,
    horizon: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build the Volve case windows without defining the academic task by them."""
    weekly = pd.read_parquet(PROC / "fieldwide_weekly.parquet")
    design = json.loads((PROC / "fieldwide_design.json").read_text())
    start, end = design["analysis_date_range"]
    dates = pd.date_range(start, end, freq="W-SUN")
    histories = []
    actions = []
    outcomes = []
    retained_dates = []
    for date in dates:
        position = int(weekly.index.get_loc(date))
        if position - history_length < 0 or position + horizon >= len(weekly):
            continue
        histories.append(
            weekly.iloc[position - history_length : position][
                FIELD_HISTORY_COLUMNS
            ].to_numpy(float)
        )
        actions.append(
            weekly.iloc[position : position + horizon][
                ["inj_F4_m3", "inj_F5_m3"]
            ].to_numpy(float)
        )
        outcomes.append(
            weekly.iloc[position + 1 : position + horizon + 1][
                "field_oil_m3"
            ].to_numpy(float)
        )
        retained_dates.append(date)
    return (
        np.asarray(histories),
        np.asarray(actions),
        np.asarray(outcomes),
        pd.DatetimeIndex(retained_dates, name="decision_date"),
    )


# Experiment protocols (splits, tuning, ablations, statistics) live in
# ``actrn_experiments.py``.  The 2026-07-16 runners are preserved at git tag
# ``actrn-v1-registered`` (their field split leaked labels and is retired); the
# 2026-10 v2 model and runners at ``actrn-v2-registered``.
