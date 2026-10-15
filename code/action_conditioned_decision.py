"""Action-conditioned temporal response learning and injection decision search.

This module implements the Action-Conditioned Temporal Response Network
(ACTRN), a matched set of conventional baselines, a reservoir-inspired
known-truth benchmark, and a field-case forecasting dataset builder.  The
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


def make_reservoir(rng: np.random.Generator) -> Reservoir:
    connectivity = rng.uniform(0.08, 0.65, size=(3, 2))
    connectivity *= rng.uniform(0.65, 1.45, size=(1, 2))
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
) -> tuple:
    rng = np.random.default_rng(seed)
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
        reservoir = make_reservoir(rng)
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
            for _ in range(plans_per_context):
                plan = sample_plan(rng, horizon)
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


class ACTRN:
    def __init__(
        self,
        latent_dim: int = 32,
        fusion_dim: int = 48,
        learning_rate: float = 0.003,
        l2: float = 2e-4,
        max_epochs: int = 600,
        batch_size: int = 128,
        patience: int = 55,
        cumulative_weight: float = 0.0,
        ranking_weight: float = 3.0,
        seed: int = 42,
        use_interaction: bool = True,
        action_features: str = "multiscale",
    ) -> None:
        self.latent_dim = latent_dim
        self.fusion_dim = fusion_dim
        self.learning_rate = learning_rate
        self.l2 = l2
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.cumulative_weight = cumulative_weight
        self.ranking_weight = ranking_weight
        self.seed = seed
        self.use_interaction = use_interaction
        self.action_features = action_features
        self._interaction_gate = 1.0 if use_interaction else 0.0

    @staticmethod
    def _encode_history(histories: np.ndarray) -> np.ndarray:
        return encode_history(histories)

    def _encode_actions(self, actions: np.ndarray) -> np.ndarray:
        return encode_actions(actions, self.saturation_scale_, self.action_features)

    def _init_params(self, h_dim: int, a_dim: int, out_dim: int) -> None:
        rng = np.random.default_rng(self.seed)
        def weight(n_in: int, n_out: int) -> np.ndarray:
            return rng.normal(0.0, np.sqrt(2.0 / (n_in + n_out)), (n_in, n_out))
        d = self.latent_dim
        self.params = {
            "Wh": weight(h_dim, d),
            "bh": np.zeros(d),
            "Wa": weight(a_dim, d),
            "ba": np.zeros(d),
            "Wf": weight(3 * d, self.fusion_dim),
            "bf": np.zeros(self.fusion_dim),
            "Wo": weight(self.fusion_dim, out_dim),
            "Wbase": np.zeros((h_dim, out_dim)),
            "Waction": np.zeros((a_dim, out_dim)),
            "Winter": np.zeros((d, out_dim)),
            "bo": np.zeros(out_dim),
        }

    def _forward(self, h: np.ndarray, a: np.ndarray):
        p = self.params
        zh = np.tanh(h @ p["Wh"] + p["bh"])
        za = np.tanh(a @ p["Wa"] + p["ba"])
        interaction = self._interaction_gate * (zh * za)
        fusion = np.concatenate([zh, za, interaction], axis=1)
        zf = np.tanh(fusion @ p["Wf"] + p["bf"])
        pred = (
            h @ p["Wbase"]
            + a @ p["Waction"]
            + interaction @ p["Winter"]
            + zf @ p["Wo"]
            + p["bo"]
        )
        return pred, (h, a, zh, za, interaction, fusion, zf)

    def _backprop(self, d_pred, cache):
        h, a, zh, za, interaction, fusion, zf = cache
        p = self.params
        grads = {
            "Wo": zf.T @ d_pred + self.l2 * p["Wo"],
            "Wbase": h.T @ d_pred + self.l2 * p["Wbase"],
            "Waction": a.T @ d_pred + self.l2 * p["Waction"],
            "Winter": interaction.T @ d_pred + self.l2 * p["Winter"],
            "bo": d_pred.sum(axis=0),
        }
        d_zf = d_pred @ p["Wo"].T * (1.0 - zf**2)
        grads["Wf"] = fusion.T @ d_zf + self.l2 * p["Wf"]
        grads["bf"] = d_zf.sum(axis=0)
        d_fusion = d_zf @ p["Wf"].T
        d = self.latent_dim
        d_interaction = self._interaction_gate * (
            d_fusion[:, 2 * d :] + d_pred @ p["Winter"].T
        )
        d_zh = d_fusion[:, :d] + d_interaction * za
        d_za = d_fusion[:, d : 2 * d] + d_interaction * zh
        d_hpre = d_zh * (1.0 - zh**2)
        d_apre = d_za * (1.0 - za**2)
        grads["Wh"] = h.T @ d_hpre + self.l2 * p["Wh"]
        grads["bh"] = d_hpre.sum(axis=0)
        grads["Wa"] = a.T @ d_apre + self.l2 * p["Wa"]
        grads["ba"] = d_apre.sum(axis=0)
        return grads

    def _gradients(self, pred, truth, cache):
        d_pred = 2.0 * (pred - truth) / pred.size
        cumulative_error = pred.mean(axis=1) - truth.mean(axis=1)
        d_pred += (
            2.0
            * self.cumulative_weight
            * cumulative_error[:, None]
            / (len(pred) * pred.shape[1])
        )
        return self._backprop(d_pred, cache)

    def _ranking_gradient(
        self,
        pred: np.ndarray,
        truth: np.ndarray,
        context_ids: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Return a pairwise decision-ranking gradient on matched contexts."""
        pred_physical = self.y_scaler.inverse_transform(pred)
        pred_value = pred_physical.mean(axis=1)
        truth_value = np.asarray(truth).mean(axis=1)
        normalizer = max(float(np.std(truth_value)), 1e-6)
        pred_value = pred_value / normalizer
        truth_value = truth_value / normalizer
        d_value = np.zeros(len(pred), dtype=float)
        losses = []
        for context_id in np.unique(context_ids):
            indices = np.flatnonzero(context_ids == context_id)
            for left_pos in range(len(indices)):
                for right_pos in range(left_pos + 1, len(indices)):
                    left = int(indices[left_pos])
                    right = int(indices[right_pos])
                    sign = float(np.sign(truth_value[left] - truth_value[right]))
                    if sign == 0.0:
                        continue
                    margin = sign * (pred_value[left] - pred_value[right])
                    losses.append(float(np.logaddexp(0.0, -margin)))
                    derivative = -sign / (1.0 + np.exp(np.clip(margin, -40.0, 40.0)))
                    d_value[left] += derivative
                    d_value[right] -= derivative
        if not losses:
            return np.zeros_like(pred), 0.0
        d_value /= len(losses)
        output_weights = self.y_scaler.scale_ / (
            pred.shape[1] * normalizer
        )
        return d_value[:, None] * output_weights[None, :], float(np.mean(losses))

    def fit(
        self,
        histories,
        actions,
        truth,
        validation_data=None,
        context_ids: np.ndarray | None = None,
        validation_context_ids: np.ndarray | None = None,
    ):
        self.saturation_scale_ = saturation_scale(actions)
        h = self._encode_history(histories)
        a = self._encode_actions(actions)
        y_physical = np.asarray(truth)
        self.h_scaler = StandardScaler().fit(h)
        self.a_scaler = StandardScaler().fit(a)
        self.y_scaler = StandardScaler().fit(y_physical)
        h = self.h_scaler.transform(h)
        a = self.a_scaler.transform(a)
        y = self.y_scaler.transform(y_physical)
        if validation_data is None:
            split = max(1, int(0.85 * len(h)))
            vh, va, vy = h[split:], a[split:], y[split:]
            vy_physical = y_physical[split:]
            if context_ids is not None:
                validation_context_ids = np.asarray(context_ids)[split:]
                context_ids = np.asarray(context_ids)[:split]
            h, a, y = h[:split], a[:split], y[:split]
            y_physical = y_physical[:split]
        else:
            vh0, va0, vy0 = validation_data
            vh = self.h_scaler.transform(self._encode_history(vh0))
            va = self.a_scaler.transform(self._encode_actions(va0))
            vy_physical = np.asarray(vy0)
            vy = self.y_scaler.transform(vy_physical)
        if context_ids is not None:
            context_ids = np.asarray(context_ids)
            if len(context_ids) != len(h):
                raise ValueError("context_ids must align with the training samples")
        if validation_context_ids is not None:
            validation_context_ids = np.asarray(validation_context_ids)
            if len(validation_context_ids) != len(vh):
                raise ValueError(
                    "validation_context_ids must align with validation samples"
                )
        self._init_params(h.shape[1], a.shape[1], y.shape[1])
        rng = np.random.default_rng(self.seed + 1)
        moments = {name: np.zeros_like(value) for name, value in self.params.items()}
        velocities = {name: np.zeros_like(value) for name, value in self.params.items()}
        best = {name: value.copy() for name, value in self.params.items()}
        best_loss = np.inf
        stale = 0
        step = 0
        for epoch in range(self.max_epochs):
            order = rng.permutation(len(h))
            for start in range(0, len(h), self.batch_size):
                idx = order[start : start + self.batch_size]
                pred, cache = self._forward(h[idx], a[idx])
                grads = self._gradients(pred, y[idx], cache)
                step += 1
                for name, grad in grads.items():
                    moments[name] = 0.9 * moments[name] + 0.1 * grad
                    velocities[name] = 0.999 * velocities[name] + 0.001 * grad**2
                    m_hat = moments[name] / (1.0 - 0.9**step)
                    v_hat = velocities[name] / (1.0 - 0.999**step)
                    self.params[name] -= self.learning_rate * m_hat / (
                        np.sqrt(v_hat) + 1e-8
                    )
            ranking_loss = 0.0
            if self.ranking_weight > 0.0 and context_ids is not None:
                rank_pred, rank_cache = self._forward(h, a)
                rank_output_grad, ranking_loss = self._ranking_gradient(
                    rank_pred, y_physical, context_ids
                )
                rank_grads = self._backprop(
                    self.ranking_weight * rank_output_grad, rank_cache
                )
                step += 1
                for name, grad in rank_grads.items():
                    moments[name] = 0.9 * moments[name] + 0.1 * grad
                    velocities[name] = 0.999 * velocities[name] + 0.001 * grad**2
                    m_hat = moments[name] / (1.0 - 0.9**step)
                    v_hat = velocities[name] / (1.0 - 0.999**step)
                    self.params[name] -= self.learning_rate * m_hat / (
                        np.sqrt(v_hat) + 1e-8
                    )
            val_pred, _ = self._forward(vh, va)
            val_loss = float(
                np.mean((val_pred - vy) ** 2)
                + self.cumulative_weight
                * np.mean((val_pred.mean(axis=1) - vy.mean(axis=1)) ** 2)
            )
            if self.ranking_weight > 0.0 and validation_context_ids is not None:
                _, val_ranking_loss = self._ranking_gradient(
                    val_pred, vy_physical, validation_context_ids
                )
                val_loss += self.ranking_weight * val_ranking_loss
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best = {name: value.copy() for name, value in self.params.items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.params = best
        self.n_epochs_ = epoch + 1
        self.validation_loss_ = best_loss
        return self

    def predict(self, histories, actions):
        h = self.h_scaler.transform(self._encode_history(histories))
        a = self.a_scaler.transform(self._encode_actions(actions))
        pred, _ = self._forward(h, a)
        return self.y_scaler.inverse_transform(pred)


BASELINE_DEFAULTS = {
    "Ridge": {"alpha": 10.0},
    "Random forest": {"max_depth": 14, "min_samples_leaf": 3},
    "HGB": {"learning_rate": 0.05, "max_leaf_nodes": 24, "min_samples_leaf": 12},
    "MLP": {"hidden_layer_sizes": (72, 48), "alpha": 2e-4, "learning_rate_init": 0.001},
}


class FlatModel:
    """Direct baselines on raw flattened inputs or on ACTRN's engineered features."""

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
            return np.c_[encode_history(histories),
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


def score_decision_sets(model, decision_sets: list[dict]) -> tuple[dict, list[dict]]:
    """Score model-selected decisions against the same candidate-set truth."""
    regrets = []
    relative_regrets = []
    exact = []
    top_decile = []
    details = []
    for item in decision_sets:
        context = item["context"]
        plans = item["plans"]
        truth = item["truth"]
        n_candidates = len(plans)
        histories = np.repeat(context.history[None, ...], n_candidates, axis=0)
        predicted = model.predict(histories, plans)
        truth_value = truth.sum(axis=1)
        predicted_value = predicted.sum(axis=1)
        best = int(np.argmax(truth_value))
        selected = int(np.argmax(predicted_value))
        regret = float(truth_value[best] - truth_value[selected])
        regrets.append(regret)
        relative_regrets.append(100 * regret / max(abs(truth_value[best]), 1e-8))
        exact.append(best == selected)
        rank = int(np.sum(truth_value > truth_value[selected]) + 1)
        is_top_decile = rank <= max(1, int(np.ceil(0.10 * n_candidates)))
        top_decile.append(is_top_decile)
        details.append(
            {
                "reservoir_id": context.reservoir_id,
                "decision_age": context.age,
                "candidate_count": n_candidates,
                "fixed_budget": context.budget,
                "true_best_value": float(truth_value[best]),
                "selected_true_value": float(truth_value[selected]),
                "selected_predicted_value": float(predicted_value[selected]),
                "decision_regret": regret,
                "decision_regret_pct": relative_regrets[-1],
                "selected_truth_rank": rank,
                "optimal_selection": bool(best == selected),
                "top_decile_selection": bool(is_top_decile),
            }
        )
    summary = {
        "decision_regret": float(np.mean(regrets)),
        "decision_regret_pct": float(np.mean(relative_regrets)),
        "optimal_selection_pct": float(100 * np.mean(exact)),
        "top_decile_selection_pct": float(100 * np.mean(top_decile)),
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
# ``actrn-v1-registered``; their field split leaked labels and is retired.
