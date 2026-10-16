"""Regression tests for the ACTRN protocol (leakage-free field split, scale-aware features,
response proxies, contrastive response learning, ablation switches)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from action_conditioned_decision import (  # noqa: E402
    FlatModel,
    build_field_case_dataset,
    encode_actions,
    history_features,
    injection_effect,
    regression_proxies,
    response_proxies,
    saturation_scale,
    score_decision_sets,
    score_predictions,
    simulate_dataset,
)
from actrn_model import ACTRN  # noqa: E402
from actrn_experiments import (  # noqa: E402
    ABLATIONS,
    CASE_LEVEL_M3,
    FIELD_FOLD_ENDS,
    FIELD_TEST,
    GRIDS,
    field_blocks,
    field_data,
)

HORIZON = 8


def _label_end(dates, i):
    return dates[i] + np.timedelta64(7 * HORIZON, "D")


def test_field_fitting_labels_precede_every_evaluation_block():
    hist, act, out, dates, *_ = field_data()
    starts = [end - FIELD_TEST for end in FIELD_FOLD_ENDS]
    level = np.median(hist[:, -4:, 2:7].sum(axis=2), axis=1)
    starts.append(int(np.flatnonzero(level >= CASE_LEVEL_M3)[-1]))
    for start in starts:
        blocks = field_blocks(start)
        first_eval_decision = dates[start]
        first_val_decision = dates[blocks["validation"][0]]
        # every label week used for fitting or model selection is observed before the block
        assert _label_end(dates, blocks["validation"][-1]) < first_eval_decision
        assert _label_end(dates, blocks["train"][-1]) < first_val_decision


def test_field_saturation_feature_is_not_constant():
    _, actions, _, _ = build_field_case_dataset()
    scale = saturation_scale(actions[:200])
    features = encode_actions(actions, scale)
    saturation = features[:, -actions.shape[1] * actions.shape[2]:]
    positive = actions.reshape(len(actions), -1) > 0
    assert np.all(scale > 1000.0)
    assert np.unique(np.round(saturation[positive], 6)).size > 100


def test_benchmark_robustness_response_differs_from_action_feature():
    a = np.linspace(0.02, 1.75, 50)
    assert not np.allclose(injection_effect(a, "hill"), injection_effect(a, "exp"))


def test_interaction_ablation_makes_plan_ranking_state_independent():
    rng = np.random.default_rng(3)
    h, a = rng.normal(size=(60, 5, 4)), np.abs(rng.normal(size=(60, 4, 2)))
    y = rng.normal(size=(60, 4))
    model = ACTRN(latent_dim=4, max_epochs=5, ensemble=1, interaction="none").fit(
        h[:45], a[:45], y[:45], validation_data=(h[45:], a[45:], y[45:]))
    assert all("Wh" not in paths.p for paths in model.paths_)
    plans = a[:6]
    first = model.predict(np.repeat(h[:1], 6, axis=0), plans)
    second = model.predict(np.repeat(h[1:2], 6, axis=0), plans)
    assert np.allclose(first - first.mean(0), second - second.mean(0))


def test_contrastive_response_ignores_state_level_shifts():
    """Adding an arbitrary constant per state must not change how plans are ranked."""
    rng = np.random.default_rng(5)
    n_ctx, plans = 40, 6
    h = np.repeat(rng.normal(size=(n_ctx, 5, 4)), plans, axis=0)
    a = np.abs(rng.normal(size=(n_ctx * plans, 4, 2)))
    ctx = np.repeat(np.arange(n_ctx), plans)
    y = a.sum(axis=2) * (1.0 + h[:, -1, 2:3]) + 0.1 * rng.normal(size=(n_ctx * plans, 4))
    shift = np.repeat(rng.normal(scale=5.0, size=(n_ctx, 1)), plans, axis=0)
    tr, va = ctx < 30, ctx >= 30
    preds = []
    for target in (y, y + shift):
        # linear response: exactly invariant (the nonlinear paths see the same centred residuals
        # up to the output scale, which Adam and tanh do not preserve exactly)
        model = ACTRN(nonlinear=False, seed=1)
        model.fit(h[tr], a[tr], target[tr], (h[va], a[va], target[va]), ctx[tr], ctx[va])
        p = model.predict(h[va], a[va]).reshape(-1, plans, 4)
        preds.append(p - p.mean(axis=1, keepdims=True))
    assert np.allclose(preds[0], preds[1], atol=1e-8)
    pooled = ACTRN(nonlinear=False, contrastive=False, seed=1)
    pooled.fit(h[tr], a[tr], (y + shift)[tr], (h[va], a[va], (y + shift)[va]), ctx[tr], ctx[va])
    p = pooled.predict(h[va], a[va]).reshape(-1, plans, 4)
    assert not np.allclose(p - p.mean(axis=1, keepdims=True), preds[0], atol=1e-3)


def test_response_proxies_detect_a_connected_channel():
    rng = np.random.default_rng(7)
    rates = rng.uniform(0.2, 1.0, size=(3, 10, 2))
    connected = np.cumsum(0.8 * rates[:, :, :1], axis=1)       # increments follow injector 1
    unrelated = rng.normal(size=(3, 10, 1))
    proxies = response_proxies(np.concatenate([rates, connected, unrelated], axis=2))
    lag0 = proxies[:, :4].reshape(3, 2, 2)                     # (injector, channel)
    assert np.allclose(lag0[:, 0, 0], 1.0)
    assert np.all(np.abs(lag0[:, 0, 1]) < 0.9)


def test_regression_proxies_separate_two_injectors():
    rng = np.random.default_rng(11)
    rates = rng.uniform(0.2, 1.0, size=(4, 10, 2))
    follows_second = np.cumsum(0.9 * rates[:, :, 1:2], axis=1)   # increments follow injector 2
    proxies = regression_proxies(np.concatenate([rates, follows_second], axis=2))
    coef = proxies.reshape(4, 4, 1)[:, :, 0]                     # [u1_t, u2_t, u1_t-1, u2_t-1]
    assert np.all(np.abs(coef[:, 1]) > 3 * np.abs(coef[:, 0]))


def test_history_feature_sets_nest():
    h = np.random.default_rng(2).normal(size=(5, 10, 8))
    none, corr, full = (history_features(h, p).shape[1] for p in ("none", "correlation", "full"))
    assert none < corr < full


def test_default_training_design_and_population_leave_registered_data_unchanged():
    a = simulate_dataset(n_reservoirs=3, seed=5, return_context_ids=True)
    b = simulate_dataset(n_reservoirs=3, seed=5, return_context_ids=True,
                         training_design="variable", population="base")
    assert all(np.array_equal(x, y) for x, y in zip(a[:4], b[:4]))
    m = simulate_dataset(n_reservoirs=3, seed=5, return_context_ids=True,
                         training_design="matched")
    assert np.array_equal(a[0], m[0])                            # same histories
    totals = m[1].sum(axis=(1, 2))
    for c in np.unique(m[5]):
        assert np.ptp(totals[m[5] == c]) < 1e-9                  # one budget per state


def test_prediction_based_scoring_matches_model_scoring():
    h, a, y, g, ctx = simulate_dataset(n_reservoirs=4, seed=9)
    from action_conditioned_decision import make_decision_sets
    sets = make_decision_sets(ctx[:3], np.random.default_rng(1), 8, 20)
    model = FlatModel("Ridge", params={"alpha": 1.0}, features="engineered").fit(h, a, y)
    summary, details = score_decision_sets(model, sets)
    preds = [model.predict(np.repeat(s["context"].history[None], 20, 0), s["plans"]) for s in sets]
    assert summary == score_predictions(sets, preds)[0]
    assert all(-1e9 < d["gain_capture_pct"] <= 100 + 1e-9 for d in details)


def test_every_model_has_an_equal_eight_configuration_budget():
    for setting in ("benchmark", "field"):
        sizes = {name: len(grid) for name, grid in GRIDS[setting].items()}
        assert set(sizes.values()) == {8}
    assert set(ABLATIONS) == {"ACTRN w/o contrastive learning", "ACTRN w/o interaction",
                              "ACTRN w/o response proxies", "ACTRN correlation proxies only",
                              "ACTRN raw action input", "ACTRN linear response",
                              "ACTRN concatenation interaction"}
