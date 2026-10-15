"""Regression tests for the ACTRN v2 protocol (leakage-free field split, scale-aware features)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from action_conditioned_decision import (  # noqa: E402
    ACTRN,
    build_field_case_dataset,
    encode_actions,
    injection_effect,
    saturation_scale,
)
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


def test_interaction_ablation_removes_the_multiplicative_path():
    rng = np.random.default_rng(3)
    h, a = rng.normal(size=(40, 5, 3)), np.abs(rng.normal(size=(40, 4, 2)))
    y = rng.normal(size=(40, 4))
    model = ACTRN(latent_dim=4, fusion_dim=6, max_epochs=3, ranking_weight=0.0,
                  use_interaction=False).fit(h[:30], a[:30], y[:30],
                                            validation_data=(h[30:], a[30:], y[30:]))
    hs = model.h_scaler.transform(model._encode_history(h))
    as_ = model.a_scaler.transform(model._encode_actions(a))
    _, cache = model._forward(hs, as_)
    assert np.allclose(cache[4], 0.0)


def test_every_model_has_an_equal_eight_configuration_budget():
    for setting in ("benchmark", "field"):
        sizes = {name: len(grid) for name, grid in GRIDS[setting].items()}
        assert set(sizes.values()) == {8}
    assert set(ABLATIONS) == {"ACTRN w/o ranking loss", "ACTRN w/o interaction",
                              "ACTRN raw action input"}
