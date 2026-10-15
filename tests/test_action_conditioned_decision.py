import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from action_conditioned_decision import (  # noqa: E402
    ACTRN,
    build_field_case_dataset,
    sample_plan,
    simulate_dataset,
)


def test_candidate_plan_preserves_fixed_budget_and_bounds():
    rng = np.random.default_rng(9)
    for budget in (1.0, 8.0, 16.0, 25.0):
        plan = sample_plan(rng, horizon=8, budget=budget)
        assert plan.shape == (8, 2)
        assert np.isclose(plan.sum(), budget, rtol=0.0, atol=1e-9)
        assert np.all(plan >= 0.02 - 1e-12)
        assert np.all(plan <= 1.75 + 1e-12)


def test_known_truth_samples_keep_context_groups_and_case_independence():
    histories, actions, outcomes, reservoirs, contexts, context_ids = simulate_dataset(
        n_reservoirs=12,
        samples_per_reservoir=24,
        plans_per_context=6,
        history_length=6,
        horizon=4,
        return_context_ids=True,
    )
    assert histories.shape == (288, 6, 8)
    assert actions.shape == (288, 4, 2)
    assert outcomes.shape == (288, 4)
    assert len(np.unique(reservoirs)) == 12
    assert np.all(np.bincount(context_ids) == 6)
    assert len(contexts) >= 24


def test_actrn_fits_action_conditioned_multioutput_signal():
    rng = np.random.default_rng(41)
    histories = rng.normal(size=(240, 5, 4))
    actions = rng.normal(size=(240, 3, 2))
    state = histories[:, -1, :2]
    truth = np.column_stack(
        [
            0.7 * state[:, 0]
            + 0.4 * actions[:, step, 0]
            + 0.8 * state[:, 1] * actions[:, step, 1]
            for step in range(3)
        ]
    )
    train = np.arange(180)
    validation = np.arange(180, 210)
    test = np.arange(210, 240)
    model = ACTRN(
        latent_dim=12,
        fusion_dim=16,
        learning_rate=0.003,
        l2=0.001,
        max_epochs=350,
        patience=45,
        cumulative_weight=0.0,
        ranking_weight=0.0,
        seed=17,
    ).fit(
        histories[train],
        actions[train],
        truth[train],
        validation_data=(
            histories[validation],
            actions[validation],
            truth[validation],
        ),
    )
    prediction = model.predict(histories[test], actions[test])
    model_mae = np.mean(np.abs(prediction - truth[test]))
    mean_mae = np.mean(np.abs(truth[train].mean(axis=0) - truth[test]))
    assert prediction.shape == truth[test].shape
    assert np.isfinite(prediction).all()
    assert model_mae < 0.70 * mean_mae


def test_field_case_parameters_remain_in_case_builder():
    histories, actions, outcomes, dates = build_field_case_dataset()
    assert histories.shape == (400, 10, 12)
    assert actions.shape == (400, 8, 2)
    assert outcomes.shape == (400, 8)
    assert len(dates) == 400
