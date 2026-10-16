"""Deep sequence baselines: evaluation must not depend on the other examples in a batch."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from actrn_sequence_baselines import SequenceModel, _AutoCorrelation, _Autoformer  # noqa: E402

L, K, C, M = 10, 8, 8, 2


def _batch(n, seed):
    g = torch.Generator().manual_seed(seed)
    hist = torch.randn(n, L, C, generator=g)
    oil = torch.randn(n, L, 1, generator=g)
    inj = torch.randn(n, L + K, M, generator=g)
    return hist, oil, inj[:, :L], inj[:, L:]


@pytest.mark.parametrize("seed", [0, 71])
def test_autoformer_prediction_does_not_depend_on_batch_companions(seed):
    torch.manual_seed(seed)
    net = _Autoformer(C, M, 32, 0.0).eval()
    x = _batch(80, seed)
    with torch.no_grad():
        together = net(*x, K)
        alone = torch.cat([net(*(t[i:i + 1] for t in x), K) for i in range(80)])
    assert torch.allclose(together, alone, atol=1e-5)


def test_auto_correlation_shares_delays_only_in_training():
    torch.manual_seed(3)
    layer = _AutoCorrelation(16, 4)
    x = torch.randn(32, L, 16)
    layer.train()
    with torch.no_grad():
        shared = layer(x, x, x)
        assert not torch.allclose(shared[:1], layer(x[:1], x[:1], x[:1]), atol=1e-5)
    layer.eval()
    with torch.no_grad():
        assert torch.allclose(layer(x, x, x)[:1], layer(x[:1], x[:1], x[:1]), atol=1e-6)


@pytest.mark.parametrize("kind", ["RNN", "LSTM", "Transformer", "Autoformer"])
def test_fitted_sequence_models_predict_each_example_independently(kind):
    rng = np.random.default_rng(5)
    h = rng.normal(size=(64, L, M + 6)) + 10
    u = rng.normal(size=(64, K, M)) + 10
    y = h[:, -1:, M:M + 3].sum(-1) + u.sum(-1)
    model = SequenceModel(kind, seed=1, params={"d_model": 16}, max_epochs=3, patience=3)
    model.fit(h[:48], u[:48], y[:48], (h[48:], u[48:], y[48:]))
    together = model.predict(h, u)
    alone = np.concatenate([model.predict(h[i:i + 1], u[i:i + 1]) for i in range(len(h))])
    np.testing.assert_allclose(together, alone, rtol=1e-5, atol=1e-5)
