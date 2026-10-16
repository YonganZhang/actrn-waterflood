"""Deep sequence baselines: Elman RNN, LSTM, Transformer, Autoformer and Reformer.

Every model maps a history ``H`` (L steps x C channels; the first m channels are the injector
rates) and a candidate schedule ``U`` (K steps x m) to the K-step oil trajectory, and is trained by
mean squared error with Adam and early stopping on validation data, which is their standard use.

* RNN, LSTM, Transformer and Reformer read one sequence of L + K tokens.  Each token carries the
  injection rates (historical for the first L tokens, the candidate schedule for the last K), the
  other history channels (zero in the future) and a future flag; the prediction for planning
  step k is read from the hidden state of token L + k.
* Autoformer (Wu et al., 2021) follows the official encoder-decoder design with series
  decomposition, Auto-Correlation (batch-shared delays in training, per-example delays in
  evaluation) and seasonal layer normalisation: the encoder reads the history, the decoder starts
  from the decomposed last ``label_len`` history values of field oil plus K placeholders, and the
  injection rates enter both as known covariates.  The field-oil history is expressed in the units of the target
  (``relative_oil`` for the field target, which is a change relative to the recent level).
* Reformer (Kitaev et al., 2020) uses the reference implementation ``reformer_pytorch``
  (locality-sensitive-hashing attention with reversible layers).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEQUENCE_KINDS = ("RNN", "LSTM", "Transformer", "Autoformer", "Reformer")


# ----------------------------------------------------------------- encoders on one token sequence
class _TokenModel(nn.Module):
    def __init__(self, kind, n_in, d, dropout, n_tokens):
        super().__init__()
        self.kind = kind
        self.embed = nn.Linear(n_in, d)
        if kind in ("RNN", "LSTM"):
            rnn = nn.RNN if kind == "RNN" else nn.LSTM
            self.body = rnn(d, d, num_layers=2, batch_first=True, dropout=dropout)
        else:
            self.position = nn.Parameter(torch.randn(1, n_tokens, d) * 0.02)
            if kind == "Transformer":
                layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d,
                                                   dropout=dropout, batch_first=True)
                self.body = nn.TransformerEncoder(layer, num_layers=2)
            else:
                from reformer_pytorch import Autopadder, Reformer
                self.body = Autopadder(Reformer(dim=d, depth=2, heads=4, bucket_size=4,
                                                n_hashes=2, causal=False, ff_dropout=dropout,
                                                lsh_dropout=dropout, layer_dropout=0.0))
        self.head = nn.Linear(d, 1)

    def forward(self, tokens, horizon):
        x = self.embed(tokens)
        if self.kind in ("RNN", "LSTM"):
            x, _ = self.body(x)
        else:
            x = self.body(x + self.position)
        return self.head(x[:, -horizon:]).squeeze(-1)


# ----------------------------------------------------------------- Autoformer
# Follows the official implementation (github.com/thuml/Autoformer: layers/AutoCorrelation.py,
# layers/Autoformer_EncDec.py, models/Autoformer.py).  Adaptations to the 10-step history and the
# single target: moving-average kernel 5 instead of 25, a one-channel decoder for the target (the
# official model decodes all input variables and keeps the target afterwards), the known injection
# rates embedded in place of the calendar time features, and default initialisation of the token
# convolution (the official code uses Kaiming initialisation).
class _MovingAvg(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self.pool = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=0)

    def forward(self, x):
        front = x[:, :1].repeat(1, (self.kernel - 1) // 2, 1)
        end = x[:, -1:].repeat(1, (self.kernel - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)


class _Decomp(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.avg = _MovingAvg(kernel)

    def forward(self, x):
        trend = self.avg(x)
        return x - trend, trend


class _SeasonalNorm(nn.Module):
    """``my_Layernorm`` of the official code: layer norm with the time mean removed."""

    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        x = self.norm(x)
        return x - x.mean(dim=1, keepdim=True)


class _AutoCorrelation(nn.Module):
    """Period-based dependencies and time-delay aggregation.

    As in the official code, training shares the top-k delays across the batch, whereas
    evaluation selects them for every example, so predictions do not depend on the other
    examples in a batch.
    """

    def __init__(self, d, heads, factor=1):
        super().__init__()
        self.heads, self.factor = heads, factor
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def _aggregate(self, values, mean_corr, top_k):
        b, h, c, length = values.shape
        if self.training:
            index = torch.topk(mean_corr.mean(dim=0), top_k, dim=-1)[1]
            weights = torch.softmax(torch.stack([mean_corr[:, i] for i in index], dim=-1), dim=-1)
            agg = torch.zeros_like(values)
            for i in range(top_k):
                agg = agg + torch.roll(values, -int(index[i]), dims=-1) * weights[:, i].view(b, 1, 1, 1)
            return agg
        weights, delay = torch.topk(mean_corr, top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)
        doubled = values.repeat(1, 1, 1, 2)
        base = torch.arange(length, device=values.device).view(1, 1, 1, length)
        agg = torch.zeros_like(values)
        for i in range(top_k):
            index = (base + delay[:, i].view(b, 1, 1, 1)).expand(b, h, c, length)
            agg = agg + torch.gather(doubled, -1, index) * weights[:, i].view(b, 1, 1, 1)
        return agg

    def forward(self, queries, keys, values):
        b, lq, d = queries.shape
        lk = keys.shape[1]
        h = self.heads
        q = self.q(queries).view(b, lq, h, d // h)
        k = self.k(keys).view(b, lk, h, d // h)
        v = self.v(values).view(b, lk, h, d // h)
        if lq > lk:
            pad = torch.zeros_like(q[:, : lq - lk])
            k, v = torch.cat([k, pad], dim=1), torch.cat([v, pad], dim=1)
        else:
            k, v = k[:, :lq], v[:, :lq]
        qf = torch.fft.rfft(q.permute(0, 2, 3, 1).contiguous(), dim=-1)
        kf = torch.fft.rfft(k.permute(0, 2, 3, 1).contiguous(), dim=-1)
        corr = torch.fft.irfft(qf * torch.conj(kf), n=lq, dim=-1)       # b, h, c, l
        top_k = max(1, int(self.factor * math.log(lq)))
        agg = self._aggregate(v.permute(0, 2, 3, 1).contiguous(), corr.mean(dim=1).mean(dim=1),
                              top_k)
        return self.out(agg.permute(0, 3, 1, 2).reshape(b, lq, d))


def _feed_forward(d, dropout):
    return nn.Sequential(nn.Conv1d(d, 2 * d, kernel_size=1, bias=False), nn.GELU(),
                         nn.Dropout(dropout), nn.Conv1d(2 * d, d, kernel_size=1, bias=False),
                         nn.Dropout(dropout))


class _EncoderLayer(nn.Module):
    def __init__(self, d, heads, kernel, dropout):
        super().__init__()
        self.attn = _AutoCorrelation(d, heads)
        self.ff = _feed_forward(d, dropout)
        self.d1, self.d2 = _Decomp(kernel), _Decomp(kernel)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.d1(x + self.drop(self.attn(x, x, x)))
        x, _ = self.d2(x + self.ff(x.transpose(1, 2)).transpose(1, 2))
        return x


class _DecoderLayer(nn.Module):
    def __init__(self, d, heads, kernel, dropout, c_out):
        super().__init__()
        self.self_attn = _AutoCorrelation(d, heads)
        self.cross_attn = _AutoCorrelation(d, heads)
        self.ff = _feed_forward(d, dropout)
        self.d1, self.d2, self.d3 = _Decomp(kernel), _Decomp(kernel), _Decomp(kernel)
        self.trend = nn.Conv1d(d, c_out, kernel_size=3, padding=1, padding_mode="circular",
                               bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cross):
        x, t1 = self.d1(x + self.drop(self.self_attn(x, x, x)))
        x, t2 = self.d2(x + self.drop(self.cross_attn(x, cross, cross)))
        x, t3 = self.d3(x + self.ff(x.transpose(1, 2)).transpose(1, 2))
        trend = self.trend((t1 + t2 + t3).permute(0, 2, 1)).permute(0, 2, 1)
        return x, trend


class _Embedding(nn.Module):
    """``DataEmbedding_wo_pos``: circular token convolution plus covariate embedding."""

    def __init__(self, n_value, n_cov, d, dropout):
        super().__init__()
        self.value = nn.Conv1d(n_value, d, kernel_size=3, padding=1, padding_mode="circular",
                               bias=False)
        self.cov = nn.Linear(n_cov, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cov):
        return self.drop(self.value(x.permute(0, 2, 1)).permute(0, 2, 1) + self.cov(cov))


class _Autoformer(nn.Module):
    def __init__(self, n_hist, n_inj, d, dropout, label_len=5, kernel=5):
        super().__init__()
        self.label_len = label_len
        self.decomp = _Decomp(kernel)
        self.enc_embedding = _Embedding(n_hist, n_inj, d, dropout)
        self.dec_embedding = _Embedding(1, n_inj, d, dropout)
        self.encoder = nn.ModuleList([_EncoderLayer(d, 4, kernel, dropout) for _ in range(2)])
        self.enc_norm = _SeasonalNorm(d)
        self.decoder = nn.ModuleList([_DecoderLayer(d, 4, kernel, dropout, c_out=1)])
        self.dec_norm = _SeasonalNorm(d)
        self.projection = nn.Linear(d, 1)

    def forward(self, hist, oil, inj_hist, inj_future, horizon):
        # oil: field-oil history in the (scalar) units of the standardised target
        mean = oil.mean(dim=1, keepdim=True).repeat(1, horizon, 1)
        seasonal, trend = self.decomp(oil)
        trend = torch.cat([trend[:, -self.label_len:], mean], dim=1)
        seasonal = torch.cat([seasonal[:, -self.label_len:], torch.zeros_like(mean)], dim=1)
        enc = self.enc_embedding(hist, inj_hist)
        for layer in self.encoder:
            enc = layer(enc)
        enc = self.enc_norm(enc)
        dec = self.dec_embedding(seasonal,
                                 torch.cat([inj_hist[:, -self.label_len:], inj_future], dim=1))
        for layer in self.decoder:
            dec, residual_trend = layer(dec, enc)
            trend = trend + residual_trend
        out = trend + self.projection(self.dec_norm(dec))
        return out[:, -horizon:, 0]


# ----------------------------------------------------------------- wrapper
class SequenceModel:
    """Common fit/predict interface; MSE training with Adam and validation early stopping."""

    def __init__(self, kind, seed=42, params=None, max_epochs=300, patience=30, batch_size=128,
                 relative_oil=False):
        if kind not in SEQUENCE_KINDS:
            raise ValueError(f"unknown sequence model {kind!r}")
        self.kind, self.seed, self.relative_oil = kind, seed, relative_oil
        self.params = {"d_model": 64, "learning_rate": 1e-3, "dropout": 0.1, **(params or {})}
        self.max_epochs, self.patience, self.batch_size = max_epochs, patience, batch_size

    def _scale(self, histories, actions):
        h = np.asarray(histories, dtype=float)
        u = np.asarray(actions, dtype=float)
        m = u.shape[-1]
        inj = np.concatenate([h[:, :, :m], u], axis=1)
        inj = (inj - self.inj_mean_) / self.inj_std_
        other = (h[:, :, m:] - self.other_mean_) / self.other_std_
        n_oil = (h.shape[-1] - m) // 2
        oil = h[:, :, m:m + n_oil].sum(axis=-1, keepdims=True)
        if self.relative_oil:     # field target: change relative to the last four-week median
            level = np.median(oil[:, -4:], axis=1, keepdims=True)
            oil = (oil - level) / (level + 5000.0)
        oil = (oil - self.y_mean_) / self.y_std_
        return (torch.tensor(other, dtype=torch.float32),
                torch.tensor(oil, dtype=torch.float32),
                torch.tensor(inj[:, : h.shape[1]], dtype=torch.float32),
                torch.tensor(inj[:, h.shape[1]:], dtype=torch.float32))

    def _forward(self, other, oil, inj_hist, inj_future):
        horizon = inj_future.shape[1]
        if self.kind == "Autoformer":
            hist = torch.cat([inj_hist, other], dim=-1)
            return self.net(hist, oil, inj_hist, inj_future, horizon)
        n, length, _ = other.shape
        flag = torch.cat([torch.zeros(n, length, 1), torch.ones(n, horizon, 1)], dim=1)
        rest = torch.cat([other, torch.zeros(n, horizon, other.shape[-1])], dim=1)
        tokens = torch.cat([torch.cat([inj_hist, inj_future], dim=1), rest, flag], dim=-1)
        return self.net(tokens, horizon)

    def fit(self, histories, actions, truth, validation_data):
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        h, u = np.asarray(histories, dtype=float), np.asarray(actions, dtype=float)
        m = u.shape[-1]
        inj = np.concatenate([h[:, :, :m], u], axis=1).reshape(-1, m)
        self.inj_mean_, self.inj_std_ = inj.mean(0), inj.std(0) + 1e-9
        other = h[:, :, m:].reshape(-1, h.shape[-1] - m)
        self.other_mean_, self.other_std_ = other.mean(0), other.std(0) + 1e-9
        truth = np.asarray(truth, dtype=float)
        # one scalar standardisation for all horizon steps, shared with the field-oil history
        self.y_mean_, self.y_std_ = float(truth.mean()), float(truth.std()) + 1e-9
        p = self.params
        if self.kind == "Autoformer":
            self.net = _Autoformer(h.shape[-1], m, p["d_model"], p["dropout"])
        else:
            self.net = _TokenModel(self.kind, h.shape[-1] + 1, p["d_model"], p["dropout"],
                                   h.shape[1] + u.shape[1])
        x = self._scale(h, u)
        y = torch.tensor((truth - self.y_mean_) / self.y_std_, dtype=torch.float32)
        vx = self._scale(*validation_data[:2])
        vy = torch.tensor((np.asarray(validation_data[2]) - self.y_mean_) / self.y_std_,
                          dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=p["learning_rate"], weight_decay=1e-4)
        gen = torch.Generator().manual_seed(self.seed + 1)
        best, best_state, stale = np.inf, None, 0
        for epoch in range(self.max_epochs):
            self.net.train()
            order = torch.randperm(len(y), generator=gen)
            for start in range(0, len(y), self.batch_size):
                idx = order[start:start + self.batch_size]
                loss = F.mse_loss(self._forward(*(t[idx] for t in x)), y[idx])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val = F.mse_loss(self._forward(*vx), vy).item()
            if val < best - 1e-6:
                best, stale = val, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.epochs_ = epoch + 1
        return self

    def predict(self, histories, actions):
        self.net.eval()
        x = self._scale(histories, actions)
        out = []
        with torch.no_grad():
            for start in range(0, len(x[0]), 2048):
                out.append(self._forward(*(t[start:start + 2048] for t in x)).numpy())
        return np.concatenate(out) * self.y_std_ + self.y_mean_
