"""Action-Conditioned Temporal Response Network (ACTRN), level-response design.

The forecast for a history H and a candidate injection plan U is split into a level that
depends on the history only and a response to the plan that may depend on both:

    y_hat = f(x_H) + g(x_H, x_U)
    f(x_H)      = x_H W_B + b
    g(x_H, x_U) = x_U W_A + (z_H * z_U) W_I + z_A W_Z,
    z_H = tanh(x_H W_h + b_h),  z_U = tanh(x_U W_a + b_a),  z_A = tanh(x_U W_u + b_u).

Only g can change the ranking of candidate plans issued from the same state, and only its
interaction path z_H * z_U lets that ranking depend on the state.  When the training data
contain several plans per state (simulated restarts), g is learned from within-state
contrasts: W_A by ridge on context-centred data and the nonlinear paths by a centred loss,
so the large between-state variation in production level cannot leak into the plan ranking.
The level path is fitted afterwards on y - g.  Without such contrasts (field data) all
linear paths are fitted jointly by ridge and the nonlinear paths learn the residual by MSE.
Nonlinear paths start at zero output, are early-stopped on validation data and averaged
over an ensemble of initialisation seeds.

The history representation contains response proxies (within-window correlations and joint
ridge coefficients of production increments on injector rates), so the interaction path can tell
which injectors the producers currently respond to.

Earlier designs are preserved at git tags ``actrn-v1-registered``, ``actrn-v2-registered`` and
``actrn-v3-registered``.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from action_conditioned_decision import (encode_actions, history_features,
                                         saturation_scale)


def center_by_group(values: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """Subtract the group mean from every row (groups are 0..n_groups-1)."""
    sums = np.zeros((n_groups, values.shape[1]))
    np.add.at(sums, groups, values)
    counts = np.maximum(np.bincount(groups, minlength=n_groups), 1)[:, None]
    return values - (sums / counts)[groups]


def _nmae(truth: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(truth - pred)) / np.mean(np.abs(truth)))


class _ResponsePaths:
    """Nonlinear response paths (z_H * z_U) W_I + z_A W_Z, or a concatenation MLP."""

    def __init__(self, latent_dim, l2, seed, interaction="product", action_path=True):
        self.d, self.l2, self.seed = latent_dim, l2, seed
        self.interaction, self.action_path = interaction, action_path

    def init(self, h_dim, a_dim, out_dim):
        rng = np.random.default_rng(self.seed)

        def weight(n_in, n_out):
            return rng.normal(0.0, np.sqrt(2.0 / (n_in + n_out)), (n_in, n_out))

        d, p = self.d, {}
        if self.interaction == "product":
            p.update(Wh=weight(h_dim, d), bh=np.zeros(d), Wa=weight(a_dim, d), ba=np.zeros(d),
                     Wi=np.zeros((d, out_dim)))
        elif self.interaction == "concat":
            p.update(Wc=weight(h_dim + a_dim, d), bc=np.zeros(d), Wi=np.zeros((d, out_dim)))
        if self.action_path:
            p.update(Wu=weight(a_dim, d), bu=np.zeros(d), Wz=np.zeros((d, out_dim)))
        self.p = p

    def forward(self, h, a):
        p, c, out = self.p, {}, 0.0
        if self.interaction == "product":
            c["zh"] = np.tanh(h @ p["Wh"] + p["bh"])
            c["za"] = np.tanh(a @ p["Wa"] + p["ba"])
            c["zi"] = c["zh"] * c["za"]
            out = out + c["zi"] @ p["Wi"]
        elif self.interaction == "concat":
            c["x"] = np.c_[h, a]
            c["zi"] = np.tanh(c["x"] @ p["Wc"] + p["bc"])
            out = out + c["zi"] @ p["Wi"]
        if self.action_path:
            c["zu"] = np.tanh(a @ p["Wu"] + p["bu"])
            out = out + c["zu"] @ p["Wz"]
        return out, (h, a, c)

    def backward(self, d_out, cache):
        h, a, c = cache
        p, l2, g = self.p, self.l2, {}
        if self.interaction != "none":
            g["Wi"] = c["zi"].T @ d_out + l2 * p["Wi"]
            d_zi = d_out @ p["Wi"].T
        if self.interaction == "product":
            d_h = d_zi * c["za"] * (1.0 - c["zh"] ** 2)
            d_a = d_zi * c["zh"] * (1.0 - c["za"] ** 2)
            g.update(Wh=h.T @ d_h + l2 * p["Wh"], bh=d_h.sum(0),
                     Wa=a.T @ d_a + l2 * p["Wa"], ba=d_a.sum(0))
        elif self.interaction == "concat":
            d_c = d_zi * (1.0 - c["zi"] ** 2)
            g.update(Wc=c["x"].T @ d_c + l2 * p["Wc"], bc=d_c.sum(0))
        if self.action_path:
            g["Wz"] = c["zu"].T @ d_out + l2 * p["Wz"]
            d_u = d_out @ p["Wz"].T * (1.0 - c["zu"] ** 2)
            g.update(Wu=a.T @ d_u + l2 * p["Wu"], bu=d_u.sum(0))
        return g


class _Linear:
    def __init__(self, coef, intercept):
        self.coef, self.intercept = coef, intercept

    def predict(self, x):
        return x @ self.coef.T + self.intercept


class ACTRN:
    def __init__(
        self,
        latent_dim: int = 16,
        learning_rate: float = 0.002,
        l2: float = 1e-3,
        max_epochs: int = 900,
        patience: int = 80,
        batch_contexts: int = 21,
        batch_size: int = 128,
        ensemble: int = 5,
        seed: int = 42,
        interaction: str = "product",
        action_path: bool = True,
        nonlinear: bool = True,
        contrastive: bool = True,
        action_features: str = "multiscale",
        history_proxies: str = "full",
        level_alphas: tuple = (0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000),
        action_alphas: tuple = (0.01, 0.1, 1, 10),
    ) -> None:
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.l2 = l2
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_contexts = batch_contexts
        self.batch_size = batch_size
        self.ensemble = ensemble
        self.seed = seed
        self.interaction = interaction
        self.action_path = action_path
        self.nonlinear = nonlinear and (interaction != "none" or action_path)
        self.contrastive = contrastive
        self.action_features = action_features
        self.history_proxies = history_proxies
        self.level_alphas = level_alphas
        self.action_alphas = action_alphas

    # ------------------------------------------------------------------ features
    def _features(self, histories, actions):
        xh = history_features(histories, proxies=self.history_proxies)
        xu = encode_actions(actions, self.saturation_scale_, self.action_features)
        return xh, xu

    def _scaled(self, histories, actions):
        xh, xu = self._features(histories, actions)
        return self.h_scaler.transform(xh), self.a_scaler.transform(xu)

    # ------------------------------------------------------------------ fit
    def fit(self, histories, actions, truth, validation_data, context_ids=None,
            validation_context_ids=None):
        self.saturation_scale_ = saturation_scale(actions)
        xh, xu = self._features(histories, actions)
        self.h_scaler = StandardScaler().fit(xh)
        self.a_scaler = StandardScaler().fit(xu)
        self.y_scaler = StandardScaler().fit(truth)
        h, a = self.h_scaler.transform(xh), self.a_scaler.transform(xu)
        y = self.y_scaler.transform(truth)
        vh, va = self._scaled(validation_data[0], validation_data[1])
        v_truth = np.asarray(validation_data[2])
        vy = self.y_scaler.transform(v_truth)
        self.paths_, self.epochs_ = [], []
        if self.contrastive and context_ids is not None and validation_context_ids is not None:
            self._fit_contrastive(h, a, y, vh, va, vy, v_truth, context_ids,
                                  validation_context_ids)
        else:
            self._fit_pooled(h, a, y, vh, va, vy, v_truth)
        return self

    def _ridge_by_validation(self, x, y, vx, v_truth, offset=0.0, v_offset=0.0):
        best = None
        for alpha in self.level_alphas:
            model = Ridge(alpha=alpha).fit(x, y - offset)
            score = _nmae(v_truth, self.y_scaler.inverse_transform(model.predict(vx) + v_offset))
            if best is None or score < best[0]:
                best = (score, alpha, model)
        return best[1], best[2]

    def _fit_contrastive(self, h, a, y, vh, va, vy, v_truth, context_ids, v_context_ids):
        _, groups = np.unique(context_ids, return_inverse=True)
        _, v_groups = np.unique(v_context_ids, return_inverse=True)
        n, vn = groups.max() + 1, v_groups.max() + 1
        y_c, vy_c = center_by_group(y, groups, n), center_by_group(vy, v_groups, vn)
        a_c = center_by_group(a, groups, n)
        best = None
        for alpha in self.action_alphas:
            model = Ridge(alpha=alpha, fit_intercept=False).fit(a_c, y_c)
            loss = float(np.mean((center_by_group(model.predict(va), v_groups, vn) - vy_c) ** 2))
            if best is None or loss < best[0]:
                best = (loss, alpha, model)
        self.action_alpha_, self.action_linear_ = best[1], best[2]
        if self.nonlinear:
            r = y - self.action_linear_.predict(a)
            vr = vy - self.action_linear_.predict(va)
            for k in range(self.ensemble):
                paths = self._new_paths(k, h.shape[1], a.shape[1], y.shape[1])
                self._train(paths, h, a, r, vh, va, vr, groups, n, v_groups, vn)
                self.paths_.append(paths)
        g, vg = self._response(h, a), self._response(vh, va)
        self.level_alpha_, self.level_ = self._ridge_by_validation(h, y, vh, v_truth, g, vg)

    def _fit_pooled(self, h, a, y, vh, va, vy, v_truth):
        alpha, joint = self._ridge_by_validation(np.c_[h, a], y, np.c_[vh, va], v_truth)
        k = h.shape[1]
        self.level_alpha_ = self.action_alpha_ = alpha
        self.level_ = _Linear(joint.coef_[:, :k], joint.intercept_)
        self.action_linear_ = _Linear(joint.coef_[:, k:], 0.0)
        if self.nonlinear:
            r = y - self.level_.predict(h) - self.action_linear_.predict(a)
            vr = vy - self.level_.predict(vh) - self.action_linear_.predict(va)
            for j in range(self.ensemble):
                paths = self._new_paths(j, h.shape[1], a.shape[1], y.shape[1])
                self._train(paths, h, a, r, vh, va, vr)
                self.paths_.append(paths)

    def _new_paths(self, k, h_dim, a_dim, out_dim):
        paths = _ResponsePaths(self.latent_dim, self.l2, self.seed + 1000 * k,
                               self.interaction, self.action_path)
        paths.init(h_dim, a_dim, out_dim)
        return paths

    @staticmethod
    def _loss(paths, h, a, r, groups=None, n=None):
        e = paths.forward(h, a)[0] - r
        if groups is not None:
            e = center_by_group(e, groups, n)
        return float(np.mean(e**2))

    def _train(self, paths, h, a, r, vh, va, vr, groups=None, n=None, v_groups=None, vn=None):
        """Adam with early stopping; centred (within-state) loss when groups are given."""
        rng = np.random.default_rng(paths.seed + 1)
        moments = {k: np.zeros_like(v) for k, v in paths.p.items()}
        velocities = {k: np.zeros_like(v) for k, v in paths.p.items()}
        best_loss = self._loss(paths, vh, va, vr, v_groups, vn)
        best = {k: v.copy() for k, v in paths.p.items()}
        members = None if groups is None else [np.flatnonzero(groups == g) for g in range(n)]
        stale = step = 0
        for epoch in range(self.max_epochs):
            if members is None:
                order = rng.permutation(len(h))
                batches = [order[s:s + self.batch_size] for s in range(0, len(h), self.batch_size)]
            else:
                order = rng.permutation(n)
                batches = [np.concatenate([members[g] for g in order[s:s + self.batch_contexts]])
                           for s in range(0, n, self.batch_contexts)]
            for idx in batches:
                pred, cache = paths.forward(h[idx], a[idx])
                e = pred - r[idx]
                if members is not None:
                    _, local = np.unique(groups[idx], return_inverse=True)
                    e = center_by_group(e, local, local.max() + 1)
                grads = paths.backward(2.0 * e / e.size, cache)
                step += 1
                for k, grad in grads.items():
                    moments[k] = 0.9 * moments[k] + 0.1 * grad
                    velocities[k] = 0.999 * velocities[k] + 0.001 * grad**2
                    m_hat = moments[k] / (1.0 - 0.9**step)
                    v_hat = velocities[k] / (1.0 - 0.999**step)
                    paths.p[k] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            loss = self._loss(paths, vh, va, vr, v_groups, vn)
            if loss < best_loss - 1e-7:
                best_loss, best, stale = loss, {k: v.copy() for k, v in paths.p.items()}, 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        paths.p = best
        self.epochs_.append(epoch + 1)

    # ------------------------------------------------------------------ predict
    def _response(self, h, a):
        out = self.action_linear_.predict(a)
        if self.paths_:
            out = out + np.mean([p.forward(h, a)[0] for p in self.paths_], axis=0)
        return out

    def predict(self, histories, actions):
        h, a = self._scaled(histories, actions)
        return self.y_scaler.inverse_transform(self.level_.predict(h) + self._response(h, a))
