"""Proportional-odds / CORAL ordinal RPE head, fit by scipy MLE (§10.2-10.3) — Layer 5c.

A single shared **effort score** ``η_i = z(x_i)·β + b_{u_i}`` is compared against ordered
thresholds ``τ_1 ≤ … ≤ τ_{K-1}``:

    P(RPE_i > level_k) = σ(η_i − τ_k)

Rank-monotonicity (§10.2) is guaranteed by construction — the thresholds are increasing, so
``P(RPE > k)`` is decreasing in ``k`` with no post-hoc fix. Ordering is enforced by the
reparameterization ``τ_k = τ_1 + Σ_{j≤k} softplus(δ_j)``.

Hierarchy (§10.3): the per-user effect ``b_u`` is ridge-penalized — convex partial pooling,
the same device as the RIR head. It shrinks a thin/new user toward the population (``b_u→0``)
and lets a heavy user personalize. **``b_u`` is the object of interest** (§10.5): it is the
user's systematic over/under-reporting, and it is what the head recovers that the RIR head
cannot.

Fit by L-BFGS-B from a fixed init — deterministic, no seed. The gradient is finite-difference
(the design is tiny: a handful of features + 8 thresholds + one intercept per user).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from wristset.models.rpe.dataset import LEVELS, NUMERIC_COLUMNS, RpeDataset

__all__ = ["OrdinalRpeModel"]

_CLIP = 1e-9


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


@dataclass
class OrdinalRpeModel:
    """A fitted ordinal RPE head. Build with :meth:`fit`."""

    beta: np.ndarray            # (n_design,) shared effort weights
    thresholds: np.ndarray      # (K-1,) ordered cutpoints τ_k
    b_u: np.ndarray             # (n_users,) per-user effect
    users: list[str]
    mu: np.ndarray
    sd: np.ndarray
    ex_vocab: list[str]         # ex_vocab[0] is the reference level
    columns: tuple[str, ...] = NUMERIC_COLUMNS
    _user_of: dict = field(default_factory=dict)

    # --- design / score ---------------------------------------------------------

    def _design(self, X: np.ndarray, exercise: np.ndarray) -> np.ndarray:
        """[standardized numerics, exercise dummies] — NO intercept (τ carries the offset)."""
        Xs = (X - self.mu) / self.sd
        n = X.shape[0]
        cols = [Xs]
        for level in self.ex_vocab[1:]:
            cols.append((exercise == level).astype(np.float64).reshape(n, 1))
        return np.hstack(cols)

    def _eta(self, ds: RpeDataset) -> np.ndarray:
        D = self._design(ds.X, ds.exercise)
        row_users = [ds.users[i] for i in ds.user_idx]
        u = np.array([self.user_effect(uid) for uid in row_users])
        return D @ self.beta + u

    # --- prediction -------------------------------------------------------------

    def user_effect(self, user_id: str) -> float:
        """The user's effort offset, or 0.0 (population) for an unseen lifter (§10.3)."""
        i = self._user_of.get(user_id)
        return float(self.b_u[i]) if i is not None else 0.0

    def predict_proba(self, ds: RpeDataset) -> np.ndarray:
        """PMF over the ``LEVELS`` grid for each set, from the cumulative probabilities."""
        eta = self._eta(ds)
        surv = _sigmoid(eta[:, None] - self.thresholds[None, :])  # (n, K-1) = P(RPE>k)
        n, K = surv.shape[0], len(LEVELS)
        P = np.empty((n, K))
        P[:, 0] = 1.0 - surv[:, 0]
        P[:, 1:K - 1] = surv[:, :-1] - surv[:, 1:]
        P[:, K - 1] = surv[:, -1]
        P = np.clip(P, 0.0, None)
        return P / P.sum(axis=1, keepdims=True)

    def expected_rpe(self, ds: RpeDataset) -> np.ndarray:
        return self.predict_proba(ds) @ LEVELS

    def predict_level(self, ds: RpeDataset) -> np.ndarray:
        return LEVELS[np.argmax(self.predict_proba(ds), axis=1)]

    # --- fitting ----------------------------------------------------------------

    @classmethod
    def fit(cls, ds: RpeDataset, *, l2_user: float = 1.0, l2_ridge: float = 0.5) -> "OrdinalRpeModel":
        mu = ds.X.mean(axis=0)
        sd = ds.X.std(axis=0)
        sd[sd < 1e-9] = 1.0
        ex_vocab = sorted(set(ds.exercise.tolist()))

        model = cls(
            beta=np.zeros(1), thresholds=np.zeros(len(LEVELS) - 1),
            b_u=np.zeros(len(ds.users)), users=list(ds.users),
            mu=mu, sd=sd, ex_vocab=ex_vocab,
            _user_of={u: i for i, u in enumerate(ds.users)},
        )
        D = model._design(ds.X, ds.exercise)
        y = ds.y
        uidx = ds.user_idx
        n_beta = D.shape[1]
        n_thr = len(LEVELS) - 1          # K-1 thresholds
        n_user = len(ds.users)
        # threshold target matrix T[i,k] = 1[y_i > k]
        T = (y[:, None] > np.arange(n_thr)[None, :]).astype(np.float64)

        def unpack(theta):
            beta = theta[:n_beta]
            t0 = theta[n_beta]
            deltas = theta[n_beta + 1: n_beta + n_thr]
            b_u = theta[n_beta + n_thr:]
            tau = t0 + np.concatenate([[0.0], np.cumsum(_softplus(deltas))])
            return beta, tau, b_u

        def objective(theta):
            beta, tau, b_u = unpack(theta)
            eta = D @ beta + b_u[uidx]
            surv = _sigmoid(eta[:, None] - tau[None, :])
            s = np.clip(surv, _CLIP, 1.0 - _CLIP)
            nll = -np.sum(T * np.log(s) + (1.0 - T) * np.log(1.0 - s))
            nll += 0.5 * l2_ridge * np.sum(beta ** 2)
            nll += 0.5 * l2_user * np.sum(b_u ** 2)
            return nll

        # init: thresholds evenly spread (delta=0 -> softplus(0)=ln2 spacing), rest zero
        theta0 = np.zeros(n_beta + n_thr + n_user)
        res = minimize(objective, theta0, method="L-BFGS-B",
                       options={"maxiter": 1000, "ftol": 1e-10})
        beta, tau, b_u = unpack(res.x)
        model.beta = beta
        model.thresholds = tau
        model.b_u = b_u
        return model
