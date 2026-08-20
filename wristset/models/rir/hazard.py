"""Logistic discrete-time hazard, fit by direct MLE (§9.2-9.3) — Layer 5b.

``logit h(r) = β·[1, standardized causal features, exercise dummies] + u[user]``.

The objective is the person-period Bernoulli negative log-likelihood — which equals the
§9.3 censored likelihood — plus two convex penalties:

* ``½·l2_user·Σ u²`` on the per-user intercepts. This L2 penalty *is* partial pooling: it
  shrinks each user toward the population intercept in proportion to how little that user's
  data pulls against it, so a new/thin user defaults to the population fit and a heavy user
  personalizes. It is the convex, deterministic stand-in for the §9.2 random intercept.
* ``½·l2_ridge·Σ β²`` (intercept excluded) for numerical stability under the standardized,
  mildly-collinear design.

The problem is convex, so L-BFGS-B from a zero start reaches the unique optimum every run —
the fit is deterministic with no seed. An analytic gradient keeps it fast and exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from wristset.models.rir.dataset import NUMERIC_COLUMNS, PersonPeriod

__all__ = ["HazardModel"]

_CLIP = 1e-9  # keep log() and hazards strictly inside (0, 1)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


@dataclass
class HazardModel:
    """A fitted hazard head. Build with :meth:`fit`; never instantiated directly."""

    beta: np.ndarray            # (n_design,) fixed effects
    u: np.ndarray               # (n_users,) per-user intercepts
    users: list[str]
    mu: np.ndarray              # (n_numeric,) standardization mean
    sd: np.ndarray              # (n_numeric,) standardization std
    ex_vocab: list[str]         # exercise levels; ex_vocab[0] is the reference
    columns: tuple[str, ...] = NUMERIC_COLUMNS
    _user_of: dict = field(default_factory=dict)

    # --- design construction ----------------------------------------------------

    def _design(self, X: np.ndarray, exercise: np.ndarray) -> np.ndarray:
        """Assemble [1, standardized numerics, exercise dummies] for raw rows."""
        Xs = (X - self.mu) / self.sd
        n = X.shape[0]
        cols = [np.ones((n, 1)), Xs]
        for level in self.ex_vocab[1:]:  # drop-first reference coding
            cols.append((exercise == level).astype(np.float64).reshape(n, 1))
        return np.hstack(cols)

    # --- prediction -------------------------------------------------------------

    def user_effect(self, user_id: str) -> float:
        """The user's intercept, or 0.0 (population) for an unseen lifter (§9.2)."""
        i = self._user_of.get(user_id)
        return float(self.u[i]) if i is not None else 0.0

    def predict_hazard_rows(self, pp: PersonPeriod) -> np.ndarray:
        """Hazard for every row of a person-period table. Rows from unseen users use the
        population intercept, which is exactly what the by-user gate (§9.6) measures."""
        D = self._design(pp.X, pp.exercise)
        row_users = [pp.users[i] for i in pp.user_idx]
        u = np.array([self.user_effect(uid) for uid in row_users])
        return _sigmoid(D @ self.beta + u)

    def predict_hazard(self, features: dict, exercise: str, user_id: str) -> float:
        """Hazard for a single rep described by a :func:`feature_row` dict."""
        X = np.array([[features[c] for c in NUMERIC_COLUMNS]], dtype=np.float64)
        D = self._design(X, np.array([exercise], dtype=object))
        return float(_sigmoid(D @ self.beta + self.user_effect(user_id))[0])

    # --- fitting ----------------------------------------------------------------

    #: Default ridge on the fixed effects. Deliberately strong, not merely a numerical
    #: nudge: the causal drivers are heavily collinear (``conc_mean_vel`` correlates 0.92
    #: with ``vel_loss_to_date``), so a weak ridge lets the fit hand one feature a huge
    #: coefficient (|beta| up to 38) and give the others compensating negative weights.
    #: Predictions stay fine, but the hazard becomes a near step-function in velocity, and
    #: §9.4's forward projection then saturates it at 1.0 within a couple of reps —
    #: collapsing E[RIR] regardless of how many reps actually remain.
    #:
    #: 0.5 improves RIR MAE on every corpus seed tested (1.81->1.72, 2.72->1.96,
    #: 2.02->1.74) at no C-index cost, and gives the best mean calibration across seeds.
    #: Stronger values (5.0) improve MAE slightly further but do not help calibration:
    #: no setting is well-calibrated on every seed, because a weak ridge is near-perfectly
    #: calibrated on a *lucky* corpus (0.04 on seed 5) and badly miscalibrated elsewhere
    #: (0.33, 0.29). Calibration variance across corpora is a real open limitation, not
    #: something this constant can resolve — see the Phase-6 notes.
    DEFAULT_L2_RIDGE: float = 0.5

    @classmethod
    def fit(cls, pp: PersonPeriod, *, l2_user: float = 1.0, l2_ridge: float | None = None) -> "HazardModel":
        if l2_ridge is None:
            l2_ridge = cls.DEFAULT_L2_RIDGE
        mu = pp.X.mean(axis=0)
        sd = pp.X.std(axis=0)
        sd[sd < 1e-9] = 1.0
        ex_vocab = sorted(set(pp.exercise.tolist()))

        model = cls(
            beta=np.zeros(1), u=np.zeros(len(pp.users)), users=list(pp.users),
            mu=mu, sd=sd, ex_vocab=ex_vocab,
            _user_of={u: i for i, u in enumerate(pp.users)},
        )
        D = model._design(pp.X, pp.exercise)
        y = pp.y
        uidx = pp.user_idx
        n_beta = D.shape[1]
        n_user = len(pp.users)
        # ridge mask: penalize everything except the global intercept (column 0)
        ridge_mask = np.ones(n_beta)
        ridge_mask[0] = 0.0

        def unpack(theta):
            return theta[:n_beta], theta[n_beta:]

        def objective(theta):
            beta, u = unpack(theta)
            eta = D @ beta + u[uidx]
            h = _sigmoid(eta)
            hc = np.clip(h, _CLIP, 1.0 - _CLIP)
            nll = -np.sum(y * np.log(hc) + (1.0 - y) * np.log(1.0 - hc))
            nll += 0.5 * l2_ridge * np.sum((ridge_mask * beta) ** 2)
            nll += 0.5 * l2_user * np.sum(u ** 2)

            resid = h - y
            g_beta = D.T @ resid + l2_ridge * ridge_mask * beta
            g_u = np.bincount(uidx, weights=resid, minlength=n_user) + l2_user * u
            return nll, np.concatenate([g_beta, g_u])

        theta0 = np.zeros(n_beta + n_user)
        res = minimize(objective, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-10})
        beta, u = unpack(res.x)
        model.beta = beta
        model.u = u
        return model
