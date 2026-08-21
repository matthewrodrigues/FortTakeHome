"""Hazard -> reps-in-reserve distribution (§9.4) — Layer 5b.

``E[RIR | r] = Σ_k Π_{j≤k} (1 − h(r+j))`` needs hazards for reps the lifter has not done
yet, so the causal features must be projected forward. For the prototype this is a **linear
extrapolation of the fitted causal trends** — the dominant, documented source of error
(§9.4) — with the horizon ``K`` capped at 8 because the projection degrades with distance.

The output is a full distribution ``P(RIR=0), P(RIR=1), ...`` (§9.4), which is more honest
than a point estimate: the mass at ``RIR=0`` is the model saying "this looks like the last
rep". Index ``K`` of the returned vector is the tail ``P(RIR ≥ K)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wristset.features import SetFeatures
from wristset.models.rir.dataset import feature_row
from wristset.models.rir.hazard import HazardModel

__all__ = ["RIRPrediction", "rir_distribution", "CONFIDENT_HORIZON"]


#: Horizon beyond which an E[RIR] estimate is treated as low-confidence (§8.2).
#:
#: Chosen empirically, NOT from the distribution's own spread. Measured on held-out users
#: across three corpora (n=527 predictions), mean |E[RIR] - true| by true horizon:
#:
#:     true RIR 0-2 -> 0.89     true RIR 4-6  -> 2.63
#:     true RIR 2-4 -> 1.58     true RIR 6+   -> 1.72
#:
#: Error roughly doubles past ~3 reps of horizon, because §9.4's forward projection is
#: extrapolating further from observed data. ``expected_rir`` is the best *observable*
#: proxy for that horizon (corr +0.139 with |error|; the true horizon is not available at
#: prediction time).
#:
#: **The effect is weak, and that is the honest finding.** At this threshold the gate keeps
#: 34% of predictions with mean |error| 1.61 against 1.90 for those it drops — a separation
#: of 0.29 reps on a 1.8-rep mean error. Sweeping the threshold shows separation rising
#: monotonically (0.17 at 1.5 -> 0.43 at 5.0) with no peak, which means there is no genuine
#: confidence *boundary* to find, only the weak underlying correlation. 3.0 is therefore
#: chosen as a defensible product line ("3+ reps left is a forecast, not a reading") rather
#: than a fitted optimum. §8.2 gets a gate that is directionally correct instead of one that
#: is actively backwards, but it should not be presented as a strong reliability signal.
CONFIDENT_HORIZON: float = 3.0


@dataclass
class RIRPrediction:
    """P(RIR=k) for k=0..K, with index K holding the tail P(RIR>=K), plus the mean."""

    dist: np.ndarray
    expected_rir: float
    K: int

    @property
    def predictive_std(self) -> float:
        """Standard deviation of the predicted RIR distribution.

        **Do not use this as a confidence gate** — use :attr:`is_confident`. Measured on
        held-out users across three corpora, this std is overconfident by roughly 3x
        (median 0.5 against median |error| 1.6), only 20-30% of errors fall within one
        std (a calibrated distribution would give ~68%), and it is *anti*-correlated with
        error (r = -0.08): the predictions it calls most certain are the least accurate.

        The cause is the §9.4 projection saturating the hazard — when the extrapolated
        features cross the fitted decision boundary the PMF collapses onto a single level
        *and* is wrong, so the distribution's own shape is contaminated by the very defect
        a confidence measure would need to report. It is exposed for completeness and
        because §11's divergence *percentile* (which needs rank ordering, not spread) is
        well behaved on the same distribution.
        """
        k = np.arange(self.dist.shape[0])
        var = float(np.sum(self.dist * (k - self.expected_rir) ** 2))
        return float(np.sqrt(max(var, 0.0)))

    @property
    def is_confident(self) -> bool:
        """Weak advisory signal on forecast horizon. **Not a §8.2 confidence gate.**

        §8.2 specifies dropping the RIR term from the composite when
        ``rir_predictive_std`` is too wide. That gate is **deferred**: this head has no
        calibrated uncertainty estimate to drive it (see :attr:`predictive_std`), and the
        best available substitute — forecast horizon — separates only 0.29 reps of mean
        error on a 1.8-rep baseline, with no threshold optimum (see
        :data:`CONFIDENT_HORIZON`). A gate that weak would gate arbitrarily while implying
        a reliability guarantee the model cannot make, so Phase 8 keeps the RIR term
        unconditionally rather than dropping it on a signal this thin.

        Use this only to *annotate* an estimate as near-term versus far-horizon — which is
        a statement about how far the §9.4 projection is extrapolating, not about whether
        the answer is right. Re-enable the real gate once the projection is calibrated.
        """
        return self.expected_rir <= CONFIDENT_HORIZON


#: Floor on projected concentric velocity, as a fraction of the velocity at the rep being
#: projected from. Guards against the projection walking outside the range the model was
#: fit on: unbounded linear extrapolation reached the 1e-3 clamp within ~4 reps, far below
#: the 0.09-0.12 band the model learned as "this is the failed attempt".
#:
#: Measured to matter far less than ``VEL_DECAY_DAMPING`` — across a floor sweep of
#: 0.0-0.70 the RIR MAE moved <0.15 — so it is kept at 0 (inactive) rather than adding a
#: parameter that does nothing. Retained because a future non-linear projection, or a
#: less-regularized fit, can re-expose the same failure mode.
VEL_FLOOR_FRACTION: float = 0.0

#: Damping exponent on the projected velocity decay: the decay applied at horizon ``j`` is
#: ``slope * j**VEL_DECAY_DAMPING``. Strictly linear (1.0) overstates the decline, because
#: fatigue decay decelerates as the lifter approaches their floor; measured, 0.75 cut RIR
#: MAE from 1.92 to 1.81 while removing most of the -1.21 rep pessimistic bias.
VEL_DECAY_DAMPING: float = 0.75


def _project(base: dict, j: int, r: int, slope_vel: float) -> dict:
    """Push a rep-``r`` feature row forward by ``j`` reps (§9.4 approximation).

    Damped and bounded rather than strictly linear. See ``VEL_FLOOR_FRACTION`` — the naive
    linear form drove features outside the fitted range within 2-5 reps, saturating every
    projected hazard and collapsing E[RIR] toward 0 no matter the true reps remaining.
    """
    f = dict(base)
    f["rep_index"] = float(r + j)

    # Velocity decays along the fitted slope, damped with distance and floored relative to
    # the current rep so the projection stays inside the model's fitted domain.
    v0 = base["conc_mean_vel"]
    f["conc_mean_vel"] = max(v0 + slope_vel * (j ** VEL_DECAY_DAMPING),
                             VEL_FLOOR_FRACTION * v0, 1e-3)

    # Cumulative "to-date change" features continue at their observed average per-rep rate
    # rather than being rescaled by (r+j)/r: these are already cumulative, so multiplying
    # the accumulated total compounds the growth instead of extending it.
    for k in ("vel_loss_to_date", "ecc_change_to_date", "tremor_change_to_date"):
        per_rep = base[k] / r if r > 0 else 0.0
        f[k] = base[k] + per_rep * (j ** VEL_DECAY_DAMPING)

    # load and path divergence held constant (documented approximation)
    return f


def rir_distribution(
    model: HazardModel,
    sf: SetFeatures,
    r: int,
    *,
    exercise: str,
    user_id: str = "",
    K: int = 8,
) -> RIRPrediction:
    """Distribution over reps-in-reserve after completing rep ``r`` (§9.4)."""
    rep = next((x for x in sf.reps if x.rep_index == r), None)
    if rep is None:
        raise ValueError(f"rep {r} not in set")
    base = feature_row(sf, rep)
    slope_vel = sf.causal.get(r, {}).get("causal_conc_mean_vel_slope") or 0.0

    # future hazards h(r+1..r+K)
    p = np.array([
        model.predict_hazard(_project(base, j, r, slope_vel), exercise, user_id)
        for j in range(1, K + 1)
    ])

    survive = np.concatenate([[1.0], np.cumprod(1.0 - p)])  # survive[k] = P(reach r+k)
    dist = np.empty(K + 1)
    dist[:K] = survive[:K] * p            # P(RIR=m) = survive m reps, then fail
    dist[K] = survive[K]                  # tail: survived all K projected reps
    expected = float(np.sum(np.arange(K + 1) * dist))
    return RIRPrediction(dist=dist, expected_rir=expected, K=K)
