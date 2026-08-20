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

__all__ = ["RIRPrediction", "rir_distribution"]


@dataclass
class RIRPrediction:
    """P(RIR=k) for k=0..K, with index K holding the tail P(RIR>=K), plus the mean."""

    dist: np.ndarray
    expected_rir: float
    K: int


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
