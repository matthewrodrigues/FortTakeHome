"""Reported-vs-mechanical divergence (§11) — Layer 6.

Two independent routes to "how hard was that set":

* **Mechanical** — from the RIR hazard head: ``RPE_mech = 10 - RIR``, carrying the full
  predictive distribution the survival chain produces (§9.4, §11.1).
* **Reported** — what the lifter said.

§11.1 compares them as a **percentile, not a raw difference**: where does the reported RPE
fall within the mechanical predictive distribution? That builds the model's uncertainty into
the comparison for free — when the RIR distribution is wide, a given gap sits closer to the
middle of it and does not trigger, which is the desired behaviour (§11.3).

**Why a percentile works here when the std does not.** The same distribution's standard
deviation is badly miscalibrated (overconfident ~3x, anti-correlated with error), which is
why §8.2's confidence gate is deferred. A percentile only needs the distribution's *rank
ordering* to be sensible, not its width — and measured on held-out users the PIT of the true
outcome is roughly uniform, with 27% of sets outside the 10th/90th band against 20% for a
perfectly calibrated forecast. Rank information survives what spread information does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "LOW_PERCENTILE", "HIGH_PERCENTILE",
    "Divergence", "divergence_from_rir",
]

#: §11.3 alert band. Reported RPE outside these percentiles of the mechanical distribution
#: is flagged; inside, the two views agree well enough to say nothing.
LOW_PERCENTILE: float = 0.10
HIGH_PERCENTILE: float = 0.90


@dataclass
class Divergence:
    """Where the reported RPE sits inside the mechanically-implied RPE distribution."""

    percentile: float           # of reported_rpe within the RPE_mech distribution
    reported_rpe: float
    mechanical_rpe: float       # 10 - E[RIR], the distribution's mean
    direction: str              # "under_reporting" | "over_reporting" | "aligned"
    alert: bool

    @property
    def gap(self) -> float:
        """Reported minus mechanical, in RPE points. Reported for context only — §11.1
        deliberately thresholds on the percentile, which carries the model's uncertainty."""
        return float(self.reported_rpe - self.mechanical_rpe)


def divergence_from_rir(
    dist: np.ndarray,
    reported_rpe: float,
    *,
    low: float = LOW_PERCENTILE,
    high: float = HIGH_PERCENTILE,
) -> Divergence:
    """Locate ``reported_rpe`` within the RIR-implied RPE distribution (§11.1).

    ``dist`` is a :class:`~wristset.models.rir.RIRPrediction` PMF: ``dist[k] = P(RIR = k)``
    with the final entry holding the tail. Mapping ``RPE_mech = 10 - RIR`` reverses the
    axis, so probability mass at high RIR (lots left) becomes mass at low RPE.

    The percentile is computed with a half-mass correction at the reported level — the
    standard mid-P convention for a discrete distribution, which keeps a reported value
    sitting exactly on the distribution's mode near 0.5 rather than biased by which side of
    the tie it is counted on.
    """
    dist = np.asarray(dist, dtype=np.float64)
    total = dist.sum()
    if total <= 0:
        raise ValueError("RIR distribution has no mass")
    dist = dist / total

    rir_levels = np.arange(dist.shape[0], dtype=np.float64)
    rpe_levels = 10.0 - rir_levels                      # RPE_mech per RIR level (§11.1)
    mechanical = float(np.sum(dist * rpe_levels))       # = 10 - E[RIR]

    below = float(np.sum(dist[rpe_levels < reported_rpe]))
    at = float(np.sum(dist[np.isclose(rpe_levels, reported_rpe)]))
    percentile = below + 0.5 * at

    if percentile < low:
        # reported sits below almost all the mechanical mass: moved harder than it felt
        direction = "under_reporting"
    elif percentile > high:
        direction = "over_reporting"
    else:
        direction = "aligned"

    return Divergence(
        percentile=float(percentile),
        reported_rpe=float(reported_rpe),
        mechanical_rpe=mechanical,
        direction=direction,
        alert=direction != "aligned",
    )
