"""Effort narrative (§8.6) — the effort half, mirroring ``execution.py``.

Sourced from the RIR predictive distribution (§9.4) and the divergence percentile (§11.1),
per §8.6's worked example::

    "Estimated 1-2 reps left at set end. You reported RPE 7; movement suggested closer to 9."

Two §8.7 disciplines are load-bearing here and are enforced by tests:

* **Measured change only, no coaching.** The narrative states what was observed and what
  the model estimated; it never prescribes technique.
* **No false precision.** ``E[RIR]`` carries ~1.8 reps of mean absolute error, so the text
  reports a *range* ("1-2 reps left"), never a decimal. Rendering 2.37 as "2.4 reps left"
  would imply a resolution the model does not have.
"""

from __future__ import annotations

import numpy as np

from wristset.divergence import Divergence

__all__ = ["effort_narrative", "rir_range"]


def rir_range(dist: np.ndarray, mass: float = 0.6) -> tuple[int, int]:
    """Smallest contiguous RIR interval holding at least ``mass`` of the distribution.

    A range rather than a point estimate: §9.4's forward projection makes ``E[RIR]``
    accurate to ~1.8 reps, so a single number would overstate what the model knows. The
    interval is the honest summary of the same distribution.
    """
    d = np.asarray(dist, dtype=np.float64)
    total = d.sum()
    if total <= 0:
        raise ValueError("RIR distribution has no mass")
    d = d / total

    best = (0, d.shape[0] - 1)
    best_width = d.shape[0]
    for lo in range(d.shape[0]):
        acc = 0.0
        for hi in range(lo, d.shape[0]):
            acc += d[hi]
            if acc >= mass:
                if (hi - lo) < best_width:
                    best_width, best = hi - lo, (lo, hi)
                break
    return int(best[0]), int(best[1])


def _reps_left_clause(lo: int, hi: int, k: int) -> str:
    """Phrase the RIR interval, marking the open-ended top of the K-capped distribution."""
    if lo >= k:
        return f"{k}+ reps left"
    if hi >= k:
        return f"{lo}+ reps left"
    if lo == hi == 0:
        return "no reps left"          # not "about 0 reps left"
    if lo == hi:
        return f"about {lo} rep{'s' if lo != 1 else ''} left"
    return f"{lo}-{hi} reps left"


def effort_narrative(
    dist: np.ndarray | None,
    divergence: Divergence | None,
    *,
    mass: float = 0.6,
) -> str:
    """A short measured-change effort narrative for one set (§8.6).

    Leads with the estimated reps-in-reserve range at set end, then — only when §11.3's
    percentile band is breached — states the reported-vs-mechanical mismatch. An aligned
    set says so briefly rather than manufacturing a finding.
    """
    parts: list[str] = []

    if dist is not None:
        d = np.asarray(dist, dtype=np.float64)
        lo, hi = rir_range(d, mass=mass)
        parts.append(f"Estimated {_reps_left_clause(lo, hi, d.shape[0] - 1)} at set end.")

    if divergence is not None:
        if divergence.direction == "under_reporting":
            parts.append(
                f"You reported RPE {divergence.reported_rpe:g}; "
                f"movement suggested closer to {divergence.mechanical_rpe:.0f}."
            )
        elif divergence.direction == "over_reporting":
            parts.append(
                f"You reported RPE {divergence.reported_rpe:g}, above the "
                f"{divergence.mechanical_rpe:.0f} the movement suggested."
            )
        else:
            parts.append("Reported effort matched the movement.")

    return " ".join(parts) if parts else "Effort estimate unavailable."
