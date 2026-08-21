"""RIR availability gate (§9.5) — when has a user supplied enough data to see an estimate?

The hazard at high-fatigue rep states is identified *only* from sets that actually reached
those states, so **failure sets are the binding constraint** (§9.5). A model fit on too few
of them is unstable, and an unstable reps-in-reserve number is worse than no number: it is
a confident-looking model output sitting next to two direct measurements.

§9.5's stated thresholds:

* **>= 30 failure sets per exercise** — the prototype target; the estimate is shown normally.
* **~15-29** — usable but unstable; §9.5 says "treat outputs as illustrative only", so the
  estimate is shown *labelled provisional*.
* **< 15** — not enough signal; the RIR term is withheld entirely and the §8.1 composite
  renormalises over form and RPE (which is exactly the cold-start path it already handles).

This is the RIR analogue of §7.3's baseline cold start: capability appears as data
accumulates, and its absence is reported as absence rather than filled with a default.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

__all__ = [
    "MIN_FAILURE_SETS", "TARGET_FAILURE_SETS",
    "RirReadiness", "assess_rir_readiness",
]

#: §9.5: below this many failure sets the model is unstable — withhold the estimate.
MIN_FAILURE_SETS: int = 15

#: §9.5 prototype target: at or above this, the estimate is shown without a caveat.
TARGET_FAILURE_SETS: int = 30


@dataclass
class RirReadiness:
    """Whether a corpus supports showing a RIR estimate, and what to say if not."""

    n_failure_sets: int
    n_sets: int
    by_exercise: dict[str, int] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        """True once the RIR estimate may be shown at all (§9.5)."""
        return self.n_failure_sets >= MIN_FAILURE_SETS

    @property
    def provisional(self) -> bool:
        """True when the estimate is shown but §9.5 calls it illustrative only."""
        return self.available and self.n_failure_sets < TARGET_FAILURE_SETS

    @property
    def sets_needed(self) -> int:
        """Failure sets still required before the estimate unlocks. 0 once available."""
        return max(MIN_FAILURE_SETS - self.n_failure_sets, 0)

    def explain(self) -> str:
        """One user-facing sentence about the estimate's availability.

        States what is missing and why, in measured terms — no promise about what the
        estimate will say once it unlocks (§8.7).
        """
        if not self.available:
            return (
                f"Reps-in-reserve needs {self.sets_needed} more set"
                f"{'s' if self.sets_needed != 1 else ''} taken to failure before it can be "
                f"estimated ({self.n_failure_sets} of {MIN_FAILURE_SETS} so far)."
            )
        if self.provisional:
            return (
                f"Reps-in-reserve is provisional: estimated from {self.n_failure_sets} "
                f"failure sets, against {TARGET_FAILURE_SETS} for a settled estimate."
            )
        return f"Reps-in-reserve estimated from {self.n_failure_sets} failure sets."


def assess_rir_readiness(labeled: list) -> RirReadiness:
    """Count the failure sets in a prepared corpus and apply the §9.5 thresholds.

    ``labeled`` is the ``[(SetFeatures, RirMeta)]`` list the hazard head fits on. Only
    ``reached_failure`` sets count toward the threshold: censored sets inform the hazard at
    the states they reached, but they never observe a failure, so they cannot identify the
    high-fatigue end of the curve on their own.
    """
    failures = [meta for _, meta in labeled if meta.reached_failure]
    return RirReadiness(
        n_failure_sets=len(failures),
        n_sets=len(labeled),
        by_exercise=dict(Counter(m.exercise for m in failures)),
    )
