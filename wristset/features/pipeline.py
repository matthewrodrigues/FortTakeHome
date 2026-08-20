"""Feature-layer orchestrator: ConditionedSet + reps -> SetFeatures (§6.2-6.3, §7).

Stage order is load-bearing: the §7 baseline must resolve before ``path_dtw_baseline``
can be computed, and that channel is one of the five §6.3 summarises. Running the
summaries first would silently produce an all-None channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wristset.conditioning import ConditionedSet
from wristset.features.baseline import (
    Baseline,
    RepSource,
    attach_path_dtw_baseline,
    resolve_baseline,
)
from wristset.features.rep_features import RepFeatures, extract_rep_features
from wristset.features.trajectory import (
    KEY_CHANNELS,
    causal_summaries,
    retrospective_summaries,
)
from wristset.segmentation import RepBoundary

__all__ = ["SetFeatures", "extract_set_features"]


@dataclass
class SetFeatures:
    """Everything Layer 4 produces for one set."""

    reps: list[RepFeatures] = field(default_factory=list)
    baseline: Baseline | None = None
    #: whole-set summaries — set score (§8) and RPE head (§10.4) only
    retrospective: dict = field(default_factory=dict)
    #: per-rep causal summaries, keyed by 1-based rep index — RIR head (§9.2) only
    causal: dict[int, dict] = field(default_factory=dict)

    @property
    def n_reps(self) -> int:
        return len(self.reps)

    @property
    def n_completed(self) -> int:
        return sum(1 for r in self.reps if r.completed)

    def channel(self, name: str) -> list:
        """Per-rep series for one feature channel, in rep order."""
        return [getattr(r, name) for r in self.reps]


def extract_set_features(
    cs: ConditionedSet,
    reps: list[RepBoundary],
    *,
    exercise: str = "bench_press",
    load_kg: float = 0.0,
    set_type: str = "working",
    load_pct_e1rm: float | None = None,
    history: list[RepSource] | None = None,
    warmups: list[RepSource] | None = None,
    causal: bool = True,
) -> SetFeatures:
    """Run the full Layer 4 pipeline for one segmented set.

    ``history`` and ``warmups`` feed the §7.1 baseline hierarchy; omitting both falls
    through to the tier-3 early-set reference. ``causal=False`` skips the per-rep causal
    summaries, which are only needed to build the §9.3 person-period design matrix and
    cost O(n_reps) extra summary fits.
    """
    n = len(reps)
    features = [
        extract_rep_features(
            cs, r,
            n_reps_in_set=n,
            load_kg=load_kg,
            exercise=exercise,
            set_type=set_type,
            load_pct_e1rm=load_pct_e1rm,
        )
        for r in reps
    ]

    # §7 baseline, then the baseline-dependent §6.2 feature it unlocks
    baseline = resolve_baseline(
        current_cs=cs,
        current_reps=reps,
        exercise=exercise,
        load_kg=load_kg,
        history=history,
        warmups=warmups,
    )
    attach_path_dtw_baseline(cs, reps, features, baseline)

    retro = retrospective_summaries(features, KEY_CHANNELS)
    causal_by_rep: dict[int, dict] = {}
    if causal:
        for r in features:
            causal_by_rep[r.rep_index] = causal_summaries(features, r.rep_index, KEY_CHANNELS)

    return SetFeatures(
        reps=features,
        baseline=baseline,
        retrospective=retro,
        causal=causal_by_rep,
    )
