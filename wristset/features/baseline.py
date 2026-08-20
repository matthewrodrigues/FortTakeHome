"""Baseline management (§7) — Layer 7.

Every comparison-based feature needs a reference for "this user's clean form on this
exercise at this load". §7.1 resolves that reference through a three-tier hierarchy,
falling through when a tier is unavailable:

    1. cross-session personal template — pooled low-fatigue reps from history, load
       conditioned. Best, requires history.
    2. warmup-set reference — today's ``set_type="warmup"`` reps. Low load, low fatigue.
    3. early-set reference — reps 1-3 of the current set. Fallback only.

**§7.1's known flaw is preserved deliberately**: tier 3 reads a set that starts badly as
clean, because it has nothing else to compare against. The resolved tier is therefore
returned alongside the template so downstream layers can down-weight tier-3 comparisons
and the UI can say which reference was used (§8.4 provisional-label path).

Load conditioning (§7.2) restricts pooled reps to a +/-``LOAD_BAND`` window, since form at
90% 1RM legitimately differs from form at 60% and comparing across loads manufactures
false positives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wristset.conditioning import ConditionedSet
from wristset.segmentation import RepBoundary
from wristset.segmentation.dtw import PROFILE_LEN, dtw_distance, rep_velocity_profile

__all__ = [
    "Baseline",
    "BaselineSource",
    "resolve_baseline",
    "attach_path_dtw_baseline",
    "LOAD_BAND",
    "EARLY_SET_REPS",
]

#: §7.2 load conditioning: pooled baseline reps must sit within this fractional band of
#: the target set's load, or the comparison mixes legitimately different form.
LOAD_BAND: float = 0.10

#: §7.1 tier 3 uses reps 1-3 of the current set.
EARLY_SET_REPS: int = 3

#: Minimum reps required to form a usable template.
_MIN_TEMPLATE_REPS: int = 2


class BaselineSource:
    """Which §7.1 tier supplied the template. Ordered best to worst."""

    CROSS_SESSION = "cross_session"
    WARMUP = "warmup"
    EARLY_SET = "early_set"
    NONE = "none"


@dataclass
class Baseline:
    """A resolved form reference plus provenance (§7.1)."""

    template: np.ndarray | None  # (PROFILE_LEN,) mean velocity profile
    source: str
    n_reps: int  # how many reps were pooled into the template
    load_kg: float | None = None

    @property
    def is_reliable(self) -> bool:
        """True for tiers 1-2. Tier 3 carries §7.1's known flaw (a set that starts bad
        reads as clean), so callers should treat those comparisons as provisional."""
        return self.source in (BaselineSource.CROSS_SESSION, BaselineSource.WARMUP)


@dataclass
class RepSource:
    """One conditioned set plus its reps, as a candidate source of baseline reps."""

    cs: ConditionedSet
    reps: list[RepBoundary]
    load_kg: float
    set_type: str = "working"
    exercise: str = "bench_press"


def _pool_profiles(sources: list[RepSource], target_load: float | None) -> list[np.ndarray]:
    """Collect velocity profiles from completed reps, load-conditioned per §7.2."""
    profiles: list[np.ndarray] = []
    for src in sources:
        if target_load is not None and target_load > 0:
            if abs(src.load_kg - target_load) / target_load > LOAD_BAND:
                continue  # outside the §7.2 load band
        for r in src.reps:
            if r.completed:
                profiles.append(rep_velocity_profile(src.cs, r))
    return profiles


def resolve_baseline(
    *,
    current_cs: ConditionedSet,
    current_reps: list[RepBoundary],
    exercise: str,
    load_kg: float,
    history: list[RepSource] | None = None,
    warmups: list[RepSource] | None = None,
) -> Baseline:
    """Resolve the §7.1 baseline hierarchy, falling through unavailable tiers.

    ``history`` is prior-session sets for this user and exercise; ``warmups`` is today's
    warmup sets. Both are filtered to the matching exercise and, for tiers 1-2, to the
    §7.2 load band. Cold start (§7.3) surfaces as ``BaselineSource.NONE``, which callers
    should handle by suppressing baseline-dependent features rather than substituting a
    population default here.
    """
    # tier 1 — cross-session personal template
    hist = [s for s in (history or []) if s.exercise == exercise]
    profiles = _pool_profiles(hist, load_kg)
    if len(profiles) >= _MIN_TEMPLATE_REPS:
        return Baseline(np.mean(profiles, axis=0), BaselineSource.CROSS_SESSION,
                        len(profiles), load_kg)

    # tier 2 — today's warmup sets. NOT load-conditioned: a warmup is by definition at a
    # lower load, so applying the §7.2 band here would reject every warmup. This is the
    # documented trade-off of tier 2 — cleaner form, different load.
    warm = [s for s in (warmups or []) if s.exercise == exercise]
    profiles = _pool_profiles(warm, target_load=None)
    if len(profiles) >= _MIN_TEMPLATE_REPS:
        return Baseline(np.mean(profiles, axis=0), BaselineSource.WARMUP,
                        len(profiles), None)

    # tier 3 — early reps of the current set (§7.1 fallback, known flaw)
    early = [r for r in current_reps if r.completed][:EARLY_SET_REPS]
    if len(early) >= _MIN_TEMPLATE_REPS:
        profiles = [rep_velocity_profile(current_cs, r) for r in early]
        return Baseline(np.mean(profiles, axis=0), BaselineSource.EARLY_SET,
                        len(profiles), load_kg)

    return Baseline(None, BaselineSource.NONE, 0, load_kg)


def attach_path_dtw_baseline(
    cs: ConditionedSet,
    reps: list[RepBoundary],
    features: list,
    baseline: Baseline,
) -> None:
    """Fill ``path_dtw_baseline`` on each RepFeatures in place (§6.2 path table).

    DTW distance of each rep's velocity profile to the resolved template — §6.2's "direct
    form-consistency measure". Left as ``None`` under cold start (§7.3) so downstream code
    can distinguish "no deviation" from "no reference", which a zero would conflate.
    """
    if baseline.template is None:
        return
    for rep, feat in zip(reps, features):
        prof = rep_velocity_profile(cs, rep, length=PROFILE_LEN)
        feat.path_dtw_baseline = float(dtw_distance(prof, baseline.template))
