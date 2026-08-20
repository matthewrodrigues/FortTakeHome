"""Deterministic form subscores (§8.3-8.4) — Layer 5a form half.

See docs/superpowers/specs/2026-08-19-phase4-form-subscores-design.md for the full
design. The one-line summary of the three load-bearing decisions:

* **Normalization (§8.4):** fixed reference scale by default (flagged provisional); an
  optional per-channel ``NormRef`` switches to true z-scoring and clears the flag.
* **ROM reference:** ``rom_baseline`` is the mean ROM of the current set's first few
  completed reps (tier-3 style) — self-contained here, carrying §7.1's known flaw.
* **Score = level, pointer = trend:** each subscore's 0-100 SCORE reads a magnitude of
  deviation from the reference; its POINTER reads the within-set trend off Phase-3
  ``retrospective_summaries``.

Every subscore expresses its deviation as a **unit-free ratio** (relative shortfall or
relative rise vs the reference reps). That keeps the fixed "notable" scales below
interpretable across channels with different units, and keeps the whole layer robust to the
arbitrary units of e.g. the DTW distance and the tremor power.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from wristset.features import SetFeatures
from wristset.features.baseline import EARLY_SET_REPS

__all__ = [
    "squash",
    "MIN_REPS_BEYOND_REFERENCE",
    "Pointer",
    "NormRef",
    "FormSubscore",
    "FormSubscores",
    "score_form",
    "SUBSCORE_NAMES",
]

#: §8.3 subscores, in table order. Equal weight within the form half.
SUBSCORE_NAMES: tuple[str, ...] = (
    "rom_completeness", "path_consistency", "tempo_control", "stability",
)

# --- fixed reference scales (§8.4 cold-start path) ------------------------------
# Each is the unit-free deviation that HALVES the subscore (see :func:`squash`), so each is
# directly interpretable as "how much change this user would call notable". Parameterized
# from day one even though the prototype hardcodes them (cf. TARGET_RIR in §8.2).
ROM_LOSS_NOTABLE: float = 0.15        # 15% mean-depth shortfall vs opening reps
PATH_DIVERGENCE_NOTABLE: float = 0.60  # 60% rise in path-to-baseline DTW vs opening reps
TEMPO_CHANGE_NOTABLE: float = 0.30    # 30% eccentric shortening across the set
TREMOR_RISE_NOTABLE: float = 1.50     # 150% rise in 8-12 Hz tremor power

#: z-scale for the §8.4 personalized path: |z| that halves the subscore.
Z_NOTABLE: float = 1.5

#: Steepness constant so ``squash(scale, scale) == 50`` — ``scale`` is, by definition, the
#: deviation that halves the score.
_K = math.log(3.0)

_EPS = 1e-12


def squash(deviation: float, *, scale: float) -> float:
    """Monotone map from a non-negative deviation to a 0-100 subscore (§8.4).

    ``100`` at zero deviation, strictly decreasing, ``50`` at one ``scale`` of deviation,
    asymptotic to ``0``. Deviations below zero (a set BETTER than its reference) clamp to
    ``100`` — a subscore is never above full marks. Shared by both normalization paths:
    fixed-scale passes natural-unit deviation and a "notable" scale; the z-score path
    passes ``z`` and a z-scale.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    d = max(deviation, 0.0)
    return 200.0 * (1.0 - 1.0 / (1.0 + math.exp(-_K * d / scale)))


@dataclass
class Pointer:
    """One textual decomposition of a subscore (§8.6). Structured, not prose — Phase-8
    ``insights/`` orders these largest-magnitude-first and renders the narrative.

    ``text`` states measured change only (§8.7); ``magnitude`` (a non-negative size, used
    for ordering) and ``onset_rep`` (the rep a change began, from a §6.3 breakpoint) let the
    narrative layer rank and phrase without re-deriving anything.
    """

    subscore: str
    text: str
    magnitude: float
    feature: str
    onset_rep: int | None = None


@dataclass
class NormRef:
    """Optional §8.4 personalized normalization: per-subscore ``mean``/``std`` of the
    deviation statistic over the user's history. When supplied, scoring uses z-scores and
    drops the provisional flag; when absent, scoring uses the fixed scales above."""

    mean: dict[str, float]
    std: dict[str, float]


@dataclass
class FormSubscore:
    """One §8.3 subscore: a 0-100 level (or ``None`` when there is no reference), its raw
    drivers for debuggability, and >=1 pointer."""

    name: str
    score: float | None
    provisional: bool
    drivers: dict = field(default_factory=dict)
    pointers: list[Pointer] = field(default_factory=list)


@dataclass
class FormSubscores:
    """The four §8.3 subscores plus their equal-weight composite (§8.1 form half)."""

    subscores: list[FormSubscore] = field(default_factory=list)

    def by_name(self, name: str) -> FormSubscore:
        for s in self.subscores:
            if s.name == name:
                return s
        raise KeyError(name)

    @property
    def form_score(self) -> float | None:
        """Equal-weight mean of the subscores that HAVE a score. ``None``-scored subscores
        (no reference) are excluded, never counted as zero (§7.3 / Phase-3 invariant)."""
        present = [s.score for s in self.subscores if s.score is not None]
        return float(sum(present) / len(present)) if present else None

    @property
    def provisional(self) -> bool:
        return any(s.provisional for s in self.subscores if s.score is not None)


# --------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------


def score_form(
    sf: SetFeatures,
    *,
    norm_ref: NormRef | None = None,
    n_ref_reps: int = EARLY_SET_REPS,
) -> FormSubscores:
    """Compute the four deterministic §8.3 form subscores for one set.

    ``norm_ref`` supplied -> personalized z-score normalization, non-provisional (§8.4).
    Absent -> fixed-scale normalization flagged provisional. ``n_ref_reps`` sets how many
    opening completed reps define the in-set reference level for every channel.

    A set with too few completed reps to sustain an in-set reference yields ``None`` for
    every subscore (and therefore a ``None`` composite) rather than a score — see
    :func:`_has_usable_reference`.
    """
    completed = [r for r in sf.reps if r.completed]
    ref = completed[:n_ref_reps]
    retro = sf.retrospective

    if not _has_usable_reference(completed, n_ref_reps):
        return FormSubscores(
            subscores=[FormSubscore(n, None, norm_ref is None) for n in SUBSCORE_NAMES]
        )

    subs = [
        _rom_completeness(completed, ref, retro, norm_ref, n_ref_reps),
        _path_consistency(completed, ref, retro, norm_ref),
        _tempo_control(completed, ref, retro, norm_ref),
        _stability(completed, ref, retro, norm_ref),
    ]
    return FormSubscores(subscores=subs)


#: Completed reps required beyond the reference block before any subscore is meaningful.
#: Every subscore compares the set against its own opening ``n_ref_reps``, so there must be
#: reps OUTSIDE that block to compare against; with a remainder of one, every deviation is
#: a single-observation estimate. Two is the smallest remainder that is a sample rather
#: than a point. Derived from ``n_ref_reps`` rather than hardcoded so it tracks it.
MIN_REPS_BEYOND_REFERENCE: int = 2


def _has_usable_reference(completed: list, n_ref_reps: int) -> bool:
    """Whether the set can sustain an in-set reference (§7.1 tier-3 style).

    Without this, a 3-4 rep set is compared almost entirely against ITSELF: the deviation
    collapses to ~0 and every subscore squashes to a confident 100.0 — which reads as
    "perfect form" when it means "no information". Returning ``None`` instead keeps the
    Phase-3 invariant that absence of a reference is never reported as a good score.
    """
    return len(completed) >= n_ref_reps + MIN_REPS_BEYOND_REFERENCE


def _mean(reps: list, channel: str) -> float | None:
    """Mean of a channel over reps, skipping ``None`` (e.g. an unresolved
    ``path_dtw_baseline``). ``None`` when nothing is available."""
    vals = [getattr(r, channel) for r in reps]
    vals = [v for v in vals if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _score_from_deviation(
    name: str, deviation: float, fixed_scale: float, norm_ref: NormRef | None
) -> float:
    """Fixed-scale squash by default; z-score squash when a ``NormRef`` is supplied (§8.4)."""
    if norm_ref is not None and norm_ref.std.get(name, 0.0) > _EPS:
        z = (deviation - norm_ref.mean.get(name, 0.0)) / norm_ref.std[name]
        return squash(z, scale=Z_NOTABLE)
    return squash(deviation, scale=fixed_scale)


def _onset(retro: dict, channel: str) -> int | None:
    bp = retro.get(f"retro_{channel}_breakpoint")
    return int(bp) if bp is not None else None


def _onset_clause(rep: int | None, word: str = "starting") -> str:
    return f" {word} rep {rep}" if rep is not None else ""


def _rom_completeness(completed, ref, retro, norm_ref, n_ref_reps) -> FormSubscore:
    """Level = mean ROM vs the opening reps' ROM (§8.3 completeness). Pointer = the
    first->last retention change (§8.3 retention)."""
    name = "rom_completeness"
    rom_baseline = _mean(ref, "rom_vertical")
    mean_rom = _mean(completed, "rom_vertical")
    if not rom_baseline or mean_rom is None:
        return FormSubscore(name, None, norm_ref is None)

    completeness = mean_rom / rom_baseline
    deviation = 1.0 - completeness  # shortfall; squash clamps a surplus to score 100
    score = _score_from_deviation(name, deviation, ROM_LOSS_NOTABLE, norm_ref)

    n_short = sum(
        1 for r in completed if r.rom_vertical < 0.9 * rom_baseline
    )
    retention = retro.get("retro_rom_vertical_total_change")
    onset = _onset(retro, "rom_vertical")
    if retention is not None:
        text = f"Depth changed {retention * 100:+.0f}% from first to last rep."
        mag = abs(retention)
    else:
        text = f"Average depth was {completeness * 100:.0f}% of the opening reps."
        mag = abs(deviation)
    pointers = [Pointer(name, text, mag, "rom_vertical", onset)]
    if n_short:
        pointers.append(Pointer(
            name, f"{n_short} of {len(completed)} reps fell short of the opening depth.",
            n_short / max(len(completed), 1), "rom_vertical",
        ))

    return FormSubscore(
        name, score, norm_ref is None,
        drivers=dict(completeness=completeness, rom_baseline=rom_baseline,
                     mean_rom=mean_rom, retention=retention, n_short=n_short),
        pointers=pointers,
    )


def _path_consistency(completed, ref, retro, norm_ref) -> FormSubscore:
    """Level = relative rise in path-to-baseline DTW vs the opening reps. ``path_dtw_baseline``
    is already §7-baseline-relative, so no new reference is needed; ``None`` throughout
    (cold start, no template) -> unscored, not zero."""
    name = "path_consistency"
    dtw_ref = _mean(ref, "path_dtw_baseline")
    dtw_all = _mean(completed, "path_dtw_baseline")
    if dtw_ref is None or dtw_all is None or dtw_ref <= _EPS:
        return FormSubscore(name, None, norm_ref is None)

    deviation = dtw_all / dtw_ref - 1.0
    score = _score_from_deviation(name, deviation, PATH_DIVERGENCE_NOTABLE, norm_ref)

    slope = retro.get("retro_path_dtw_baseline_slope")
    onset = _onset(retro, "path_dtw_baseline")
    text = f"Bar path diverged from baseline{_onset_clause(onset)}."
    mag = abs(slope) if slope is not None else max(deviation, 0.0)
    return FormSubscore(
        name, score, norm_ref is None,
        drivers=dict(dtw_ref=dtw_ref, dtw_mean=dtw_all, deviation=deviation, slope=slope),
        pointers=[Pointer(name, text, mag, "path_dtw_baseline", onset)],
    )


def _tempo_control(completed, ref, retro, norm_ref) -> FormSubscore:
    """Level = eccentric shortening vs the opening reps (§8.3 ecc_duration). Pointer = the
    across-set eccentric change."""
    name = "tempo_control"
    ecc_ref = _mean(ref, "ecc_duration")
    ecc_all = _mean(completed, "ecc_duration")
    if not ecc_ref or ecc_all is None:
        return FormSubscore(name, None, norm_ref is None)

    deviation = 1.0 - ecc_all / ecc_ref  # shortening is the degradation
    score = _score_from_deviation(name, deviation, TEMPO_CHANGE_NOTABLE, norm_ref)

    change = retro.get("retro_ecc_duration_total_change")
    onset = _onset(retro, "ecc_duration")
    if change is not None:
        text = f"Eccentric duration changed {change * 100:+.0f}% across the set."
        mag = abs(change)
    else:
        text = f"Eccentric duration changed {-deviation * 100:+.0f}% from the opening reps."
        mag = abs(deviation)
    return FormSubscore(
        name, score, norm_ref is None,
        drivers=dict(ecc_ref=ecc_ref, ecc_mean=ecc_all, deviation=deviation, change=change),
        pointers=[Pointer(name, text, mag, "ecc_duration", onset)],
    )


def _stability(completed, ref, retro, norm_ref) -> FormSubscore:
    """Level = relative rise in 8-12 Hz tremor power vs the opening reps (§8.3). Pointer =
    where tremor took off (§6.3 breakpoint)."""
    name = "stability"
    tremor_ref = _mean(ref, "tremor_power_8_12")
    tremor_all = _mean(completed, "tremor_power_8_12")
    if tremor_ref is None or tremor_all is None or tremor_ref <= _EPS:
        return FormSubscore(name, None, norm_ref is None)

    deviation = tremor_all / tremor_ref - 1.0
    score = _score_from_deviation(name, deviation, TREMOR_RISE_NOTABLE, norm_ref)

    change = retro.get("retro_tremor_power_8_12_total_change")
    onset = _onset(retro, "tremor_power_8_12")
    if change is not None:
        text = f"Tremor rose {change * 100:.0f}%{_onset_clause(onset, 'after')}."
        mag = abs(change)
    else:
        text = f"Tremor rose {deviation * 100:.0f}% from the opening reps."
        mag = max(deviation, 0.0)
    return FormSubscore(
        name, score, norm_ref is None,
        drivers=dict(tremor_ref=tremor_ref, tremor_mean=tremor_all,
                     deviation=deviation, change=change),
        pointers=[Pointer(name, text, mag, "tremor_power_8_12", onset)],
    )
