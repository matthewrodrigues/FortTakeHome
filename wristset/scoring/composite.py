"""Composite set score (§8.1-8.2, §8.5) — Layer 5a effort half.

Three components, weighted effort 1/2 : form 1/2 (§8.1)::

    effort = 0.5 * (rir_score + rpe_score)
    total  = 0.5 * effort + 0.5 * form_score
           = 0.25 * rir_score + 0.25 * rpe_score + 0.5 * form_score

The 50/50 effort/form split is deliberate: ``RPE ~ 10 - RIR``, so treating RIR, RPE and
form as three equal categories would hand the effort construct 2/3 of the total (§8.1).

Both effort components are **proximity to a target**, not "more is better" (§8.2) —
overshooting is penalised, because a set pushed past failure with assistance is not better
than one that hit its target. Targets and tolerances are module-level parameters from day
one even though the prototype hardcodes them; per-user / per-block targets must be a config
change, not a refactor.

**§8.2's confidence-gated RIR is deliberately NOT implemented — see :data:`RIR_GATE_NOTE`.**

Missing components (no fitted head, cold start, an unscored form set) are **dropped and the
remaining weights renormalised**, never replaced with a population default. That is the
same invariant Phase 3/4 use throughout: absence of a measurement is reported as absence,
not as an average (§8.4, §8.7). With form alone the composite equals ``form_score``, which
is exactly the Phase-5 demo's behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "TARGET_RIR", "TARGET_RPE", "RIR_TOLERANCE", "RPE_TOLERANCE",
    "DIVERGENCE_THRESHOLD", "RIR_GATE_NOTE",
    "proximity_score", "CompositeScore", "score_composite",
]

#: §8.2 prototype targets. Hypertrophy blocks target RIR 1-3, technique work RIR 4-5,
#: peaking RIR 0 — parameterised now so those become configuration, not a rewrite.
TARGET_RIR: float = 0.0
TARGET_RPE: float = 10.0

#: Distance from target at which a component scores 0 (§8.2).
RIR_TOLERANCE: float = 3.0
RPE_TOLERANCE: float = 3.0

#: §8.5: above this gap between effort and form, the composite is collapsing two genuinely
#: different sets onto one number, so the UI must lead with text instead.
DIVERGENCE_THRESHOLD: float = 30.0

#: Why §8.2's confidence-gated RIR is absent.
#:
#: §8.2 specifies dropping the RIR term when ``rir_predictive_std`` is too wide. That gate
#: is **deferred**: the RIR head has no calibrated uncertainty estimate to drive it.
#: Measured on 527 held-out predictions across three corpora, the predictive std is
#: overconfident ~3x (median 0.5 against median error 1.6), covers only 20-30% of errors
#: within one std (a calibrated distribution gives ~68%), and is *anti*-correlated with
#: error (r = -0.08) — it calls its worst predictions its most certain. Every alternative
#: signal tested (entropy, tail mass, max probability, projected-hazard saturation,
#: forecast horizon) correlated |r| <= 0.15 with error.
#:
#: Gating on any of those would gate near-arbitrarily while implying a reliability
#: guarantee the model cannot make, so the RIR term is kept unconditionally. Re-enable once
#: §9.4's forward projection is calibrated; ``RIRPrediction.is_confident`` remains as a
#: weak advisory annotation only.
RIR_GATE_NOTE: str = (
    "confidence-gated RIR (§8.2) deferred: no calibrated uncertainty estimate "
    "(predictive std is anti-correlated with error, r=-0.08)"
)

#: Component weights when every component is present (§8.1).
_WEIGHTS: dict[str, float] = {"rir": 0.25, "rpe": 0.25, "form": 0.5}


def proximity_score(value: float, target: float, tolerance: float) -> float:
    """§8.2 proximity scoring: 100 at the target, 0 at ``tolerance`` away, clamped.

    Symmetric by design — overshooting the target is penalised exactly as much as falling
    short of it, because "more effort" is not monotonically better.
    """
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    return float(min(max(100.0 * (1.0 - abs(value - target) / tolerance), 0.0), 100.0))


@dataclass
class CompositeScore:
    """A §8.1 composite plus the parts that produced it.

    ``components`` holds the 0-100 score of each component that was available; missing
    components are simply absent, and the weights are renormalised over what remains.
    """

    total: float | None
    effort: float | None
    form: float | None
    components: dict[str, float] = field(default_factory=dict)
    provisional: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def components_used(self) -> list[str]:
        return sorted(self.components)

    @property
    def divergent(self) -> bool:
        """§8.5: effort and form disagree enough that the single number misleads.

        A failure set with collapsing form (effort 100, form 40) and a conservative set with
        clean form (effort 40, form 100) both average 70. When this is True the presentation
        layer must lead with the narrative and de-emphasise the number.
        """
        if self.effort is None or self.form is None:
            return False
        return abs(self.effort - self.form) > DIVERGENCE_THRESHOLD


def score_composite(
    *,
    form_score: float | None,
    expected_rir: float | None = None,
    reported_rpe: float | None = None,
    form_provisional: bool = False,
    target_rir: float = TARGET_RIR,
    target_rpe: float = TARGET_RPE,
    rir_tolerance: float = RIR_TOLERANCE,
    rpe_tolerance: float = RPE_TOLERANCE,
) -> CompositeScore:
    """Combine the effort and form halves into one 0-100 set score (§8.1-8.2).

    Any of the three inputs may be ``None`` — a set with no fitted RIR head, a set whose
    RPE was not reported, or a set too short to score form (§8.3 short-set guard). Present
    components keep their §8.1 weights *relative to each other*; the total renormalises over
    them. With form alone the composite is the form score.

    Returns ``total=None`` only when nothing at all was available.
    """
    components: dict[str, float] = {}
    notes: list[str] = []

    if expected_rir is not None:
        components["rir"] = proximity_score(expected_rir, target_rir, rir_tolerance)
    else:
        notes.append("RIR unavailable (no fitted hazard head for this set)")

    if reported_rpe is not None:
        components["rpe"] = proximity_score(reported_rpe, target_rpe, rpe_tolerance)
    else:
        notes.append("RPE unavailable (not reported)")

    if form_score is not None:
        components["form"] = float(form_score)
    else:
        notes.append("form unavailable (no reference / too few reps)")

    if not components:
        return CompositeScore(None, None, None, {}, form_provisional,
                              notes + [RIR_GATE_NOTE])

    weight = sum(_WEIGHTS[k] for k in components)
    total = sum(_WEIGHTS[k] * v for k, v in components.items()) / weight

    # the effort half is itself a renormalised mean of whichever effort parts exist
    effort_parts = [components[k] for k in ("rir", "rpe") if k in components]
    effort = float(sum(effort_parts) / len(effort_parts)) if effort_parts else None

    if len(components) < len(_WEIGHTS):
        notes.append(f"composite renormalised over {sorted(components)}")
    notes.append(RIR_GATE_NOTE)

    return CompositeScore(
        total=float(total),
        effort=effort,
        form=components.get("form"),
        components=components,
        provisional=form_provisional,
        notes=notes,
    )
