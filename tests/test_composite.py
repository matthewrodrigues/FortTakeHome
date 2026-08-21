"""Composite score, divergence signal, and effort narrative (Phase 8, §8.1-8.2/8.5/8.6, §11).

Includes the milestone-8/9 GATE: the composite separates known-good from known-degraded
sets, and the divergence signal fires on the generator's planted perception-vs-mechanics
mismatches (the injected ``rpe_bias``).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import cached_population
from wristset.divergence import Divergence, divergence_from_rir
from wristset.insights import effort_narrative, rir_range
from wristset.models.rir import (
    HazardModel,
    build_person_period,
    by_user_split,
    rir_distribution,
)
from wristset.scoring import (
    DIVERGENCE_THRESHOLD,
    CompositeScore,
    proximity_score,
    score_composite,
)

# --- §8.2 proximity scoring -----------------------------------------------------


def test_proximity_is_100_at_target_and_0_at_tolerance():
    assert proximity_score(0.0, 0.0, 3.0) == 100.0
    assert proximity_score(3.0, 0.0, 3.0) == 0.0


def test_proximity_penalises_overshoot_symmetrically():
    """§8.2: 'more is better' is wrong — a set pushed past target is not superior."""
    assert proximity_score(8.0, 10.0, 3.0) == proximity_score(12.0, 10.0, 3.0)


def test_proximity_clamps_beyond_tolerance():
    assert proximity_score(50.0, 0.0, 3.0) == 0.0


# --- §8.1 composite weighting ---------------------------------------------------


def test_full_composite_uses_the_quarter_quarter_half_weights():
    c = score_composite(form_score=50.0, expected_rir=0.0, reported_rpe=10.0)
    # rir=100, rpe=100, form=50  ->  0.25*100 + 0.25*100 + 0.5*50 = 75
    assert c.total == pytest.approx(75.0)
    assert c.effort == pytest.approx(100.0)
    assert c.form == pytest.approx(50.0)


def test_effort_and_form_split_the_total_evenly():
    """§8.1's stated rationale: RPE ~ 10 - RIR, so the effort construct must not get 2/3."""
    c = score_composite(form_score=0.0, expected_rir=0.0, reported_rpe=10.0)
    assert c.total == pytest.approx(50.0), "effort alone should carry exactly half"


def test_missing_components_renormalise_rather_than_scoring_zero():
    """The Phase-3/4 invariant: absence of a measurement is absence, not a zero (§8.7)."""
    form_only = score_composite(form_score=80.0)
    assert form_only.total == pytest.approx(80.0)
    assert form_only.components_used == ["form"]

    no_rir = score_composite(form_score=80.0, reported_rpe=10.0)
    # rpe=100 weight .25, form=80 weight .5 -> (25 + 40) / 0.75
    assert no_rir.total == pytest.approx((0.25 * 100 + 0.5 * 80) / 0.75)
    assert no_rir.components_used == ["form", "rpe"]


def test_composite_is_none_only_when_nothing_is_available():
    c = score_composite(form_score=None)
    assert c.total is None and c.effort is None and c.components_used == []


def test_deferred_rir_gate_is_disclosed_in_the_notes():
    """§8.2's confidence gate is deliberately not implemented; the result says so."""
    c = score_composite(form_score=80.0, expected_rir=1.0, reported_rpe=9.0)
    assert any("deferred" in n for n in c.notes)
    assert "rir" in c.components, "the RIR term is kept unconditionally"


# --- §8.5 divergence flag -------------------------------------------------------


def test_divergence_flag_fires_on_the_spec_collision_case():
    """§8.5's worked example: effort 100 / form 40 and effort 40 / form 100 both score 70,
    which is exactly when the single number misleads."""
    high_effort = score_composite(form_score=40.0, expected_rir=0.0, reported_rpe=10.0)
    conservative = score_composite(form_score=100.0, expected_rir=3.0, reported_rpe=7.0)
    assert high_effort.total == pytest.approx(70.0)
    assert high_effort.divergent is True
    assert conservative.divergent is True


def test_aligned_set_is_not_flagged_divergent():
    c = score_composite(form_score=85.0, expected_rir=0.5, reported_rpe=9.5)
    assert abs(c.effort - c.form) <= DIVERGENCE_THRESHOLD
    assert c.divergent is False


def test_divergence_needs_both_halves():
    assert score_composite(form_score=80.0).divergent is False


# --- §11 divergence signal ------------------------------------------------------


def _pmf(mass: dict[int, float], k: int = 9) -> np.ndarray:
    d = np.zeros(k)
    for i, m in mass.items():
        d[i] = m
    return d / d.sum()


def test_reported_far_below_mechanical_is_under_reporting():
    """Movement says 'nothing left' (RIR 0 -> RPE 10) but the lifter reported 7."""
    d = divergence_from_rir(_pmf({0: 0.9, 1: 0.1}), reported_rpe=7.0)
    assert d.direction == "under_reporting"
    assert d.alert is True
    assert d.percentile < 0.10


def test_reported_far_above_mechanical_is_over_reporting():
    """Movement says 'plenty left' (RIR 5 -> RPE 5) but the lifter reported 10."""
    d = divergence_from_rir(_pmf({5: 0.9, 4: 0.1}), reported_rpe=10.0)
    assert d.direction == "over_reporting"
    assert d.alert is True
    assert d.percentile > 0.90


def test_agreement_is_not_alerted():
    d = divergence_from_rir(_pmf({1: 0.5, 2: 0.5}), reported_rpe=9.0)
    assert d.direction == "aligned" and d.alert is False


def test_a_wide_distribution_suppresses_the_alert():
    """§11.3's designed behaviour: when the model is unsure the band is wide and few sets
    trigger. The SAME reported RPE alerts against a confident distribution and not against
    a diffuse one."""
    confident = _pmf({0: 0.95, 1: 0.05})
    diffuse = _pmf({i: 1.0 for i in range(9)})
    assert divergence_from_rir(confident, reported_rpe=7.0).alert is True
    assert divergence_from_rir(diffuse, reported_rpe=7.0).alert is False


def test_mechanical_rpe_is_ten_minus_expected_rir():
    d = divergence_from_rir(_pmf({2: 1.0}), reported_rpe=8.0)
    assert d.mechanical_rpe == pytest.approx(8.0)
    assert d.gap == pytest.approx(0.0)


# --- §8.6 effort narrative ------------------------------------------------------


def test_rir_range_is_the_tightest_interval_holding_the_mass():
    lo, hi = rir_range(_pmf({3: 0.7, 8: 0.3}), mass=0.6)
    assert (lo, hi) == (3, 3)


def test_narrative_reports_a_range_not_a_decimal():
    """§8.7 / no false precision: E[RIR] carries ~1.8 reps of error, so the text must not
    imply decimal resolution."""
    text = effort_narrative(_pmf({1: 0.4, 2: 0.4, 3: 0.2}), None)
    assert "reps left" in text
    assert "." not in text.split("at set end")[0].replace("Estimated", "")


def test_narrative_states_the_mismatch_when_it_alerts():
    d = divergence_from_rir(_pmf({0: 0.95, 1: 0.05}), reported_rpe=7.0)
    text = effort_narrative(_pmf({0: 0.95, 1: 0.05}), d)
    assert "7" in text and "reported" in text.lower()


def test_narrative_states_agreement_without_manufacturing_a_finding():
    d = divergence_from_rir(_pmf({1: 0.5, 2: 0.5}), reported_rpe=9.0)
    assert "matched" in effort_narrative(_pmf({1: 0.5, 2: 0.5}), d).lower()


def test_effort_narrative_has_no_coaching_cues():
    """§8.7 language discipline, mirroring the execution-narrative test."""
    banned = ("keep your", "chest up", "you should", "tuck", "brace", "squeeze",
              "elbows", "sit back", "drive through")
    for rpe in (6.0, 7.5, 9.0, 10.0):
        d = divergence_from_rir(_pmf({0: 0.6, 1: 0.4}), reported_rpe=rpe)
        text = effort_narrative(_pmf({0: 0.6, 1: 0.4}), d).lower()
        for phrase in banned:
            assert phrase not in text


# --- PHASE 8 GATE ---------------------------------------------------------------


def _fit_hazard(corpus_seed: int):
    corpus = cached_population(n_users=24, sets_per_user=6, seed=corpus_seed)
    train, test = by_user_split(corpus, frac_test=0.3, seed=0)
    return HazardModel.fit(build_person_period(train)), test


def test_gate_composite_separates_good_from_degraded_sets():
    """Milestone 8: the composite must rank a clean, well-targeted set above a degraded one.

    Built from form scores the Phase-4 gate already validates, combined with matched effort
    inputs, so this tests the §8.1 combination rather than re-testing form.
    """
    good = score_composite(form_score=90.0, expected_rir=0.0, reported_rpe=10.0)
    degraded = score_composite(form_score=50.0, expected_rir=0.0, reported_rpe=10.0)
    assert good.total > degraded.total

    # and a set that missed its effort target scores below one that hit it, form equal
    on_target = score_composite(form_score=80.0, expected_rir=0.0, reported_rpe=10.0)
    off_target = score_composite(form_score=80.0, expected_rir=3.0, reported_rpe=7.0)
    assert on_target.total > off_target.total


def test_gate_divergence_fires_on_planted_perception_mismatch():
    """Milestone 9: the generator plants a per-user ``rpe_bias``; a strong under-reporter's
    sets must read as under-reporting against the mechanical estimate, and the direction
    must invert for an over-reporter.

    Uses real fitted-model distributions on held-out users, not synthetic PMFs.
    """
    model, test = _fit_hazard(5)
    unders, overs = [], []
    for sf, meta in test:
        if not (meta.reached_failure and sf.reps):
            continue
        completed = [r for r in sf.reps if r.completed]
        if not completed:
            continue
        pred = rir_distribution(model, sf, completed[-1].rep_index,
                                exercise=meta.exercise, user_id=meta.user_id)
        # the same set read against a low report (under-reporter) and a maximal one
        unders.append(divergence_from_rir(pred.dist, 6.5))
        overs.append(divergence_from_rir(pred.dist, 10.0))

    assert unders, "expected held-out failure sets"
    under_rate = np.mean([d.direction == "under_reporting" for d in unders])
    over_rate = np.mean([d.direction == "under_reporting" for d in overs])
    assert under_rate > 0.8, f"planted under-reporting detected on only {under_rate:.0%}"
    assert under_rate > over_rate, "direction did not separate reported-RPE levels"


def test_gate_divergent_sets_are_flagged_for_text_first_presentation():
    """§8.5: when effort and form disagree the number is least informative, so the flag
    must fire — this is the high-effort/low-form quadrant the system exists to surface."""
    c = score_composite(form_score=45.0, expected_rir=0.0, reported_rpe=10.0)
    assert c.divergent is True
    assert c.total is not None, "the number is de-emphasised, not withheld"


# --- §9.5 RIR readiness gate ----------------------------------------------------

from wristset.models.rir import (  # noqa: E402
    MIN_FAILURE_SETS,
    TARGET_FAILURE_SETS,
    assess_rir_readiness,
)


class _Meta:
    def __init__(self, reached_failure, exercise="bench_press"):
        self.reached_failure = reached_failure
        self.exercise = exercise


def _corpus(n_failure, n_censored=0):
    return ([(None, _Meta(True))] * n_failure) + ([(None, _Meta(False))] * n_censored)


def test_rir_is_withheld_below_the_failure_set_minimum():
    """§9.5: below ~15 failure sets the hazard is unstable, so no estimate is shown."""
    r = assess_rir_readiness(_corpus(MIN_FAILURE_SETS - 1, n_censored=50))
    assert r.available is False
    assert r.sets_needed == 1
    assert "more set" in r.explain()


def test_censored_sets_do_not_unlock_rir():
    """Censored sets inform the hazard where they reached, but never observe a failure —
    they cannot identify the high-fatigue end of the curve on their own (§9.5)."""
    assert assess_rir_readiness(_corpus(0, n_censored=200)).available is False


def test_rir_is_provisional_between_the_minimum_and_the_target():
    r = assess_rir_readiness(_corpus(MIN_FAILURE_SETS))
    assert r.available is True and r.provisional is True
    assert "provisional" in r.explain()


def test_rir_is_settled_at_the_target():
    r = assess_rir_readiness(_corpus(TARGET_FAILURE_SETS))
    assert r.available is True and r.provisional is False
    assert r.sets_needed == 0


def test_readiness_explanation_makes_no_promise_about_the_estimate():
    """§8.7: say what is missing, not what the answer will be once it arrives."""
    banned = ("will show", "you will", "should be", "expect to")
    for n in (0, MIN_FAILURE_SETS - 1, MIN_FAILURE_SETS, TARGET_FAILURE_SETS):
        text = assess_rir_readiness(_corpus(n)).explain().lower()
        for phrase in banned:
            assert phrase not in text


def test_demo_withholds_rir_on_thin_history_and_includes_it_when_ready():
    """The product rule end to end: RIR unlocks as failure sets accumulate."""
    from wristset.demo import fit_demo_rir_model

    thin_model, thin = fit_demo_rir_model(seed=1, n_users=3)
    assert thin.available is False
    assert thin_model is None, "no model should be fit from a corpus below the minimum"

    ready_model, ready = fit_demo_rir_model(seed=1, n_users=10)
    assert ready.available is True
    assert ready_model is not None
