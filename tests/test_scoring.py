"""Layer 5a form-subscore tests (§8.3-8.4), including the Phase 4 GATE.

Gate (milestone 5): the four deterministic form subscores, and their equal-weight
composite, must separate sets the synthetic generator built as CLEAN from sets it built
as DEGRADED. This is the first stand-alone shippable product, so it is validated directly
against the generator's ground-truth degradation knobs.
"""

from __future__ import annotations

import numpy as np
import pytest

from wristset.conditioning import condition_set
from wristset.features import extract_set_features
from wristset.features.baseline import EARLY_SET_REPS
from wristset.scoring import (
    MIN_REPS_BEYOND_REFERENCE,
    FormSubscores,
    NormRef,
    score_form,
    squash,
)
from wristset.segmentation import segment_reps
from wristset.synth import SetParams, generate_set

#: subscore names, in the fixed order §8.3 lists them
SUBSCORES = ("rom_completeness", "path_consistency", "tempo_control", "stability")

# A clean set barely fatigues; a degraded set collapses velocity/ROM and grows tremor.
# Mirrors the regime dicts in test_segmentation so both layers degrade the same way.
_CLEAN = dict(vel_decay=0.08, rom_collapse=0.02, tremor_growth=1.1, ecc_shorten=0.05)
_DEGRADED = dict(vel_decay=0.5, rom_collapse=0.25, tremor_growth=3.2, ecc_shorten=0.35)


def _score(seed=1, capacity=10, **kw):
    g = generate_set(SetParams(exercise="bench_press", rom_m=0.45, capacity=capacity,
                               stop_rir=0, reached_failure=True, seed=seed, **kw))
    cs = condition_set(g.raw, reported_reps=g.reported_reps)
    res = segment_reps(cs, exercise="bench_press", reached_failure=g.reached_failure)
    sf = extract_set_features(cs, res.reps, exercise="bench_press", load_kg=80.0)
    return score_form(sf)


# --- §8.4 squash: the monotone deviation -> 0-100 map both norm paths share ------


def test_squash_is_100_at_zero_deviation():
    assert squash(0.0, scale=0.1) == 100.0


def test_squash_is_50_at_one_scale_of_deviation():
    # scale is defined as the deviation that halves the score
    assert abs(squash(0.1, scale=0.1) - 50.0) < 1e-9


def test_squash_is_monotone_decreasing_in_deviation():
    scale = 0.1
    devs = np.linspace(0.0, 1.0, 50)
    scores = [squash(d, scale=scale) for d in devs]
    assert all(b <= a for a, b in zip(scores, scores[1:]))


def test_squash_clamps_deviation_better_than_reference_to_100():
    # a set BETTER than its reference (negative deviation) is not scored above 100
    assert squash(-0.5, scale=0.1) == 100.0


def test_squash_approaches_zero_for_large_deviation():
    assert squash(10.0, scale=0.1) < 1.0


# --- §8.3 score_form structure & ranges -----------------------------------------


def test_score_form_returns_four_subscores_and_a_composite():
    fs = _score(**_CLEAN)
    assert isinstance(fs, FormSubscores)
    names = {s.name for s in fs.subscores}
    assert names == set(SUBSCORES)


def test_all_present_subscores_are_in_range():
    fs = _score(**_DEGRADED)
    for s in fs.subscores:
        if s.score is not None:
            assert 0.0 <= s.score <= 100.0, f"{s.name}={s.score}"
    assert fs.form_score is None or 0.0 <= fs.form_score <= 100.0


def test_form_score_is_equal_weight_mean_of_present_subscores():
    fs = _score(**_DEGRADED)
    present = [s.score for s in fs.subscores if s.score is not None]
    assert present, "expected at least one scored subscore"
    assert fs.form_score == pytest.approx(float(np.mean(present)))


# --- §8.3 subscore semantics ----------------------------------------------------


def test_degraded_rom_scores_below_clean_rom():
    clean = _score(seed=3, **_CLEAN).by_name("rom_completeness").score
    degraded = _score(seed=3, **_DEGRADED).by_name("rom_completeness").score
    assert degraded < clean


def test_each_scored_subscore_has_at_least_one_pointer():
    fs = _score(**_DEGRADED)
    for s in fs.subscores:
        if s.score is not None:
            assert len(s.pointers) >= 1, f"{s.name} has no pointer"


def test_pointer_text_states_measured_change_not_coaching_cues():
    """§8.7 language discipline: pointers describe change, never prescribe technique."""
    banned = ("keep your", "chest up", "you should", "tuck", "brace", "squeeze",
              "elbows", "sit back", "drive through")
    fs = _score(**_DEGRADED)
    for s in fs.subscores:
        for p in s.pointers:
            low = p.text.lower()
            for phrase in banned:
                assert phrase not in low, f"coaching cue {phrase!r} in {p.text!r}"


# --- §8.4 normalization / cold start --------------------------------------------


def test_cold_start_scores_are_flagged_provisional():
    fs = _score(**_DEGRADED)
    assert fs.provisional is True
    assert all(s.provisional for s in fs.subscores if s.score is not None)


def test_norm_ref_clears_the_provisional_flag():
    g = generate_set(SetParams(exercise="bench_press", rom_m=0.45, capacity=10,
                               stop_rir=0, reached_failure=True, seed=1, **_DEGRADED))
    cs = condition_set(g.raw, reported_reps=g.reported_reps)
    res = segment_reps(cs, exercise="bench_press", reached_failure=g.reached_failure)
    sf = extract_set_features(cs, res.reps, exercise="bench_press", load_kg=80.0)

    ref = NormRef(mean={n: 0.0 for n in SUBSCORES}, std={n: 0.2 for n in SUBSCORES})
    fs = score_form(sf, norm_ref=ref)
    assert fs.provisional is False
    assert all(not s.provisional for s in fs.subscores if s.score is not None)


def test_short_set_is_unscored_not_scored_perfect():
    """A set too short to sustain an in-set reference must yield None, not 100.

    Regression: every subscore compares the set against its own opening ``n_ref_reps``
    reps, so on a 3-4 rep set the comparison is against ITSELF — deviation collapses to ~0
    and all four subscores squashed to a confident 100.0. That reads as "perfect form" when
    it means "no information", violating the Phase-3 invariant that a missing reference is
    never reported as a good score.
    """
    for capacity in (3, 4):
        g = generate_set(SetParams(exercise="bench_press", rom_m=0.45, capacity=capacity,
                                   stop_rir=0, reached_failure=True, seed=1))
        cs = condition_set(g.raw, reported_reps=g.reported_reps)
        res = segment_reps(cs, exercise="bench_press", reached_failure=True)
        sf = extract_set_features(cs, res.reps, exercise="bench_press", load_kg=80.0)
        fs = score_form(sf)

        assert fs.form_score is None, f"capacity {capacity} scored {fs.form_score}"
        assert all(s.score is None for s in fs.subscores)
        # the four subscores are still reported, so callers see WHICH are unscored
        assert {s.name for s in fs.subscores} == set(SUBSCORES)


def test_set_just_past_the_reference_threshold_is_scored():
    """The guard must not swallow ordinary sets: once there are enough reps beyond the
    reference block, scoring proceeds normally."""
    g = generate_set(SetParams(exercise="bench_press", rom_m=0.45, capacity=6,
                               stop_rir=0, reached_failure=True, seed=1, **_DEGRADED))
    cs = condition_set(g.raw, reported_reps=g.reported_reps)
    res = segment_reps(cs, exercise="bench_press", reached_failure=True)
    sf = extract_set_features(cs, res.reps, exercise="bench_press", load_kg=80.0)
    n_completed = sum(1 for r in sf.reps if r.completed)
    assert n_completed >= EARLY_SET_REPS + MIN_REPS_BEYOND_REFERENCE

    fs = score_form(sf)
    assert fs.form_score is not None
    assert any(s.score is not None for s in fs.subscores)


def test_none_scored_subscore_is_excluded_from_the_composite():
    fs = _score(**_DEGRADED)
    # inject a None-scored subscore; the composite must ignore it, not treat it as 0
    from wristset.scoring import FormSubscore, FormSubscores
    present = [s for s in fs.subscores if s.score is not None]
    mean_present = float(np.mean([s.score for s in present]))
    augmented = FormSubscores(
        subscores=fs.subscores + [FormSubscore("bogus", None, True)]
    )
    assert augmented.form_score == pytest.approx(mean_present)


# --- Phase 4 GATE (milestone 5) -------------------------------------------------


def test_gate_form_score_separates_clean_from_degraded_across_seeds():
    """The composite form score, and every subscore, must read a degraded set as worse
    than a clean one across independent recordings."""
    for seed in (1, 2, 3, 4, 5):
        clean = _score(seed=seed, **_CLEAN)
        degraded = _score(seed=seed, **_DEGRADED)
        assert clean.form_score is not None and degraded.form_score is not None
        assert degraded.form_score < clean.form_score, f"composite, seed {seed}"

        for name in SUBSCORES:
            cs_, ds_ = clean.by_name(name).score, degraded.by_name(name).score
            if cs_ is not None and ds_ is not None:
                assert ds_ <= cs_ + 1e-6, f"{name} did not degrade, seed {seed}"


def test_pointers_trace_back_to_their_driving_feature():
    """§8.6 debuggability: every subscore's pointers name the feature they came from."""
    expected = {
        "rom_completeness": "rom_vertical",
        "path_consistency": "path_dtw_baseline",
        "tempo_control": "ecc_duration",
        "stability": "tremor_power_8_12",
    }
    fs = _score(**_DEGRADED)
    for s in fs.subscores:
        if s.score is not None:
            assert all(p.feature == expected[s.name] for p in s.pointers)


def test_z_score_path_rates_a_typical_set_higher_than_the_fixed_scale():
    """When the user's history says this much change is NORMAL for them (high mean
    deviation), the personalized z-score must forgive it, scoring above the fixed-scale
    number that assumes a generic 'notable' threshold."""
    g = generate_set(SetParams(exercise="bench_press", rom_m=0.45, capacity=10,
                               stop_rir=0, reached_failure=True, seed=2, **_DEGRADED))
    cs = condition_set(g.raw, reported_reps=g.reported_reps)
    res = segment_reps(cs, exercise="bench_press", reached_failure=g.reached_failure)
    sf = extract_set_features(cs, res.reps, exercise="bench_press", load_kg=80.0)

    fixed = score_form(sf).by_name("rom_completeness").score
    # this user historically loses a lot of depth, with little spread -> this set is typical
    ref = NormRef(mean={"rom_completeness": 0.5}, std={"rom_completeness": 0.1})
    personalized = score_form(sf, norm_ref=ref).by_name("rom_completeness").score
    assert personalized > fixed
