"""Layer 3 segmentation tests, including THE GATE (>=95% rep-count exact match).

Milestone 3 is the binding gate (§13): nothing downstream is meaningful unless
segmentation is accurate. The metric is exact-match rate of the detected completed-rep
count against ``reported_reps`` over a held-out synthetic population.
"""

from __future__ import annotations

import numpy as np
import pytest

from wristset.conditioning import condition_set
from wristset.segmentation import (
    build_template,
    detect_active_window,
    segment_reps,
    template_distance,
)
from wristset.segmentation.reps import (
    MAX_PERIOD_S,
    MIN_PERIOD_S,
    _estimate_period_samples,
)
from wristset.synth import SetParams, generate_set

# Degradation grades. clean+mild+moderate is the realistic prototype-capture operating
# distribution (§5.5 recommends brief pauses during collection); "hard" is the degraded
# regime where accuracy declines and sets are often quality-flagged.
_GRADES = {
    "clean": {},
    "mild": dict(vel_decay=0.35, rom_collapse=0.12, tremor_growth=2.5),
    "moderate": dict(vel_decay=0.45, rom_collapse=0.20, tremor_growth=3.0,
                     ecc_shorten=0.35, path_wobble_m=0.03),
    "hard": dict(vel_decay=0.55, rom_collapse=0.28, tremor_gain=0.4, tremor_growth=3.5,
                 ecc_shorten=0.4, path_wobble_m=0.04),
}


def _population(grade_kw, seeds=range(1, 4)):
    for ex, rom in [("bench_press", 0.45), ("back_squat", 0.55)]:
        for cap in (5, 6, 8, 10, 12):
            for failure in (True, False):
                for seed in seeds:
                    stop_rir = 0 if failure else 2
                    if cap - stop_rir < 3:
                        continue
                    g = generate_set(SetParams(
                        exercise=ex, rom_m=rom, capacity=cap, stop_rir=stop_rir,
                        reached_failure=failure, seed=seed, **grade_kw))
                    yield ex, g


def _exact_match_rate(grade_kw, seeds=range(1, 4)) -> float:
    match = total = 0
    for ex, g in _population(grade_kw, seeds):
        cs = condition_set(g.raw, reported_reps=g.reported_reps)
        res = segment_reps(cs, exercise=ex, reached_failure=g.reached_failure)
        match += res.n_completed == g.reported_reps
        total += 1
    return match / total


# --- basic behavior -------------------------------------------------------------


def test_clean_failure_set_counts(failure_bench_set):
    cs = condition_set(failure_bench_set.raw, reported_reps=failure_bench_set.reported_reps)
    res = segment_reps(cs, exercise="bench_press", reached_failure=True)
    assert res.n_completed == failure_bench_set.reported_reps
    # the failed attempt is present but marked incomplete, as the last excursion
    assert res.reps[-1].completed is False
    assert sum(1 for r in res.reps if not r.completed) == 1


def test_censored_set_all_completed(censored_squat_set):
    cs = condition_set(censored_squat_set.raw, reported_reps=censored_squat_set.reported_reps)
    res = segment_reps(cs, exercise="back_squat", reached_failure=False)
    assert res.n_completed == censored_squat_set.reported_reps
    assert all(r.completed for r in res.reps)


def test_phase_boundary_ordering(failure_bench_set):
    cs = condition_set(failure_bench_set.raw)
    res = segment_reps(cs, exercise="bench_press", reached_failure=True)
    for r in res.reps:
        # bench is eccentric-first: top -> bottom -> top
        assert r.t_start < r.t_bottom < r.t_end


def test_period_estimated(failure_bench_set):
    cs = condition_set(failure_bench_set.raw)
    res = segment_reps(cs, exercise="bench_press", reached_failure=True)
    assert res.period_s is not None and 0.8 < res.period_s < 4.0


# --- THE GATE -------------------------------------------------------------------


def test_segmentation_gate_operating_distribution():
    """>=95% exact match on the realistic operating distribution (clean+mild+moderate)."""
    rates = {g: _exact_match_rate(_GRADES[g]) for g in ("clean", "mild", "moderate")}
    overall = np.mean(list(rates.values()))
    assert overall >= 0.95, f"segmentation gate not met: {rates} (overall {overall:.1%})"
    # clean captures must be effectively perfect
    assert rates["clean"] >= 0.99


def test_segmentation_degraded_regime_guard():
    """Degraded regime (hard) is not the gate, but guard against gross regressions."""
    assert _exact_match_rate(_GRADES["hard"]) >= 0.75


# --- §6.1 period estimation bounds ----------------------------------------------


def test_period_estimate_stays_in_rep_band():
    """The autocorrelation estimate must never leave the physiological rep band.

    Regression: on long real recordings the unbounded estimator locked onto the
    set-rest envelope (57-101 s vs a ~4.5 s median), which drove
    ``min_dist = 0.5 * period`` high enough to suppress every rep but one.
    """
    for seed in (1, 2, 3):
        for cap in (5, 10):
            g = generate_set(SetParams(exercise="bench_press", capacity=cap,
                                       stop_rir=2, reached_failure=False, seed=seed))
            cs = condition_set(g.raw)
            res = segment_reps(cs, exercise="bench_press", reached_failure=False)
            if res.period_s is not None:
                assert MIN_PERIOD_S <= res.period_s <= MAX_PERIOD_S


def test_runaway_period_is_rejected():
    """A period far outside the rep band is discarded rather than propagated."""
    assert _estimate_period_samples(np.zeros(1000), 100.0) is None
    # a very slow single ramp has no in-band periodicity -> no usable estimate
    slow = np.linspace(0.0, 1.0, 6000)
    p = _estimate_period_samples(slow, 100.0)
    assert p is None or p / 100.0 <= MAX_PERIOD_S


# --- §6.1 set detection ---------------------------------------------------------


def test_active_window_is_noop_on_clean_capture(failure_bench_set):
    """A capture that is all working set yields a window spanning the recording (§6.1:
    with user start/stop this is a validation check, not a detection problem)."""
    cs = condition_set(failure_bench_set.raw)
    w = detect_active_window(cs)
    assert w.is_full_recording
    assert w.active_fraction >= 0.95


def test_ragged_edges_are_flagged_and_trimming_helps():
    """Recordings with non-lifting edges must be flagged, and trimming must move the
    count TOWARD truth rather than away from it.

    Uses the generator's own ragged-edge model (``lead_s``/``tail_s``), which emits
    energetic but arrhythmic setup/racking motion — the real-corpus condition. Without
    trimming this motion is counted as reps (measured bias ~+4 reps).
    """
    on_err, off_err, flagged = [], [], []
    for seed in range(1, 9):
        g = generate_set(SetParams(exercise="bench_press", rom_m=0.45, capacity=10,
                                   stop_rir=2, reached_failure=False, seed=seed,
                                   lead_s=8.0, tail_s=8.0))
        cs = condition_set(g.raw)
        on = segment_reps(cs, exercise="bench_press", reached_failure=False, trim=True)
        off = segment_reps(cs, exercise="bench_press", reached_failure=False, trim=False)
        flagged.append(on.active_fraction < 0.95)
        on_err.append(abs(on.n_completed - g.reported_reps))
        off_err.append(abs(off.n_completed - g.reported_reps))

    # Most ragged recordings must be flagged. Not all: where the edge motion is not
    # separable the detector declines to trim by design (TRIM_EVIDENCE_FRACTION), which
    # is the safe outcome rather than a failure.
    assert sum(flagged) >= 6, f"ragged edges flagged on only {sum(flagged)}/8"

    # trimming must reduce mean absolute rep-count error on ragged recordings
    assert np.mean(on_err) < np.mean(off_err), (
        f"trimming did not help: {np.mean(on_err):.2f} vs {np.mean(off_err):.2f}"
    )
    # and it must recover the exact count on a meaningful share of them
    assert sum(e == 0 for e in on_err) >= 3, f"exact matches: {on_err}"


def test_trim_preserves_index_alignment():
    """Emitted indices must address the FULL conditioned arrays, not the trimmed slice."""
    g = generate_set(SetParams(exercise="bench_press", capacity=8, stop_rir=0,
                               reached_failure=True, seed=1, lead_s=6.0, tail_s=6.0))
    cs = condition_set(g.raw)
    res = segment_reps(cs, exercise="bench_press", reached_failure=True, trim=True)
    assert res.reps
    for r in res.reps:
        assert 0 <= r.i_start < cs.t.shape[0]
        assert 0 <= r.i_bottom < cs.t.shape[0]
        assert 0 <= r.i_end < cs.t.shape[0]
        # index and time must refer to the same sample
        assert cs.t[r.i_bottom] == pytest.approx(r.t_bottom, abs=1e-6)
        assert cs.t[r.i_start] == pytest.approx(r.t_start, abs=1e-6)


def test_rhythmic_window_is_not_fragmented_by_a_missed_bottom():
    """A single missed detection must not split one set into two windows.

    Regression: taking only the *longest* rhythmic run fragmented continuous sets — on the
    real corpus the bottoms discarded that way were themselves 44-100% rhythmic at the same
    cadence, i.e. real reps. Runs separated by less than ``MERGE_GAP_CADENCES`` are merged.
    A long clean set is the sensitive case: the more reps, the more chances to drop one.
    """
    for seed in (1, 2, 3):
        g = generate_set(SetParams(exercise="bench_press", rom_m=0.45, capacity=12,
                                   stop_rir=2, reached_failure=False, seed=seed))
        cs = condition_set(g.raw)
        w = detect_active_window(cs)
        # a clean capture must not be carved up: the window has to span nearly all of it
        assert w.active_fraction >= 0.9, f"seed{seed}: set fragmented ({w.active_fraction:.2f})"


def test_trim_disabled_matches_untrimmed_window(failure_bench_set):
    """trim=False leaves segmentation on the full recording (no active window)."""
    cs = condition_set(failure_bench_set.raw)
    res = segment_reps(cs, exercise="bench_press", reached_failure=True, trim=False)
    assert res.active_window is None
    assert res.active_fraction == 1.0


# --- DTW stage-3 capability -----------------------------------------------------


def test_dtw_rejection_does_not_discard_genuine_degraded_reps():
    """Stage-3 rejection must not fire on real reps in the degraded regime.

    Heavy fatigue legitimately deforms rep shape — that deformation is the signal the
    product measures, so a template-distance cut set too tight destroys it. Measured: at
    ratio 3.0, 20.8% of genuine completed reps were rejected and the hard-regime gate fell
    83.3% -> 38.3%. This pins the conservative calibration.
    """
    hard = dict(vel_decay=0.55, rom_collapse=0.28, tremor_gain=0.4, tremor_growth=3.5,
                ecc_shorten=0.4, path_wobble_m=0.04)
    for seed in (1, 2, 3, 4, 5):
        g = generate_set(SetParams(exercise="bench_press", capacity=8, stop_rir=0,
                                   reached_failure=True, seed=seed, **hard))
        cs = condition_set(g.raw, reported_reps=g.reported_reps)
        res = segment_reps(cs, exercise="bench_press", reached_failure=True)
        assert res.n_completed == g.reported_reps, (
            f"seed{seed}: DTW rejected genuine reps ({res.n_completed} vs {g.reported_reps})"
        )


def test_dtw_template_flags_failed_attempt():
    g = generate_set(SetParams(exercise="bench_press", capacity=8, stop_rir=0,
                               reached_failure=True, seed=3))
    cs = condition_set(g.raw, reported_reps=g.reported_reps)
    res = segment_reps(cs, exercise="bench_press", reached_failure=True)
    tmpl = build_template(cs, res.reps)
    assert tmpl is not None
    dists = {r.rep_index: template_distance(cs, r, tmpl) for r in res.reps}
    completed = [d for r, d in dists.items() if res.reps[r - 1].completed]
    failed = [d for r, d in dists.items() if not res.reps[r - 1].completed]
    # the failed attempt is the least template-like rep in the set
    assert max(failed) >= max(completed)
