"""Layer 3 segmentation tests, including THE GATE (>=95% rep-count exact match).

Milestone 3 is the binding gate (§13): nothing downstream is meaningful unless
segmentation is accurate. The metric is exact-match rate of the detected completed-rep
count against ``reported_reps`` over a held-out synthetic population.
"""

from __future__ import annotations

import numpy as np
import pytest

from wristset.conditioning import condition_set
from wristset.segmentation import build_template, segment_reps, template_distance
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


# --- DTW stage-3 capability -----------------------------------------------------


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
