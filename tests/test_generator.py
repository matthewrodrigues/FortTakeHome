"""Synthetic generator sanity + ground-truth label correctness."""

from __future__ import annotations

import numpy as np

from wristset.contract import validate_raw
from wristset.synth import SetParams, generate_set, generate_session


def test_raw_frame_is_contract_valid(failure_bench_set):
    validate_raw(failure_bench_set.raw)  # raises on any violation


def test_sample_rate_roughly_nominal(failure_bench_set):
    t = failure_bench_set.raw["t_ns"].to_numpy() / 1e9
    hz = (len(t) - 1) / (t[-1] - t[0])
    assert 80 < hz < 120  # ~100 Hz with jitter


def test_failure_set_labels(failure_bench_set):
    gt = failure_bench_set.ground_truth
    assert gt.reached_failure is True
    assert failure_bench_set.reported_reps == 8  # completed reps
    assert gt.failure_rep == 9  # the failed attempt
    # exactly one incomplete rep, and it is the last
    incomplete = [r for r in failure_bench_set.reps if not r.completed]
    assert len(incomplete) == 1 and incomplete[0].rep_index == 9


def test_censored_set_has_no_failure(censored_squat_set):
    gt = censored_squat_set.ground_truth
    assert gt.reached_failure is False
    assert gt.failure_rep is None
    assert censored_squat_set.reported_reps == 8  # capacity 10 - stop_rir 2
    assert all(r.completed for r in censored_squat_set.reps)


def test_mechanical_rpe_reflects_stop_rir(censored_squat_set):
    # stopped with 2 RIR -> mechanical RPE 8
    assert censored_squat_set.ground_truth.mechanical_rpe_true == 8.0


def test_fatigue_reduces_concentric_velocity(failure_bench_set):
    completed = [r for r in failure_bench_set.reps if r.completed]
    first, last = completed[0], completed[-1]
    assert last.conc_mean_vel_true < first.conc_mean_vel_true


def test_rpe_bias_shifts_reported_downward():
    # a strong under-reporter should report below the mechanical truth on average
    diffs = []
    for s in range(20):
        g = generate_set(
            SetParams(capacity=8, stop_rir=0, reached_failure=True, rpe_bias=-2.0, seed=s)
        )
        diffs.append(g.reported_rpe - g.ground_truth.mechanical_rpe_true)
    assert np.mean(diffs) < -0.5


def test_ground_truth_rom_positive(failure_bench_set):
    assert all(r.rom_true_m > 0 for r in failure_bench_set.reps)


def test_generate_session_shares_session_id():
    sets = generate_session([SetParams(seed=0), SetParams(seed=0, stop_rir=3, reached_failure=False)])
    assert len({g.session_id for g in sets}) == 1
    assert [g.set_index for g in sets] == [1, 2]
