"""Layer 2 conditioning tests, including the Phase 1 GATE.

Gate (mirrors milestone 2): reconstructed per-rep vertical displacement is within ~15%
of the generator's ground-truth ROM. Rep boundaries here come from ground truth because
segmentation is Phase 2 — this isolates the conditioning/ZUPT accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from wristset.conditioning import condition_set
from wristset.conditioning.frame import to_world_frame
from wristset.conditioning.resample import resample_to_uniform
from wristset.conditioning.zupt import detect_stationary, zupt_integrate
from wristset.synth import SetParams, generate_set


# --- §5.1 resample --------------------------------------------------------------


def test_resample_uniform_spacing(failure_bench_set):
    rr = resample_to_uniform(failure_bench_set.raw, fs=100.0)
    dt = np.diff(rr.t)
    assert np.allclose(dt, 0.01, atol=1e-9)
    assert rr.quat.shape[1] == 4
    assert np.allclose(np.linalg.norm(rr.quat, axis=1), 1.0, atol=1e-6)


def test_gap_flagging_fires():
    g = generate_set(SetParams(capacity=6, gap_prob=0.02, gap_ms=120.0, seed=5))
    cs = condition_set(g.raw, reported_reps=g.reported_reps)
    assert len(cs.gap_intervals) > 0
    assert cs.gap_mask.any()
    assert "sample_gaps" in cs.quality["flags"]


def test_no_gap_flag_on_clean_capture(failure_bench_set):
    cs = condition_set(failure_bench_set.raw, reported_reps=failure_bench_set.reported_reps)
    assert cs.gap_intervals == []


# --- §5.3 world frame -----------------------------------------------------------


def test_gravity_defines_up(failure_bench_set):
    rr = resample_to_uniform(failure_bench_set.raw)
    wf = to_world_frame(rr.lin_acc, rr.grav, rr.quat, rr.fs)
    # gravity rotates to ~[0,0,-g]; up_hat ~ +z
    assert np.allclose(wf.grav_world.mean(axis=0), [0, 0, -9.80665], atol=0.2)
    assert np.allclose(wf.up_hat, [0, 0, 1], atol=1e-2)
    # linear accel is genuinely gravity-free (physiological magnitude, not ~1g)
    assert np.median(np.linalg.norm(wf.a_world, axis=1)) < 5.0


# --- §5.5 ZUPT ------------------------------------------------------------------


def test_stationary_intervals_detected(failure_bench_set):
    cs = condition_set(failure_bench_set.raw, reported_reps=failure_bench_set.reported_reps)
    # expect roughly a leading top pause + ~2 anchors per rep
    assert cs.n_stationary_intervals >= failure_bench_set.reported_reps


def test_velocity_near_zero_at_anchors(failure_bench_set):
    rr = resample_to_uniform(failure_bench_set.raw)
    wf = to_world_frame(rr.lin_acc, rr.grav, rr.quat, rr.fs)
    stat = detect_stationary(wf.a_world, rr.rot_rate, rr.fs)
    v, _ = zupt_integrate(wf.a_vert, stat.anchor_idx, rr.fs)
    assert np.max(np.abs(v[stat.anchor_idx])) < 1e-6  # ZUPT forces v=0 at anchors


# --- Phase 1 GATE ---------------------------------------------------------------


def _per_rep_rom_errors(cs, reps):
    """Return {rep_index: relative ROM error} for completed reps."""
    out = {}
    for r in reps:
        if not r.completed:
            continue
        m = (cs.t >= r.t_start) & (cs.t <= r.t_end)
        rom_est = cs.disp_vert[m].max() - cs.disp_vert[m].min()
        out[r.rep_index] = abs(rom_est - r.rom_true_m) / r.rom_true_m
    return out


@pytest.mark.parametrize("exercise,rom", [("bench_press", 0.45), ("back_squat", 0.55)])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_rom_reconstruction_failure_set(exercise, rom, seed):
    """Failure sets: reconstruction is within 15% for every clean rep. The single rep
    immediately before the failed attempt is the physically hardest case (no clean
    lockout after it, §4.2 'qualitatively different') and is allowed a looser bound."""
    g = generate_set(
        SetParams(exercise=exercise, rom_m=rom, capacity=8, stop_rir=0,
                  reached_failure=True, seed=seed)
    )
    cs = condition_set(g.raw, reported_reps=g.reported_reps)
    errs = _per_rep_rom_errors(cs, g.reps)
    boundary = max(errs)  # last completed rep, adjacent to the failed attempt

    mean_err = np.mean(list(errs.values()))
    assert mean_err < 0.15, f"{exercise} seed{seed}: mean ROM err {mean_err:.1%}"
    for idx, e in errs.items():
        limit = 0.20 if idx == boundary else 0.15
        assert e < limit, f"{exercise} seed{seed} rep{idx}: ROM err {e:.1%} >= {limit:.0%}"


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_rom_reconstruction_clean_ending(seed):
    """Censored sets end with a clean lockout: EVERY rep is within 15% (milestone 2)."""
    g = generate_set(
        SetParams(exercise="bench_press", rom_m=0.45, capacity=10, stop_rir=2,
                  reached_failure=False, seed=seed)
    )
    cs = condition_set(g.raw, reported_reps=g.reported_reps)
    errs = _per_rep_rom_errors(cs, g.reps)
    worst = max(errs.values())
    assert worst < 0.15, f"seed{seed}: worst per-rep ROM err {worst:.1%}"


# --- §5.6 quality gates ---------------------------------------------------------


def test_clean_set_is_high_confidence(failure_bench_set):
    cs = condition_set(failure_bench_set.raw, reported_reps=failure_bench_set.reported_reps)
    assert cs.low_confidence is False
