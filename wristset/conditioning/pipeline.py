"""Conditioning orchestrator: raw set -> ConditionedSet (§5).

Wires the ordered stages together and applies the §5.6 quality gates. The result is the
uniform-grid, world-frame, ZUPT-integrated representation every downstream layer reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from wristset.conditioning.filters import bandpass, lowpass
from wristset.conditioning.frame import to_world_frame
from wristset.conditioning.resample import resample_to_uniform
from wristset.conditioning.zupt import detect_stationary, zupt_integrate

__all__ = ["ConditionedSet", "condition_set"]


@dataclass
class ConditionedSet:
    """Fully conditioned set on a uniform world-frame grid."""

    t: np.ndarray  # (N,) seconds, uniform 1/fs
    fs: float

    a_world: np.ndarray  # (N,3) kinematic-filtered world-frame linear accel
    a_vert: np.ndarray  # (N,) vertical accel, + = up
    a_horiz: np.ndarray  # (N,2) horizontal-plane accel
    up_hat: np.ndarray  # (3,) world up direction

    v_vert: np.ndarray  # (N,) ZUPT vertical velocity
    disp_vert: np.ndarray  # (N,) ZUPT vertical displacement
    v_horiz: np.ndarray  # (N,2) ZUPT horizontal velocity
    disp_horiz: np.ndarray  # (N,2) ZUPT horizontal displacement

    tremor: np.ndarray  # (N,) 8-12 Hz band-passed gyro magnitude

    stationary: np.ndarray  # (N,) bool mask
    anchor_idx: np.ndarray  # stationary anchor sample indices
    gap_mask: np.ndarray  # (N,) bool: inside a large delivery gap
    gap_intervals: list[tuple[float, float]]

    quality: dict = field(default_factory=dict)

    @property
    def n_stationary_intervals(self) -> int:
        return int(self.anchor_idx.size)

    @property
    def low_confidence(self) -> bool:
        return bool(self.quality.get("low_confidence", False))


def condition_set(
    raw: pl.DataFrame,
    *,
    fs: float = 100.0,
    reported_reps: int | None = None,
) -> ConditionedSet:
    """Run the full §5 conditioning pipeline on a raw set.

    ``reported_reps`` (from set metadata) enables the §5.6 stationary-count quality gate:
    a clean set should present at least one stationary anchor per rep.
    """
    # §5.1 resample onto uniform grid + gap flags
    rr = resample_to_uniform(raw, fs=fs)

    # §5.3 world frame + vertical/horizontal split (rotate BEFORE filtering so the
    # low-pass acts on physically-meaningful world axes)
    wf = to_world_frame(rr.lin_acc, rr.grav, rr.quat, fs)

    # §5.4 kinematic low-pass branch (feeds velocity/displacement/tempo/path)
    a_vert_k = lowpass(wf.a_vert, fs)
    a_horiz_k = lowpass(wf.a_horiz, fs)
    a_world_k = lowpass(wf.a_world, fs)

    # §5.4 tremor branch: 8-12 Hz band power of gyro magnitude
    gyro_mag = np.linalg.norm(rr.rot_rate, axis=1)
    tremor = np.abs(bandpass(gyro_mag, fs))

    # §5.5 ZUPT: stationary detection then drift-corrected integration
    stat = detect_stationary(a_world_k, rr.rot_rate, fs)
    v_vert, disp_vert = zupt_integrate(a_vert_k, stat.anchor_idx, fs)
    v_horiz, disp_horiz = zupt_integrate(a_horiz_k, stat.anchor_idx, fs)

    # §5.6 quality gates
    quality = _quality_gates(
        n=rr.t.shape[0],
        gap_intervals=rr.gap_intervals,
        n_stationary=stat.anchor_idx.size,
        reported_reps=reported_reps,
    )

    return ConditionedSet(
        t=rr.t,
        fs=fs,
        a_world=a_world_k,
        a_vert=a_vert_k,
        a_horiz=a_horiz_k,
        up_hat=wf.up_hat,
        v_vert=v_vert,
        disp_vert=disp_vert,
        v_horiz=v_horiz,
        disp_horiz=disp_horiz,
        tremor=tremor,
        stationary=stat.mask,
        anchor_idx=stat.anchor_idx,
        gap_mask=rr.gap_mask,
        gap_intervals=rr.gap_intervals,
        quality=quality,
    )


def _quality_gates(
    *,
    n: int,
    gap_intervals: list[tuple[float, float]],
    n_stationary: int,
    reported_reps: int | None,
) -> dict:
    """§5.6 quality gates. Flags mark a set low-confidence (shown to the user with a
    caveat, excluded from training).

    NOTE — touch-and-go detection (§5.5) is intentionally deferred. Gyro-based ZUPT
    recovers a near-zero-velocity anchor at each rep reversal whether or not the lifter
    dwells, so accuracy degrades gracefully; and the current synthetic generator models
    smooth reversals only, so a touch-and-go flag cannot yet be validated. It returns
    when the generator models sharp (non-zero-velocity) reversals or real data lands.
    """
    flags: list[str] = []
    if gap_intervals:
        flags.append("sample_gaps")
    if reported_reps is not None and n_stationary < reported_reps:
        # fewer stationary anchors than reps -> integration anchors missing (§5.6)
        flags.append("insufficient_stationary")
    return {
        "flags": flags,
        "low_confidence": bool(flags),
        "n_samples": n,
        "n_stationary": n_stationary,
        "n_gaps": len(gap_intervals),
    }
