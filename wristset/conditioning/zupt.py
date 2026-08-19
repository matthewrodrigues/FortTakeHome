"""Zero-velocity-update (ZUPT) integration (§5.5).

Naive double integration of accelerometer data diverges quadratically. We bound it by
detecting stationary intervals (top lockout, bottom pause) where velocity is truly zero,
then integrating between those anchors with a piecewise-linear drift correction so
velocity returns to zero at each anchor. The integration horizon is thus ~1-3 s per
half-rep instead of ~60 s per set, which is what makes rep-to-rep comparison valid even
if absolute displacement carries bias (§5.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from wristset.conditioning.filters import lowpass

__all__ = [
    "StationaryDetection",
    "detect_stationary",
    "zupt_integrate",
    "ACC_STATIONARY_THRESH",
    "GYRO_STATIONARY_THRESH",
    "MIN_STATIONARY_S",
]

# Gyro is the PRIMARY stationary signal: rotation rate tapers to ~0 approaching a
# pause, whereas linear acceleration PEAKS at the bottom turnaround (zero velocity,
# maximum acceleration). An accel-only test therefore misses bottom anchors; the accel
# threshold here is only a loose sanity gate against large-accel outliers (§5.5).
#: |rotation rate| below this (rad/s), sustained, counts as stationary — primary gate.
GYRO_STATIONARY_THRESH: float = 0.6
#: |linear accel| below this (m/s^2) — loose secondary gate, not the discriminator.
ACC_STATIONARY_THRESH: float = 3.0
#: Minimum sustained duration for a stationary interval (§5.5).
MIN_STATIONARY_S: float = 0.10


@dataclass
class StationaryDetection:
    mask: np.ndarray  # (N,) bool
    intervals: list[tuple[int, int]]  # inclusive (start_idx, end_idx) sample ranges
    anchor_idx: np.ndarray  # representative index per interval (midpoint), where v=0


def detect_stationary(
    a_world: np.ndarray,
    rot_rate: np.ndarray,
    fs: float,
    *,
    acc_thresh: float = ACC_STATIONARY_THRESH,
    gyro_thresh: float = GYRO_STATIONARY_THRESH,
    min_dur_s: float = MIN_STATIONARY_S,
) -> StationaryDetection:
    """Detect sustained low-motion intervals from linear-accel and gyro magnitudes.

    The accel/gyro *vectors* are low-passed at ~3 Hz before taking magnitude. This
    strips the 8-12 Hz tremor (which is present even during isometric holds and would
    otherwise raise the pause-time magnitude above threshold) while preserving the
    sub-3 Hz rep motion, giving a clean separation between pause and movement.
    """
    a_mag = np.linalg.norm(lowpass(a_world, fs, cutoff=3.0, order=2), axis=1)
    w_mag = np.linalg.norm(lowpass(rot_rate, fs, cutoff=3.0, order=2), axis=1)
    quiet = (a_mag < acc_thresh) & (w_mag < gyro_thresh)

    min_samples = max(int(round(min_dur_s * fs)), 1)
    intervals: list[tuple[int, int]] = []
    i, n = 0, quiet.shape[0]
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) >= min_samples:
                intervals.append((i, j - 1))
            i = j
        else:
            i += 1

    mask = np.zeros(n, dtype=bool)
    for a, b in intervals:
        mask[a : b + 1] = True
    anchor_idx = np.array([(a + b) // 2 for a, b in intervals], dtype=int)
    return StationaryDetection(mask=mask, intervals=intervals, anchor_idx=anchor_idx)


def zupt_integrate(
    a_signal: np.ndarray,
    anchor_idx: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate acceleration -> velocity -> displacement with ZUPT drift correction.

    ``a_signal`` may be 1-D (vertical) or 2-D ``(N, k)`` (e.g. horizontal plane); the
    same anchors are applied per component. Velocity is forced to zero at each anchor
    index by subtracting a piecewise-linear drift fit through the raw velocity sampled
    at the anchors. Returns ``(velocity, displacement)`` with the same shape as input.
    """
    a = np.asarray(a_signal, dtype=np.float64)
    squeeze = a.ndim == 1
    if squeeze:
        a = a[:, None]
    n = a.shape[0]
    dt = 1.0 / fs

    v_raw = cumulative_trapezoid(a, dx=dt, axis=0, initial=0.0)

    if anchor_idx.size >= 1:
        t = np.arange(n)
        drift = np.empty_like(v_raw)
        for k in range(a.shape[1]):
            # piecewise-linear (held flat beyond the outer anchors) drift through v_raw@anchors
            drift[:, k] = np.interp(t, anchor_idx, v_raw[anchor_idx, k])
        v = v_raw - drift
    else:
        v = v_raw  # no anchors -> uncorrected (will be flagged low-confidence upstream)

    disp = cumulative_trapezoid(v, dx=dt, axis=0, initial=0.0)

    if squeeze:
        return v[:, 0], disp[:, 0]
    return v, disp
