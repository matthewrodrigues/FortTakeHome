"""Resample raw irregular samples onto a uniform grid (§5.1).

Watch sensor delivery is frequently irregular; silently treating irregular samples as
uniform corrupts every downstream integration (§3.3). We interpolate onto a uniform
100 Hz grid using the *recorded* timestamps, and flag any gap larger than
``GAP_THRESHOLD_S`` so the affected reps can later be marked low-confidence (§5.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from wristset.contract import RAW_COLUMNS

__all__ = ["ResampledRaw", "resample_to_uniform", "GAP_THRESHOLD_S"]

#: Gaps longer than this (seconds) are flagged (§5.1).
GAP_THRESHOLD_S: float = 0.050

_VEC_COLS = {
    "lin_acc": ("lin_acc_x", "lin_acc_y", "lin_acc_z"),
    "rot_rate": ("rot_rate_x", "rot_rate_y", "rot_rate_z"),
    "grav": ("grav_x", "grav_y", "grav_z"),
}
_QUAT_COLS = ("quat_w", "quat_x", "quat_y", "quat_z")


@dataclass
class ResampledRaw:
    """Uniformly-sampled raw signals in the device frame."""

    t: np.ndarray  # (N,) seconds from set start, uniform 1/fs spacing
    fs: float
    lin_acc: np.ndarray  # (N,3) device frame
    rot_rate: np.ndarray  # (N,3) device frame
    grav: np.ndarray  # (N,3) device frame
    quat: np.ndarray  # (N,4) normalized, device->world
    gap_mask: np.ndarray  # (N,) bool: sample lies inside a >threshold delivery gap
    gap_intervals: list[tuple[float, float]]  # (t_start, t_end) of each large gap


def resample_to_uniform(raw: pl.DataFrame, fs: float = 100.0) -> ResampledRaw:
    """Interpolate a raw set onto a uniform ``fs`` Hz grid; flag large delivery gaps."""
    t_ns = raw["t_ns"].to_numpy()
    t_src = (t_ns - t_ns[0]) / 1e9  # seconds from set start
    duration = t_src[-1]

    n = int(np.floor(duration * fs)) + 1
    t_uniform = np.arange(n) / fs

    def interp_cols(cols: tuple[str, ...]) -> np.ndarray:
        return np.stack(
            [np.interp(t_uniform, t_src, raw[c].to_numpy()) for c in cols], axis=1
        )

    lin_acc = interp_cols(_VEC_COLS["lin_acc"])
    rot_rate = interp_cols(_VEC_COLS["rot_rate"])
    grav = interp_cols(_VEC_COLS["grav"])

    quat = interp_cols(_QUAT_COLS)
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    quat = quat / norms  # renormalize after componentwise interpolation

    # gap detection on the ORIGINAL timestamps
    dts = np.diff(t_src)
    gap_intervals: list[tuple[float, float]] = []
    gap_mask = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(dts > GAP_THRESHOLD_S):
        g0, g1 = float(t_src[i]), float(t_src[i + 1])
        gap_intervals.append((g0, g1))
        gap_mask |= (t_uniform >= g0) & (t_uniform <= g1)

    return ResampledRaw(
        t=t_uniform,
        fs=fs,
        lin_acc=lin_acc,
        rot_rate=rot_rate,
        grav=grav,
        quat=quat,
        gap_mask=gap_mask,
        gap_intervals=gap_intervals,
    )


def _assert_raw_columns(raw: pl.DataFrame) -> None:
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"raw frame missing columns for resample: {missing}")
