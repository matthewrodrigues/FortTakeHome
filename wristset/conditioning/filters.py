"""Zero-phase filtering (§5.4).

Two parallel branches from the same conditioned signal:

    Kinematic : 4th-order zero-phase Butterworth low-pass (~10-15 Hz) -> velocity,
                displacement, path, tempo.
    Tremor    : band-pass 8-12 Hz -> tremor power features.

Zero-phase (``sosfiltfilt``) is mandatory: a causal filter introduces lag that
corrupts rep-boundary timing (§5.4). SOS form is used for numerical stability.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

__all__ = ["lowpass", "highpass", "bandpass", "KINEMATIC_CUTOFF_HZ", "TREMOR_BAND_HZ", "DISP_DRIFT_HZ"]

#: Displacement drift-removal cutoff. Rep fundamental is 0.3-1 Hz (§3.3); drift from ZUPT
#: residuals sits below this, so a gentle high-pass removes drift without touching reps.
DISP_DRIFT_HZ: float = 0.15

#: Kinematic low-pass cutoff. 10 Hz sits just below the 8-12 Hz tremor band so it keeps
#: rep kinematics while attenuating tremor leakage into velocity/displacement (§5.4).
KINEMATIC_CUTOFF_HZ: float = 10.0

#: Physiological tremor band (§3.3, §5.4).
TREMOR_BAND_HZ: tuple[float, float] = (8.0, 12.0)


def _min_len_for(order: int) -> int:
    # sosfiltfilt needs the signal longer than the filter's padding length.
    return 3 * (2 * order + 1)


def lowpass(x: np.ndarray, fs: float, cutoff: float = KINEMATIC_CUTOFF_HZ, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0. Passes short signals through unchanged."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] <= _min_len_for(order):
        return x.copy()
    sos = butter(order, cutoff / (0.5 * fs), btype="low", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def highpass(x: np.ndarray, fs: float, cutoff: float = DISP_DRIFT_HZ, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth high-pass along axis 0. Passes short signals through
    unchanged — on a signal too short to resolve the cutoff, high-passing would distort
    more than the drift it removes."""
    x = np.asarray(x, dtype=np.float64)
    # need enough cycles of the cutoff to filter meaningfully (~2 periods) and enough
    # samples for filtfilt padding
    if x.shape[0] <= max(_min_len_for(order), int(2.0 * fs / cutoff)):
        return x.copy()
    sos = butter(order, cutoff / (0.5 * fs), btype="high", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def bandpass(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float] = TREMOR_BAND_HZ,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass along axis 0."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] <= _min_len_for(order):
        return np.zeros_like(x)
    lo, hi = band
    sos = butter(order, [lo / (0.5 * fs), hi / (0.5 * fs)], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=0)
