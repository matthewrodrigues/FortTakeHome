"""Rep segmentation from conditioned vertical kinematics (§6.1).

Three-stage approach, cheapest first (§6.1):

    1. Vertical displacement minima (rep bottoms) bracketed by top lockouts, validated
       by vertical-velocity sign changes. Handles the majority of clean sets.
    2. Autocorrelation period estimate to set a minimum inter-bottom spacing, rejecting
       spurious double-detections.
    3. DTW template matching (see ``dtw.py``) as a fallback for degraded late reps where
       velocity zero-crossings get noisy.

A detected bottom is a *completed* rep only if the movement recovers back near the top
baseline afterwards; a failed final attempt descends but stalls partway (§4.2) and is
recorded with ``completed=False`` so it does not inflate the completed count that must
match ``reported_reps``.

Phase boundaries (§6.1): ``t_start`` (top, begin eccentric) -> ``t_bottom`` (velocity
sign change) -> ``t_end`` (top, end concentric). Deadlift inverts this order (concentric
first); the per-exercise seam is present but only bench/squat are active (§1.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from wristset.conditioning import ConditionedSet
from wristset.segmentation.activity import ActiveWindow, detect_active_window

__all__ = ["RepBoundary", "SegmentationResult", "segment_reps", "RECOVERY_THRESHOLD"]

#: Fraction of descent depth a rep must recover to count as completed (vs failed attempt).
RECOVERY_THRESHOLD: float = 0.6

#: Physiological rep-period band (§3.3: rep fundamental 0.3-1 Hz). The autocorrelation
#: estimate is confined here; anything outside is an envelope artifact, not a rep cadence.
MIN_PERIOD_S: float = 0.3
MAX_PERIOD_S: float = 10.0

#: Max ratio between the autocorrelation and inter-bottom period estimates before the
#: local (inter-bottom) estimate is preferred. 2.0 tolerates half/double-period lock-on.
PERIOD_AGREEMENT: float = 2.0

#: Concentric-first exercises invert the eccentric/concentric phase order (§6.1).
_CONCENTRIC_FIRST = {"deadlift"}


@dataclass
class RepBoundary:
    """One detected rep with phase boundaries and completion status."""

    rep_index: int  # 1-based
    i_start: int
    i_bottom: int
    i_end: int
    t_start: float
    t_bottom: float
    t_end: float
    completed: bool
    rom_vertical: float  # reconstructed peak-to-peak vertical displacement (m)
    recovery: float  # fraction of descent depth recovered afterwards


@dataclass
class SegmentationResult:
    reps: list[RepBoundary] = field(default_factory=list)
    period_s: float | None = None
    method: str = "displacement_minima"
    active_window: "ActiveWindow | None" = None  # §6.1 set-detection validation signal

    @property
    def active_fraction(self) -> float:
        """Fraction of the recording that is active lifting. <1 means the recording
        carried non-lifting time (setup/racking/rest) outside the working set."""
        return 1.0 if self.active_window is None else self.active_window.active_fraction

    @property
    def n_completed(self) -> int:
        return sum(1 for r in self.reps if r.completed)

    @property
    def n_detected(self) -> int:
        return len(self.reps)


def _estimate_period_samples(disp: np.ndarray, fs: float) -> float | None:
    """First prominent autocorrelation peak of the (detrended) displacement (§6.1).

    Bounded to the physiological rep band. Autocorrelation is a *global* view, so on a
    long recording containing rest or several sets it can lock onto the set-rest envelope
    instead of the rep cadence — observed at 57-101 s on the real corpus against a ~4.5 s
    median. An unbounded estimate is catastrophic downstream: ``min_dist = 0.5 * period``
    then suppresses every rep but one. Returning None is strictly safer, since the caller
    falls back to a fixed minimum spacing.
    """
    x = disp - disp.mean()
    if np.allclose(x, 0):
        return None
    ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    lo = int(MIN_PERIOD_S * fs)
    hi = min(int(MAX_PERIOD_S * fs), len(ac) - 2)
    if lo >= hi:
        return None
    # search only within the rep band, so an out-of-band envelope peak is never selected
    peaks, _ = find_peaks(ac[lo : hi + 1], height=0.2)
    if peaks.size == 0:
        return None
    return float(peaks[0] + lo)


def _refine_period(period: float | None, bottoms: np.ndarray, fs: float) -> float | None:
    """Cross-check the autocorrelation period against observed inter-bottom spacing.

    The two estimators fail independently: autocorrelation is global and frequency-domain
    (fooled by set-rest envelopes), while inter-bottom spacing is local and time-domain
    (fooled only by spurious peaks, and then boundedly). When they disagree by more than
    ``PERIOD_AGREEMENT`` the local estimate is preferred — its failure mode is a modest
    spacing error, whereas the global one's is total set suppression.
    """
    if bottoms.size < 3:
        return period
    observed = float(np.median(np.diff(bottoms)))
    if observed <= 0:
        return period
    if period is None:
        return observed if MIN_PERIOD_S * fs <= observed <= MAX_PERIOD_S * fs else None
    ratio = max(period, observed) / min(period, observed)
    return period if ratio <= PERIOD_AGREEMENT else observed


def segment_reps(
    cs: ConditionedSet,
    *,
    exercise: str = "bench_press",
    reached_failure: bool = False,
    expected_reps: int | None = None,
    trim: bool = True,
) -> SegmentationResult:
    """Segment reps from a ConditionedSet.

    ``reached_failure`` is the user-reported metadata flag (§3.5), NOT the rep count. When
    true, exactly one failed attempt ends the set; it is excluded from the completed count
    via the recovery test, with a fallback that demotes the weakest trailing excursion if
    the recovery test alone lets a shallow failed attempt through under heavy fatigue.
    ``expected_reps`` is accepted for the DTW fallback trigger but never used to fabricate
    or force the count.

    ``trim`` applies §6.1 set detection: segmentation is restricted to the detected active
    window so setup/racking/rest motion outside the working set is not counted as reps. On
    a clean capture the window is the whole recording and this is a no-op; the detected
    ``active_fraction`` is then the §6.1 validation signal. Emitted indices and times
    always refer to the FULL conditioned arrays, so downstream layers need no offset.
    """
    # disp_vert is already drift-corrected (high-passed) in conditioning, so rep minima
    # and the return-to-lockout recovery test are both stable references here.
    fs = cs.fs
    if cs.disp_vert.shape[0] < 3:
        return SegmentationResult()

    window = detect_active_window(cs) if trim else None
    off = window.i0 if window else 0
    disp = cs.disp_vert[off : window.i1 + 1] if window else cs.disp_vert
    t = cs.t[off : window.i1 + 1] if window else cs.t
    n = disp.shape[0]
    if n < 3:
        # active window too short to segment; fall back to the full recording
        window, off = None, 0
        disp, t, n = cs.disp_vert, cs.t, cs.disp_vert.shape[0]

    period = _estimate_period_samples(disp, fs)

    span = float(np.percentile(disp, 95) - np.percentile(disp, 5))
    if span <= 0:
        return SegmentationResult(period_s=None, active_window=window)

    prominence = 0.2 * span

    def _find_bottoms(p: float | None) -> np.ndarray:
        min_dist = int(0.5 * p) if p else int(0.4 * fs)
        # rep bottoms = prominent minima of the (drift-free) vertical displacement
        found, _ = find_peaks(-disp, prominence=prominence, distance=max(min_dist, 1))
        return found

    # Two-pass: detect with the autocorrelation period, cross-check it against the
    # spacing actually observed, and re-detect if the refined period disagrees (§6.1).
    bottoms = _find_bottoms(period)
    refined = _refine_period(period, bottoms, fs)
    if refined is not None and (period is None or abs(refined - period) > 1e-6):
        period = refined
        bottoms = _find_bottoms(period)

    if bottoms.size == 0:
        return SegmentationResult(period_s=_samples_to_s(period, fs), active_window=window)

    reps: list[RepBoundary] = []
    concentric_first = exercise in _CONCENTRIC_FIRST
    for k, b in enumerate(bottoms):
        left = bottoms[k - 1] if k > 0 else 0
        right = bottoms[k + 1] if k + 1 < bottoms.size else n - 1
        i_start = left + int(np.argmax(disp[left : b + 1]))
        i_end = b + int(np.argmax(disp[b : right + 1]))

        top_before = disp[i_start]
        top_after = disp[i_end]
        bottom_val = disp[b]
        depth = top_before - bottom_val
        recovery = (top_after - bottom_val) / depth if depth > 1e-9 else 0.0
        completed = bool(recovery >= RECOVERY_THRESHOLD)
        rom = float(max(top_before, top_after) - bottom_val)

        t_start, t_bottom, t_end = t[i_start], t[b], t[i_end]
        if concentric_first:
            # deadlift: bar starts at the bottom; swap the phase semantics
            t_start, t_end = t_end, t_start
            i_start, i_end = i_end, i_start

        # emitted indices are re-based onto the FULL conditioned arrays (see docstring)
        reps.append(
            RepBoundary(
                rep_index=k + 1,
                i_start=int(i_start) + off,
                i_bottom=int(b) + off,
                i_end=int(i_end) + off,
                t_start=float(t_start),
                t_bottom=float(t_bottom),
                t_end=float(t_end),
                completed=completed,
                rom_vertical=rom,
                recovery=float(recovery),
            )
        )

    # Failed-attempt reconciliation (§3.5, §4.2). A failed attempt always ends the set.
    # If the user reported reaching failure but every detected excursion passed the
    # recovery test, the shallow failed attempt slipped through under heavy fatigue —
    # demote the weakest-recovery excursion, which is the attempt.
    if reached_failure and reps and all(r.completed for r in reps):
        weakest = min(range(len(reps)), key=lambda i: reps[i].recovery)
        r = reps[weakest]
        reps[weakest] = RepBoundary(
            rep_index=r.rep_index, i_start=r.i_start, i_bottom=r.i_bottom, i_end=r.i_end,
            t_start=r.t_start, t_bottom=r.t_bottom, t_end=r.t_end,
            completed=False, rom_vertical=r.rom_vertical, recovery=r.recovery,
        )

    return SegmentationResult(
        reps=reps,
        period_s=_samples_to_s(period, fs),
        method="displacement_minima",
        active_window=window,
    )


def _samples_to_s(samples: float | None, fs: float) -> float | None:
    return None if samples is None else samples / fs
