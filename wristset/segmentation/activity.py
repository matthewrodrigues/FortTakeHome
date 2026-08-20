"""Set detection — active-window isolation (§6.1).

§6.1: "Rolling-window energy threshold on ``|a_linear|`` distinguishes active from rest.
For the prototype, the user explicitly starts/stops the set, so this is a validation
check rather than a detection problem."

Both roles are served here. When capture supplies clean boundaries the detected window
spans essentially the whole recording and trimming is a no-op — the useful output is then
the *validation* signal ``active_fraction``. When a recording carries ragged edges (setup,
racking, rest before/after the working set), trimming restricts segmentation to the
lifting window so non-lifting motion is not counted as reps.

The real-corpus sweep (2026-08-19) measured ~25% non-lifting time inside recordings, with
segmentation tracking recording duration (r=0.84) more strongly than true rep count
(r=0.43) — the failure mode this module addresses.

Thresholding is *relative to the set's own energy distribution*, never an absolute m/s^2
value: an absolute threshold would need retuning per exercise, load, and user, whereas a
set is defined by contrast with its own rest periods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wristset.conditioning import ConditionedSet

__all__ = ["ActiveWindow", "detect_active_window", "ENERGY_WINDOW_S", "ACTIVE_QUANTILE"]

#: Rolling RMS window (§6.1). ~1 s spans a rep phase without smearing set boundaries.
ENERGY_WINDOW_S: float = 1.0

#: Energy threshold as a quantile-interpolated level between the set's quiet floor and its
#: active level. 0.35 sits below sustained lifting but above rest-period jitter.
ACTIVE_QUANTILE: float = 0.35

#: Gaps in activity shorter than this are bridged — between-rep pauses and lockouts are
#: genuinely low-energy and must not split one set into fragments. Sized against measured
#: real inter-rep spacing (~4.4 s median, real corpus) with headroom: at 2.5 s real sets
#: fragmented into a median of 2 runs (up to 11) and "longest run" selected one fragment,
#: capturing only ~30% of the recording. Rest *between sets* is far longer than this, so
#: 6 s bridges within-set pauses without merging separate sets.
BRIDGE_GAP_S: float = 6.0

#: An active run shorter than this cannot be a working set; discarded as incidental motion.
MIN_ACTIVE_S: float = 5.0

#: Refuse to trim below this fraction of the recording. Trimming away most of a recording
#: on energy evidence alone loses reps outright; keeping too much only risks a mild
#: over-count that downstream tests already filter. Asymmetric costs, asymmetric guard.
MIN_KEEP_FRACTION: float = 0.5

#: Padding added to each end of the detected window. The first and last reps END in a top
#: lockout, which is by definition low-energy, so the energy threshold marks the last
#: *movement* rather than the true set end. Without padding the trailing lockout is clipped
#: and the recovery test — which needs the return-to-top — demotes a completed final rep.
EDGE_PAD_S: float = 1.5


@dataclass
class ActiveWindow:
    """Detected active-lifting window within a recording."""

    i0: int  # inclusive start sample
    i1: int  # inclusive end sample
    t0: float
    t1: float
    active_fraction: float  # detected active span / total recording duration
    n_candidate_runs: int  # active runs found before selecting the longest

    @property
    def is_full_recording(self) -> bool:
        """True when the window spans essentially the whole recording (clean capture)."""
        return self.active_fraction >= 0.95


def _rolling_rms(x: np.ndarray, win: int) -> np.ndarray:
    """Centered rolling RMS via a cumulative-sum box filter (O(n), edge-padded)."""
    if win < 1:
        win = 1
    pad = win // 2
    xp = np.pad(x**2, (pad, pad), mode="edge")
    c = np.cumsum(np.insert(xp, 0, 0.0))
    out = (c[win:] - c[:-win]) / win
    return np.sqrt(np.maximum(out[: x.shape[0]], 0.0))


def detect_active_window(cs: ConditionedSet) -> ActiveWindow:
    """Find the active-lifting window of a conditioned set (§6.1).

    Returns the longest contiguous active run after bridging short intra-set pauses. On a
    clean capture this is the whole recording (``is_full_recording``), making the call a
    no-op for trimming and a validation signal via ``active_fraction``.
    """
    fs = cs.fs
    n = cs.t.shape[0]
    full = ActiveWindow(0, max(n - 1, 0), 0.0, float(cs.t[-1]) if n else 0.0, 1.0, 1)
    if n < 3:
        return full

    energy = _rolling_rms(np.linalg.norm(cs.a_world, axis=1), int(ENERGY_WINDOW_S * fs))

    # Relative threshold: interpolate between the set's own quiet floor and active level.
    lo, hi = np.percentile(energy, 10), np.percentile(energy, 90)
    if hi - lo < 1e-9:
        return full  # uniform energy -> no rest to distinguish; treat as fully active
    thresh = lo + ACTIVE_QUANTILE * (hi - lo)
    active = energy >= thresh
    if not active.any():
        return full

    runs = _contiguous_runs(active)
    runs = _bridge(runs, int(BRIDGE_GAP_S * fs))
    min_len = int(MIN_ACTIVE_S * fs)
    long_runs = [r for r in runs if (r[1] - r[0]) >= min_len] or runs

    i0, i1 = max(long_runs, key=lambda r: r[1] - r[0])

    # extend past the last detected movement to retain the bounding lockouts (EDGE_PAD_S)
    pad = int(EDGE_PAD_S * fs)
    i0 = max(0, i0 - pad)
    i1 = min(n - 1, i1 + pad)
    frac = (i1 - i0 + 1) / n

    # Conservative guard. Energy alone cannot always tell lifting from vigorous
    # non-lifting motion (racking, walking) — the latter can be *louder* than a controlled
    # rep, in which case the longest above-threshold run is a padding fragment rather than
    # the set. Discarding most of a recording on that evidence is the costly error
    # (reps are lost outright), whereas keeping too much only risks a modest over-count
    # that the prominence and recovery tests already filter. So below MIN_KEEP_FRACTION we
    # decline to trim and report the full recording, leaving active_fraction as the
    # validation signal that something about the boundaries was unusual (§6.1).
    if frac < MIN_KEEP_FRACTION:
        return ActiveWindow(0, n - 1, float(cs.t[0]), float(cs.t[-1]), float(frac), len(long_runs))

    return ActiveWindow(
        i0=int(i0),
        i1=int(i1),
        t0=float(cs.t[i0]),
        t1=float(cs.t[i1]),
        active_fraction=float(frac),
        n_candidate_runs=len(long_runs),
    )


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs for each contiguous True run."""
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1))
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.shape[0] - 1)
    return list(zip(starts, ends))


def _bridge(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    """Merge runs separated by less than ``max_gap`` samples.

    Between-rep pauses and top lockouts are genuinely low-energy; without bridging, one
    set fragments into one run per rep and the "longest run" becomes a single rep.
    """
    if not runs:
        return []
    merged = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        if s - pe <= max_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged
