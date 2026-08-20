"""Per-rep feature extraction (§6.2) — Layer 4.

Computed for every detected rep. This is the core representation every downstream layer
consumes: form subscores (§8.3), the RIR hazard head (§9.2), and the RPE head (§10.4).

Four families, per §6.2:

    kinematic  : concentric/eccentric velocity, ROM, sticking point
    tempo      : phase durations, bottom pause, tempo ratio
    path       : arc length, efficiency, horizontal excursion, DTW-to-baseline
    smoothness : jerk RMS, spectral arc length, 8-12 Hz tremor power

Every feature is derived from the ConditionedSet's world-frame signals, so all of them
inherit the §5.3 caveat that horizontal channels are in an arbitrary-yaw frame (see §5.3
implementation note). Features keyed only on magnitude are unaffected; ``horiz_excursion``
and ``path_length`` are magnitude-based and therefore safe.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from wristset.conditioning import ConditionedSet
from wristset.segmentation import RepBoundary

__all__ = ["RepFeatures", "extract_rep_features", "FEATURE_NAMES"]


@dataclass
class RepFeatures:
    """The §6.2 per-rep feature vector, plus the metadata §6.2 attaches to each rep."""

    # --- identity / metadata (§6.2 "Metadata attached to each rep") ---
    rep_index: int
    reps_remaining_in_set: int  # known post hoc; retrospective-only (leakage risk, §6.3)
    completed: bool
    load_kg: float
    exercise: str
    set_type: str
    load_pct_e1rm: float | None = None

    # --- kinematic ---
    conc_mean_vel: float = 0.0
    conc_peak_vel: float = 0.0
    ecc_mean_vel: float = 0.0
    rom_vertical: float = 0.0
    #: Sticking point (§6.2). ``None`` when the concentric has no interior slowdown — a
    #: monotonic ascent genuinely has no sticking point, and 0.0 would misread as "the bar
    #: stopped dead". Both are None together.
    min_vel_position: float | None = None  # normalized ROM position of the velocity minimum
    min_vel_value: float | None = None  # velocity at that minimum

    # --- tempo ---
    ecc_duration: float = 0.0
    conc_duration: float = 0.0
    bottom_pause: float = 0.0
    tempo_ratio: float = 0.0

    # --- path ---
    path_length: float = 0.0
    path_efficiency: float = 0.0
    horiz_excursion: float = 0.0
    path_dtw_baseline: float | None = None  # filled by the baseline layer (§7)

    # --- smoothness / tremor ---
    jerk_rms: float = 0.0
    spectral_arc_length: float = 0.0
    tremor_power_8_12: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


#: §6.2 features that are ALWAYS a finite float. Excludes metadata and the three
#: optional channels, which are ``None`` when undefined rather than a misleading zero:
#: ``path_dtw_baseline`` (no §7 template yet) and the two sticking-point features (no
#: interior slowdown in the ascent). Callers iterating features for a model matrix should
#: use this tuple plus explicit handling of :data:`OPTIONAL_FEATURE_NAMES`.
FEATURE_NAMES: tuple[str, ...] = (
    "conc_mean_vel", "conc_peak_vel", "ecc_mean_vel", "rom_vertical",
    "ecc_duration", "conc_duration", "bottom_pause", "tempo_ratio",
    "path_length", "path_efficiency", "horiz_excursion",
    "jerk_rms", "spectral_arc_length", "tremor_power_8_12",
)

#: §6.2 features that may legitimately be ``None``. See :data:`FEATURE_NAMES`.
OPTIONAL_FEATURE_NAMES: tuple[str, ...] = (
    "min_vel_position", "min_vel_value", "path_dtw_baseline",
)


def _spectral_arc_length(v: np.ndarray, fs: float) -> float:
    """SPARC smoothness metric on a velocity profile (§6.2).

    Arc length of the normalized magnitude spectrum. More negative = less smooth (more
    spectral content = more corrections/tremor). Duration-robust by construction, which is
    why §6.2 prefers it over raw jerk for cross-rep comparison: a slow late rep is not
    penalised for being slow, only for being *rougher*.
    """
    if v.shape[0] < 8:
        return 0.0
    # zero-pad to a fixed power of two for a stable frequency grid across differing
    # rep durations, then take the single-sided normalized magnitude spectrum
    nfft = max(int(2 ** np.ceil(np.log2(v.shape[0]))) * 4, 64)
    mag = np.abs(np.fft.rfft(v - v.mean(), n=nfft))
    if mag.max() <= 0:
        return 0.0
    mag = mag / mag.max()
    freq = np.fft.rfftfreq(nfft, d=1.0 / fs)

    # restrict to the band carrying movement content (SPARC convention: cut at the point
    # the spectrum falls below a small amplitude threshold, capped at 20 Hz)
    cutoff = 20.0
    keep = freq <= cutoff
    f, m = freq[keep], mag[keep]
    if f.size < 3:
        return 0.0
    df = (f[-1] - f[0]) or 1.0
    return float(-np.sum(np.sqrt((np.diff(f) / df) ** 2 + np.diff(m) ** 2)))


def extract_rep_features(
    cs: ConditionedSet,
    rep: RepBoundary,
    *,
    n_reps_in_set: int,
    load_kg: float = 0.0,
    exercise: str = "bench_press",
    set_type: str = "working",
    load_pct_e1rm: float | None = None,
) -> RepFeatures:
    """Extract the §6.2 feature vector for one detected rep.

    Phase slicing uses the segmenter's boundaries: ``i_start -> i_bottom`` is the eccentric
    and ``i_bottom -> i_end`` the concentric (inverted for concentric-first exercises, which
    the segmenter already encodes by swapping the indices).
    """
    fs = cs.fs
    i0, ib, i1 = rep.i_start, rep.i_bottom, rep.i_end
    lo, hi = min(i0, i1), max(i0, i1)

    v = cs.v_vert
    d = cs.disp_vert

    ecc = slice(min(i0, ib), max(i0, ib) + 1)
    conc = slice(min(ib, i1), max(ib, i1) + 1)

    v_conc = v[conc]
    v_ecc = v[ecc]

    conc_mean_vel = float(np.mean(np.abs(v_conc))) if v_conc.size else 0.0
    conc_peak_vel = float(np.max(np.abs(v_conc))) if v_conc.size else 0.0
    ecc_mean_vel = float(np.mean(np.abs(v_ecc))) if v_ecc.size else 0.0

    seg_d = d[lo : hi + 1]
    rom_vertical = float(seg_d.max() - seg_d.min()) if seg_d.size else 0.0

    # sticking point: slowest moment of the concentric, as a normalized ROM position
    min_vel_position, min_vel_value = _sticking_point(v_conc, d[conc], rom_vertical)

    # --- tempo ---
    ecc_duration = abs(rep.t_bottom - rep.t_start)
    conc_duration = abs(rep.t_end - rep.t_bottom)
    tempo_ratio = ecc_duration / conc_duration if conc_duration > 1e-9 else 0.0
    bottom_pause = _bottom_pause(cs, ib)

    # --- path ---
    path_length, horiz_excursion = _path_metrics(cs, lo, hi)
    path_efficiency = path_length / rom_vertical if rom_vertical > 1e-9 else 0.0

    # --- smoothness ---
    a_seg = cs.a_vert[lo : hi + 1]
    jerk = np.diff(a_seg) * fs if a_seg.size > 1 else np.zeros(1)
    jerk_rms = float(np.sqrt(np.mean(jerk**2)))
    sparc = _spectral_arc_length(v[lo : hi + 1], fs)
    tremor_power = float(np.mean(cs.tremor[lo : hi + 1] ** 2)) if hi > lo else 0.0

    return RepFeatures(
        rep_index=rep.rep_index,
        reps_remaining_in_set=max(n_reps_in_set - rep.rep_index, 0),
        completed=rep.completed,
        load_kg=load_kg,
        exercise=exercise,
        set_type=set_type,
        load_pct_e1rm=load_pct_e1rm,
        conc_mean_vel=conc_mean_vel,
        conc_peak_vel=conc_peak_vel,
        ecc_mean_vel=ecc_mean_vel,
        rom_vertical=rom_vertical,
        min_vel_position=min_vel_position,
        min_vel_value=min_vel_value,
        ecc_duration=float(ecc_duration),
        conc_duration=float(conc_duration),
        bottom_pause=bottom_pause,
        tempo_ratio=float(tempo_ratio),
        path_length=path_length,
        path_efficiency=path_efficiency,
        horiz_excursion=horiz_excursion,
        jerk_rms=jerk_rms,
        spectral_arc_length=sparc,
        tremor_power_8_12=tremor_power,
    )


#: Fraction of the concentric's peak speed above which the lift counts as "moving".
#: The sticking point is the slowest moment of genuine ascent — NOT the bottom pause,
#: which is a long zero-velocity plateau inside the phase (measured: 56 of 134 concentric
#: samples below 0.05 m/s on a clean rep). Searching the raw phase minimum therefore
#: returns the pause every time, at position ~0 with value ~0.
_MOVING_FRACTION: float = 0.10

#: Fraction of the moving span trimmed from each end before searching for the minimum.
#: A concentric ramps up and back down, so velocity is legitimately lowest at both ends of
#: the movement; without a margin the "minimum" is always an endpoint and its position is
#: arbitrary. A sticking point is a slowdown *between* acceleration and lockout.
#:
#: NOTE — ``min_vel_position`` is not validated on current synthetic data. The generator
#: models each concentric as one smooth half-cosine, which by construction has no interior
#: slowdown, so the position degenerates to noise even with this margin. ``min_vel_value``
#: is meaningful (it tracks fatigue correctly). Revisit when the generator models a
#: genuine sticking region or real bench/squat data lands.
_STICKING_MARGIN: float = 0.20

#: Minimum speed, as a fraction of the concentric peak, for a dip to count as a sticking
#: point rather than a direction reversal. A grind slows the bar; it does not stop it.
_MIN_STICKING_SPEED: float = 0.02


def _sticking_point(
    v_conc: np.ndarray, d_conc: np.ndarray, rom: float
) -> tuple[float | None, float | None]:
    """Locate the concentric velocity minimum as a normalized ROM position (§6.2).

    Position is normalized by the rep's OWN ROM so it stays comparable as ROM collapses
    with fatigue: an absolute height would make the sticking point appear to descend
    through the set when in fact the whole lift got shorter. 0 = bottom, 1 = top.

    Returns ``(None, None)`` when the ascent has no genuine sticking point. A concentric
    with no interior slowdown is a monotonic arch — speed rises, peaks, falls — and
    ``argmin`` over it just returns whichever search-window edge is lower, a value with no
    physical meaning. ``None`` says "no sticking point in this rep", which is distinct from
    a 0.0 that would read as "the bar stopped dead". Callers must handle both.
    """
    if v_conc.size < 5 or d_conc.size != v_conc.size or rom <= 1e-9:
        return None, None
    speeds = np.abs(v_conc)
    peak = float(speeds.max())
    if peak <= 1e-9:
        return None, None

    moving = np.flatnonzero(speeds >= _MOVING_FRACTION * peak)
    if moving.size < 3:
        return None, None
    lo, hi = int(moving[0]), int(moving[-1])
    span = hi - lo
    if span < 5:
        return None, None

    margin = max(int(_STICKING_MARGIN * span), 1)
    a, b = lo + margin, hi - margin
    if b - a < 3:
        return None, None

    window = speeds[a:b]
    j = int(np.argmin(window))
    # Degenerate unless the minimum is a genuine INTERIOR dip: not pinned to either edge
    # of the search window, and strictly lower than both endpoints. A monotonic arch (the
    # current generator's half-cosine concentric) fails both tests, as it should.
    if j == 0 or j == window.size - 1:
        return None, None
    if not (window[j] < window[0] and window[j] < window[-1]):
        return None, None
    # A sticking point is a SLOWDOWN, not a stop. A near-zero minimum inside the span means
    # the window crossed a direction reversal (one rep's lockout running into the next) —
    # a segmentation boundary artifact rather than a grind. Measured on late fatigued reps:
    # 1-2 exactly-zero samples inside an otherwise-moving span.
    if window[j] < _MIN_STICKING_SPEED * peak:
        return None, None

    k = j + a
    base = float(np.min(d_conc))
    pos = (float(d_conc[k]) - base) / rom
    return float(np.clip(pos, 0.0, 1.0)), float(speeds[k])


def _bottom_pause(cs: ConditionedSet, i_bottom: int) -> float:
    """Duration of the stationary interval containing (or nearest) the rep bottom.

    Reads the conditioning layer's stationary mask rather than re-thresholding, so the
    definition of "stationary" stays single-sourced with ZUPT (§5.5).
    """
    mask = cs.stationary
    n = mask.shape[0]
    if not (0 <= i_bottom < n) or not mask[i_bottom]:
        return 0.0
    a = i_bottom
    while a > 0 and mask[a - 1]:
        a -= 1
    b = i_bottom
    while b + 1 < n and mask[b + 1]:
        b += 1
    return float((b - a + 1) / cs.fs)


def _path_metrics(cs: ConditionedSet, lo: int, hi: int) -> tuple[float, float]:
    """3-D arc length and max horizontal deviation over the rep (§6.2 path table).

    Horizontal deviation is measured from the vertical line through the rep's *start*
    position, per §6.2. Both are magnitude quantities, so neither depends on the
    unresolved yaw convention (§5.3).
    """
    if hi <= lo:
        return 0.0, 0.0
    z = cs.disp_vert[lo : hi + 1]
    h = cs.disp_horiz[lo : hi + 1]
    pts = np.column_stack([h, z])  # (N,3): horizontal x, horizontal y, vertical
    path_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    horiz_excursion = float(np.max(np.linalg.norm(h - h[0], axis=1)))
    return path_length, horiz_excursion
