"""Set-level trajectory summaries (§6.3) — Layer 4.

Per-rep features treated as independent lose the *shape* of degradation, which is the
informative part (§6.3). For each key channel this computes five summaries across the set:

    {f}_slope         OLS slope of f vs rep_index
    {f}_curvature     quadratic coefficient — is degradation accelerating?
    {f}_total_change  (f_last - f_first) / f_first
    {f}_max_dev       largest single-rep deviation from the fitted trend
    {f}_breakpoint    rep index of the best two-segment piecewise-linear fit,
                      plus the fit improvement over a single line

CAUSAL vs RETROSPECTIVE (§6.3) — the leakage boundary
-----------------------------------------------------
The same summary computed over reps 1..r (causal) and over the whole set (retrospective)
are different numbers, and mixing them silently corrupts the RIR head:

* **Causal** summaries are the ONLY ones the per-rep RIR hazard head may use (§9.2). That
  head predicts, at rep r, whether the set is about to fail — so any statistic containing
  information from reps > r leaks the future into a prediction about the future. The
  failure is invisible: no error is raised, the C-index simply comes out inflated.
* **Retrospective** summaries use the whole set and are legitimate for the set score (§8)
  and the RPE head (§10.4), both computed once at set close.

These are deliberately separate functions returning separately-named keys rather than one
function with a flag, so a caller cannot silently obtain the wrong variant.
"""

from __future__ import annotations

import numpy as np

from wristset.features.rep_features import RepFeatures

__all__ = [
    "KEY_CHANNELS",
    "SUMMARY_NAMES",
    "retrospective_summaries",
    "causal_summaries",
    "summarize_channel",
]

#: §6.3: "Apply to at minimum: conc_mean_vel, ecc_duration, rom_vertical,
#: path_dtw_baseline, tremor_power_8_12."
KEY_CHANNELS: tuple[str, ...] = (
    "conc_mean_vel",
    "ecc_duration",
    "rom_vertical",
    "path_dtw_baseline",
    "tremor_power_8_12",
)

SUMMARY_NAMES: tuple[str, ...] = (
    "slope", "curvature", "total_change", "max_dev", "breakpoint", "breakpoint_gain",
)

#: Minimum reps before a summary is meaningful. Slope needs 2, curvature 3, and a
#: two-segment breakpoint needs at least 4 to leave 2 points per segment.
_MIN_FOR_SLOPE = 2
_MIN_FOR_CURVATURE = 3
_MIN_FOR_BREAKPOINT = 4


def summarize_channel(values: np.ndarray) -> dict[str, float | None]:
    """Five §6.3 summaries for one channel's per-rep series.

    Returns ``None`` for summaries the series is too short to support rather than a
    misleading zero — a 2-rep set genuinely has no curvature, and downstream models must
    be able to tell "no degradation" from "not enough reps to say".
    """
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    out: dict[str, float | None] = dict.fromkeys(SUMMARY_NAMES, None)
    n = v.shape[0]
    if n < _MIN_FOR_SLOPE or not np.all(np.isfinite(v)):
        return out

    x = np.arange(n, dtype=np.float64)

    # slope: OLS of f vs rep index
    slope, intercept = np.polyfit(x, v, 1)
    out["slope"] = float(slope)

    # total change, relative to the first rep (§6.3 definition)
    if abs(v[0]) > 1e-12:
        out["total_change"] = float((v[-1] - v[0]) / v[0])

    # max deviation from the fitted linear trend
    out["max_dev"] = float(np.max(np.abs(v - (slope * x + intercept))))

    if n >= _MIN_FOR_CURVATURE:
        out["curvature"] = float(np.polyfit(x, v, 2)[0])

    if n >= _MIN_FOR_BREAKPOINT:
        bp, gain = _best_breakpoint(x, v)
        out["breakpoint"] = bp
        out["breakpoint_gain"] = gain

    return out


def _best_breakpoint(x: np.ndarray, v: np.ndarray) -> tuple[float | None, float | None]:
    """Best two-segment piecewise-linear split, and its improvement over one line (§6.3).

    Returns the 1-based rep index of the split and the fractional SSE reduction. §6.3 notes
    this doubles as the regime diagnostic: breakpoints clustering at the same index across
    channels and sets would justify a switching model, while scatter confirms degradation
    is smooth and the continuous representation is right. The feature is computed either
    way, so the answer is free.
    """
    n = x.shape[0]
    sse_single = _sse(x, v)
    if sse_single <= 0:
        return None, None

    best_sse, best_k = np.inf, None
    # leave >=2 points per segment so each fit is determined
    for k in range(2, n - 1):
        sse = _sse(x[:k], v[:k]) + _sse(x[k:], v[k:])
        if sse < best_sse:
            best_sse, best_k = sse, k
    if best_k is None:
        return None, None
    gain = (sse_single - best_sse) / sse_single
    return float(best_k + 1), float(max(gain, 0.0))  # +1 -> 1-based rep index


def _sse(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 0.0
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.sum((y - (slope * x + intercept)) ** 2))


def _channel_series(reps: list[RepFeatures], channel: str) -> np.ndarray:
    return np.array([getattr(r, channel) for r in reps], dtype=object)


def retrospective_summaries(
    reps: list[RepFeatures],
    channels: tuple[str, ...] = KEY_CHANNELS,
) -> dict[str, float | None]:
    """Whole-set summaries (§6.3). For the set score (§8) and RPE head (§10.4) ONLY.

    Keys are prefixed ``retro_`` so a retrospective value can never be mistaken for a
    causal one downstream. Never feed these to the RIR head (§9.2) — see module docstring.
    """
    out: dict[str, float | None] = {}
    for ch in channels:
        summ = summarize_channel(_channel_series(reps, ch))
        for name, val in summ.items():
            out[f"retro_{ch}_{name}"] = val
    return out


def causal_summaries(
    reps: list[RepFeatures],
    up_to_rep: int,
    channels: tuple[str, ...] = KEY_CHANNELS,
) -> dict[str, float | None]:
    """Summaries using only reps 1..``up_to_rep`` (§6.3). Safe for the RIR head (§9.2).

    ``up_to_rep`` is 1-based and inclusive, matching ``RepFeatures.rep_index``. Keys are
    prefixed ``causal_``. Computing this for every r yields the person-period design matrix
    the §9.3 hazard model expands into.
    """
    prefix = [r for r in reps if r.rep_index <= up_to_rep]
    out: dict[str, float | None] = {}
    for ch in channels:
        summ = summarize_channel(_channel_series(prefix, ch))
        for name, val in summ.items():
            out[f"causal_{ch}_{name}"] = val
    return out
