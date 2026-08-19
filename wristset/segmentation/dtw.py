"""DTW template matching — segmentation stage 3 (§6.1).

The velocity-zero-crossing + recovery segmenter (``reps.py``) handles clean through
moderately-degraded sets. For heavily degraded late reps, where velocity zero-crossings
get noisy, §6.1 prescribes DTW template matching: build a template from the set's first
few clean reps (low fatigue) and score each candidate rep by its DTW distance to it. A
completed rep matches the template shape; a failed attempt (truncated concentric) is far
from it.

This module provides the DTW primitive and template scoring. It is a self-contained
NumPy implementation (no extra dependency) and is exposed as a shape-based signal /
fallback; the primary path already clears the segmentation gate on the operating
distribution, so this is used to harden the degraded regime and as a per-rep
form-consistency feature later (``path_dtw_baseline``, §6.2).
"""

from __future__ import annotations

import numpy as np

from wristset.conditioning import ConditionedSet
from wristset.segmentation.reps import RepBoundary

__all__ = ["dtw_distance", "resample_curve", "rep_velocity_profile", "build_template", "template_distance"]

#: Length every rep profile is resampled to before DTW (duration-invariant comparison).
PROFILE_LEN: int = 64


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Classic dynamic-time-warping distance between two 1-D sequences (Euclidean local
    cost, full DP). Returned value is normalized by the warping-path length so it is
    comparable across differently-sized inputs."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("inf")
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def resample_curve(x: np.ndarray, length: int = PROFILE_LEN) -> np.ndarray:
    """Linearly resample a 1-D curve to a fixed length (duration normalization)."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return np.zeros(length)
    src = np.linspace(0.0, 1.0, x.shape[0])
    dst = np.linspace(0.0, 1.0, length)
    return np.interp(dst, src, x)


def rep_velocity_profile(cs: ConditionedSet, rep: RepBoundary, length: int = PROFILE_LEN) -> np.ndarray:
    """Vertical-velocity profile of one rep, resampled to a common length and scaled to
    unit peak so template matching compares SHAPE, not magnitude."""
    i0, i1 = sorted((rep.i_start, rep.i_end))
    seg = cs.v_vert[i0 : i1 + 1]
    prof = resample_curve(seg, length)
    peak = np.max(np.abs(prof))
    return prof / peak if peak > 1e-9 else prof


def build_template(cs: ConditionedSet, reps: list[RepBoundary], n_ref: int = 3) -> np.ndarray | None:
    """Average velocity-profile template from the first ``n_ref`` completed reps (§7 clean
    reference: low-fatigue, most likely genuine form). Returns None if too few reps."""
    completed = [r for r in reps if r.completed]
    if len(completed) < 2:
        return None
    profs = [rep_velocity_profile(cs, r) for r in completed[:n_ref]]
    return np.mean(profs, axis=0)


def template_distance(cs: ConditionedSet, rep: RepBoundary, template: np.ndarray) -> float:
    """DTW distance of one rep's velocity profile to the template."""
    return dtw_distance(rep_velocity_profile(cs, rep), template)
