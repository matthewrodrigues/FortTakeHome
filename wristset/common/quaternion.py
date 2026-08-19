"""Quaternion utilities for rotating between device and world frames.

Convention: quaternions are stored as ``(w, x, y, z)`` (scalar-first), matching
Apple ``CoreMotion`` ``CMQuaternion``. A quaternion ``q`` represents the rotation
that takes a vector expressed in the **device** frame into the **world** frame:

    v_world = q ⊗ v_device ⊗ q*

which is exactly the operation §5.3 of the architecture requires. The inverse
(world → device, used by the synthetic generator to emit device-frame signals) is
performed with the conjugate ``q*``.

All functions are pure and operate on NumPy arrays so they are trivially testable.
Vectorised over a leading time axis: pass ``q`` of shape ``(N, 4)`` and ``v`` of
shape ``(N, 3)`` to rotate a whole timeseries at once.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalize",
    "conjugate",
    "rotate_vectors",
    "rotate_vectors_inverse",
]


def _as2d(a: np.ndarray, width: int, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width}) or ({width},); got {a.shape}")
    return a


def normalize(q: np.ndarray) -> np.ndarray:
    """Return unit quaternion(s). Zero-norm rows fall back to identity ``(1,0,0,0)``."""
    q = _as2d(q, 4, "q")
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    out = np.empty_like(q)
    bad = norm[:, 0] == 0.0
    out[~bad] = q[~bad] / norm[~bad]
    out[bad] = np.array([1.0, 0.0, 0.0, 0.0])
    return out


def conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate ``(w, -x, -y, -z)``. Inverse rotation for a unit quaternion."""
    q = _as2d(q, 4, "q")
    out = q.copy()
    out[:, 1:] *= -1.0
    return out


def _hamilton(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (N,4) scalar-first quaternion arrays."""
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )


def rotate_vectors(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate device-frame vectors into the world frame: ``v_world = q ⊗ v ⊗ q*``.

    ``q`` is normalised internally. Accepts a single quaternion broadcast over many
    vectors, or one quaternion per vector.
    """
    q = normalize(q)
    v = _as2d(v, 3, "v")
    if q.shape[0] == 1 and v.shape[0] != 1:
        q = np.repeat(q, v.shape[0], axis=0)
    if q.shape[0] != v.shape[0]:
        raise ValueError(f"q rows ({q.shape[0]}) must match v rows ({v.shape[0]}) or be 1")
    vq = np.concatenate([np.zeros((v.shape[0], 1)), v], axis=1)  # pure quaternion (0, v)
    return _hamilton(_hamilton(q, vq), conjugate(q))[:, 1:]


def rotate_vectors_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world-frame vectors into the device frame: ``v_device = q* ⊗ v ⊗ q``."""
    return rotate_vectors(conjugate(q), v)
