"""Rotate into a gravity-aligned world frame and split vertical/horizontal (§5.2-5.3).

Without this rotation the model learns watch orientation on the wrist rather than
movement (§5.3). We rotate the device-frame linear acceleration into the world frame
with the attitude quaternion, then define "up" from the gravity direction (reliable,
§5.3) and project acceleration into a vertical scalar and a horizontal 2-vector.

Yaw is unconstrained without a magnetometer (§5.3). The vertical axis needs no yaw.
The horizontal plane here is left in the quaternion's (arbitrary-yaw) world frame;
per-set yaw alignment to the first rep's displacement direction is deferred to the
feature layer that consumes horizontal path (Phase 3), as §5.3 prescribes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wristset.common.quaternion import rotate_vectors
from wristset.conditioning.filters import lowpass

__all__ = ["WorldFrame", "to_world_frame", "GRAVITY_LP_HZ"]

#: Low-pass cutoff used to estimate the steady gravity/up direction (§5.2 fallback also
#: uses ~0.3 Hz to separate gravity).
GRAVITY_LP_HZ: float = 0.3


@dataclass
class WorldFrame:
    """World-frame acceleration with vertical/horizontal decomposition."""

    a_world: np.ndarray  # (N,3) linear accel, gravity-aligned world frame
    a_vert: np.ndarray  # (N,) vertical accel, positive = up
    a_horiz: np.ndarray  # (N,2) horizontal accel in the world XY plane
    up_hat: np.ndarray  # (3,) unit up direction in the world frame
    grav_world: np.ndarray  # (N,3) gravity rotated to world (should be ~[0,0,-g])


def to_world_frame(
    lin_acc_dev: np.ndarray,
    grav_dev: np.ndarray,
    quat: np.ndarray,
    fs: float,
) -> WorldFrame:
    """Rotate device-frame accel to world and decompose into vertical/horizontal.

    ``up_hat`` is derived from the (low-passed) rotated gravity vector, which makes the
    decomposition independent of the exact world-frame convention — it works for any
    platform whose quaternion is gravity-consistent, including the real CoreMotion data.
    """
    a_world = rotate_vectors(quat, lin_acc_dev)
    grav_world = rotate_vectors(quat, grav_dev)

    # steady gravity direction (low-pass to reject motion-induced wobble), §5.2
    grav_lp = lowpass(grav_world, fs, cutoff=GRAVITY_LP_HZ, order=2)
    g_mean = grav_lp.mean(axis=0)
    g_norm = np.linalg.norm(g_mean)
    if g_norm == 0:
        up_hat = np.array([0.0, 0.0, 1.0])
    else:
        up_hat = -g_mean / g_norm  # gravity points down; up is the opposite

    a_vert = a_world @ up_hat  # signed vertical component, + = up
    a_horiz_3d = a_world - np.outer(a_vert, up_hat)

    # express horizontal accel in a 2-D basis orthogonal to up_hat
    e1 = _orthonormal_to(up_hat)
    e2 = np.cross(up_hat, e1)
    a_horiz = np.stack([a_horiz_3d @ e1, a_horiz_3d @ e2], axis=1)

    return WorldFrame(
        a_world=a_world,
        a_vert=a_vert,
        a_horiz=a_horiz,
        up_hat=up_hat,
        grav_world=grav_world,
    )


def _orthonormal_to(u: np.ndarray) -> np.ndarray:
    """Return a unit vector orthogonal to u (stable choice)."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e = ref - (ref @ u) * u
    return e / np.linalg.norm(e)
