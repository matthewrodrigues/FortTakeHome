"""Quaternion rotation correctness — the operation §5.3 depends on."""

from __future__ import annotations

import numpy as np

from wristset.common.quaternion import (
    conjugate,
    normalize,
    rotate_vectors,
    rotate_vectors_inverse,
)


def test_identity_rotation_is_noop():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    v = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    out = rotate_vectors(q, v)
    assert np.allclose(out, v)


def test_forward_then_inverse_recovers_input():
    rng = np.random.default_rng(0)
    q = normalize(rng.standard_normal((5, 4)))
    v = rng.standard_normal((5, 3))
    back = rotate_vectors_inverse(q, rotate_vectors(q, v))
    assert np.allclose(back, v, atol=1e-10)


def test_rotation_preserves_norm():
    rng = np.random.default_rng(1)
    q = normalize(rng.standard_normal((3, 4)))
    v = rng.standard_normal((3, 3))
    out = rotate_vectors(q, v)
    assert np.allclose(np.linalg.norm(out, axis=1), np.linalg.norm(v, axis=1))


def test_90deg_z_rotation():
    # rotation of +90deg about world z maps device-x -> world-y
    q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    out = rotate_vectors(q, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(out, [0.0, 1.0, 0.0], atol=1e-12)


def test_conjugate_of_unit_is_inverse():
    rng = np.random.default_rng(2)
    q = normalize(rng.standard_normal((1, 4)))
    v = rng.standard_normal((1, 3))
    assert np.allclose(rotate_vectors(conjugate(q), rotate_vectors(q, v)), v, atol=1e-10)
