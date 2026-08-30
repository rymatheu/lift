"""Tests for the geometry helpers that were deduplicated out of the scripts.

Run with: pytest
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lift.geometry.metrics import cosine_similarity, rescale_embeddings
from lift.geometry.pca import pca
from lift.geometry.transforms import find_rigid_transform, rod_rot, skew


def test_cosine_similarity_1d_returns_scalar():
    a = jnp.array([1.0, 0.0, 0.0])
    assert cosine_similarity(a, a).shape == ()
    assert cosine_similarity(a, a) == pytest.approx(1.0)
    assert cosine_similarity(a, jnp.array([0.0, 1.0, 0.0])) == pytest.approx(0.0)
    assert cosine_similarity(a, -a) == pytest.approx(-1.0)


def test_cosine_similarity_2d_is_rowwise():
    A = jnp.array([[1.0, 0.0], [0.0, 2.0]])
    B = jnp.array([[3.0, 0.0], [0.0, -1.0]])
    out = cosine_similarity(A, B)
    assert out.shape == (2,)
    np.testing.assert_allclose(out, [1.0, -1.0], atol=1e-6)


def test_cosine_similarity_handles_zero_vectors():
    zero = jnp.zeros(3)
    assert jnp.isfinite(cosine_similarity(zero, jnp.array([1.0, 2.0, 3.0])))


def test_rescale_sets_row_lengths():
    E = jnp.array([[3.0, 4.0], [1.0, 0.0]])
    scales = jnp.array([1.0, 7.0])
    out = rescale_embeddings(E, scales)
    np.testing.assert_allclose(jnp.linalg.norm(out, axis=-1), scales, atol=1e-6)


def test_rescale_preserves_direction():
    """The invariant the old main.py analysis() was eyeballing by hand."""
    rng = np.random.default_rng(0)
    E = jnp.array(rng.normal(size=(16, 8)))
    scales = jnp.array(rng.uniform(0.5, 5.0, size=16))
    out = rescale_embeddings(E, scales)
    np.testing.assert_allclose(cosine_similarity(E, out), np.ones(16), atol=1e-6)


def test_rescale_leaves_zero_rows_alone():
    E = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    out = rescale_embeddings(E, jnp.array([2.0, 2.0]))
    assert jnp.all(jnp.isfinite(out))
    np.testing.assert_allclose(out[0], [0.0, 0.0])


def test_pca_recovers_a_planar_subspace():
    rng = np.random.default_rng(1)
    # Points on a plane embedded in 5D: the third component carries no variance.
    coeffs = rng.normal(size=(200, 2))
    basis = rng.normal(size=(2, 5))
    X = coeffs @ basis

    proj, var = pca(X, 3)
    assert proj.shape == (200, 3)
    assert var[0] + var[1] == pytest.approx(1.0, abs=1e-5)
    assert var[2] == pytest.approx(0.0, abs=1e-6)


def test_pca_is_centred():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 6)) + 10.0
    proj, _ = pca(X, 3)
    np.testing.assert_allclose(proj.mean(axis=0), np.zeros(3), atol=1e-4)


def test_skew_is_antisymmetric():
    v = jnp.array([1.0, 2.0, 3.0])
    S = skew(v)
    np.testing.assert_allclose(S, -S.T, atol=1e-6)
    # skew(v) @ w == cross(v, w)
    w = jnp.array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(S @ w, np.cross(np.asarray(v), np.asarray(w)), atol=1e-5)


def test_rod_rot_is_a_rotation():
    axis = jnp.array([0.0, 0.0, 1.0])
    R = rod_rot(axis, jnp.pi / 2)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert float(jnp.linalg.det(R)) == pytest.approx(1.0, abs=1e-6)
    # A quarter turn about z sends x to y.
    np.testing.assert_allclose(R @ jnp.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6)


def test_find_rigid_transform_recovers_a_known_motion():
    rng = np.random.default_rng(3)
    P = jnp.array(rng.normal(size=(40, 3)))

    R_true = rod_rot(jnp.array([0.0, 0.0, 1.0]), 0.7)
    d_true = jnp.array([1.0, -2.0, 0.5])
    Q = P @ R_true.T + d_true

    R, d = find_rigid_transform(P, Q)
    np.testing.assert_allclose(R, R_true, atol=1e-4)
    np.testing.assert_allclose(d, d_true, atol=1e-4)
    np.testing.assert_allclose(P @ R.T + d, Q, atol=1e-4)


def test_find_rigid_transform_never_reflects():
    """A mirrored target must not produce a determinant -1 'rotation'."""
    rng = np.random.default_rng(4)
    P = jnp.array(rng.normal(size=(40, 3)))
    Q = P * jnp.array([1.0, 1.0, -1.0])

    R, _ = find_rigid_transform(P, Q)
    assert float(jnp.linalg.det(R)) == pytest.approx(1.0, abs=1e-5)
