"""SVD-based PCA, in both jax and numpy flavours."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnames=("n",))
def pca_jax(X: jnp.ndarray, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (projected, explained_variance_ratio) for the top n components."""
    X = X - X.mean(axis=0)
    _, S, Vt = jnp.linalg.svd(X, full_matrices=False)
    var_ratio = (S ** 2) / (S ** 2).sum()
    return X @ Vt[:n].T, var_ratio[:n]


def pca(X: np.ndarray, n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """numpy-in, numpy-out wrapper around pca_jax."""
    proj, var_ratio = pca_jax(jnp.asarray(X), n)
    return np.asarray(proj), np.asarray(var_ratio)
