"""Similarity and rescaling helpers shared across experiments."""

import jax
import jax.numpy as jnp


@jax.jit
def cosine_similarity(A, B, eps=1e-8):
    """
    Computes cosine similarity supporting both 1D vectors and 2D matrices.

    Supports inputs of shapes:
    - A: (d,) and B: (d,)     -> Returns scalar or (1,) array
    - A: (N, d) and B: (N, d) -> Returns (N,) array
    """
    # Force both inputs to be at least 2D: (d,) becomes (1, d)
    A_2d = jnp.atleast_2d(A)
    B_2d = jnp.atleast_2d(B)

    # Standard row-wise calculation
    dot_product = jnp.sum(A_2d * B_2d, axis=-1)
    norm_A = jnp.linalg.norm(A_2d, axis=-1)
    norm_B = jnp.linalg.norm(B_2d, axis=-1)

    result = dot_product / jnp.maximum(norm_A * norm_B, eps)

    # Optional: If the original input was 1D, squeeze the result back to a scalar
    return jnp.squeeze(result) if A.ndim == 1 else result


@jax.jit
def rescale_embeddings(E, scales):
    """Rescale each row of E to the corresponding length in scales."""
    # Add a small epsilon to prevent division by zero for zero-vectors
    norms = jnp.linalg.norm(E, axis=-1, keepdims=True)
    norms = jnp.where(norms == 0, 1.0, norms)

    # Expand M_norms to shape (N, 1) for proper broadcasting
    scales = scales[:, jnp.newaxis]

    # Rescale M'
    return E * (scales / norms)
