"""Building blocks shared by the Gemma model implementations.

RMSNorm and the RoPE helpers are identical across the Gemma 4 variants, so
they live here rather than being copied into each model file.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

DEFAULT_RMS_EPS = 1e-6


class RMSNorm(eqx.Module):
    """Root-mean-square layer norm, with an optional learnable scale.

    with_scale=False matches the value normalisation in Gemma 4 attention,
    which normalises V without a learnable weight.
    """

    weight: jax.Array | None
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = DEFAULT_RMS_EPS, with_scale: bool = True):
        self.weight = jnp.ones(dim) if with_scale else None
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        normed = x / rms
        return normed * self.weight if self.weight is not None else normed


def make_rope_cache(max_seq: int, head_dim: int, theta: float,
                    partial_factor: float = 1.0):
    """Return (cos, sin) each of shape (max_seq, head_dim).

    For partial_factor < 1 the high-frequency tail of inv_freq is zeroed so
    those head dimensions receive cos=1, sin=0 (identity rotation = no-RoPE).
    This is the "proportional" RoPE used by Gemma 4 full-attention layers,
    and matches transformers' _compute_proportional_rope_parameters.
    """
    n_rot = int(partial_factor * head_dim // 2)   # number of non-zero inv_freq entries
    n_nope = head_dim // 2 - n_rot                # zero-padded entries

    inv_rot = 1.0 / (theta ** (
        jnp.arange(0, 2 * n_rot, 2, dtype=jnp.float32) / head_dim
    ))
    inv_freq = jnp.concatenate([inv_rot, jnp.zeros(n_nope, dtype=jnp.float32)])

    freqs = jnp.outer(jnp.arange(max_seq, dtype=jnp.float32), inv_freq)  # (T, HD/2)
    emb = jnp.concatenate([freqs, freqs], axis=-1)                       # (T, HD)
    return jnp.cos(emb), jnp.sin(emb)


def rotate_half(x: jax.Array) -> jax.Array:
    h = x.shape[-1] // 2
    return jnp.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """x: (S, H, HD), cos/sin: (S, HD) -> (S, H, HD)."""
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]
