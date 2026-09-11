"""Extracting a context-conditioned embedding for every vocabulary token.

The probe works by appending a chunk of vocabulary tokens to the context in
a single forward pass, under a mask that lets each probe token see the whole
context and itself but not the other probes. Every probe therefore stands in
for the same slot -- "the next token after the context" -- and they all have
to share one RoPE position, `n`.

That last point is the part that is easy to get wrong. Letting positions run
on as arange(n + C) gives probe k the position of a token n + k steps into
the future, and only the first probe in each chunk comes out correct. This
module keeps the mask and the position ids together so the two models cannot
drift apart on it.
"""

import jax.numpy as jnp
import numpy as np


def probe_mask(n: int, C: int, sliding_window: int | None = None) -> jnp.ndarray:
    """(n + C, n + C) boolean attention mask for a context plus C probes.

    Context rows attend causally to the context. Probe rows attend to the
    whole context and to themselves only.

    sliding_window restricts both to a trailing window, matching what the
    model's sliding-attention layers do. Pass None for the full-attention
    layers. A probe sits at position n, so its window covers the last
    sliding_window - 1 context positions plus itself.
    """
    ctx_ctx = jnp.tril(jnp.ones((n, n), jnp.bool_))
    probe_ctx = jnp.ones((C, n), jnp.bool_)

    if sliding_window is not None:
        # Context token i may look back to i - (window - 1).
        ctx_ctx = ctx_ctx & jnp.triu(jnp.ones((n, n), jnp.bool_), -(sliding_window - 1))
        # Probe at position n may look back to n - (window - 1).
        first_visible = n - (sliding_window - 1)
        probe_ctx = probe_ctx & (jnp.arange(n)[None, :] >= first_visible)

    top = jnp.concatenate([ctx_ctx, jnp.zeros((n, C), jnp.bool_)], axis=1)
    bot = jnp.concatenate([probe_ctx, jnp.eye(C, dtype=jnp.bool_)], axis=1)
    return jnp.concatenate([top, bot], axis=0)


def probe_positions(n: int, C: int) -> jnp.ndarray:
    """RoPE positions: the context in order, then every probe pinned at n."""
    return jnp.concatenate([jnp.arange(n), jnp.full((C,), n)])


def contextual_embeddings(
    forward_masked,
    context_ids,
    vocab_size: int,
    hidden_size: int,
    chunk_size: int = 2048,
    path: str = "embed_contextual.npy",
    save: bool = True,
    sliding_window: int | None = None,
    progress_every: int = 20,
    token_ids=None,
):
    """Run the probe over the vocabulary, one chunk at a time.

    forward_masked(input_ids, mask, position_ids) must return the final
    hidden states, (n + C, hidden). When the model has sliding-attention
    layers, pass its window and forward_masked will receive a mask built as
    a dict {"sliding": ..., "full": ...} instead of a single array.

    token_ids probes only those tokens instead of the whole vocabulary,
    returning rows in the order given. A full probe of a 262k vocabulary
    writes a multi-gigabyte matrix; a few thousand tokens is enough to see
    the geometry and runs on a laptop.
    """
    n = context_ids.shape[0]

    probes = jnp.arange(vocab_size) if token_ids is None else jnp.asarray(token_ids)
    total = int(probes.shape[0])
    out = np.zeros((total, hidden_size), dtype=np.float32)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        C = end - start

        input_ids = jnp.concatenate([context_ids, probes[start:end]])
        position_ids = probe_positions(n, C)

        if sliding_window is None:
            mask = probe_mask(n, C)
        else:
            mask = {
                "sliding": probe_mask(n, C, sliding_window),
                "full": probe_mask(n, C),
            }

        hidden = forward_masked(input_ids, mask, position_ids)
        out[start:end] = np.array(hidden[n:].astype(jnp.float32))

        if progress_every and start % (chunk_size * progress_every) == 0:
            print(f"  {end}/{total}")

    if save:
        np.save(path, out)
        print(f"saved contextual embeddings ({total}, {hidden_size}) -> {path}")
    return out
