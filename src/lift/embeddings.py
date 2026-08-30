"""Loading embedding matrices off disk.

Every experiment needs the same two things: the initial (pre-transformer)
embedding table, and one or more contextual tables rescaled so their row
lengths match the initial ones -- the transformer changes both direction and
magnitude, and rescaling isolates the directional change.
"""

import jax.numpy as jnp
import numpy as np

from lift.geometry.metrics import rescale_embeddings
from lift.paths import CONTEXT_DIR, EMBEDDINGS_DIR
from lift.study import CONTEXT_SUBDIR


def load(path) -> jnp.ndarray:
    """Load a .npy matrix as a jax array."""
    return jnp.array(np.load(path))


def initial(model: str = "gemma4") -> jnp.ndarray:
    """The model's initial embedding table, (vocab, hidden)."""
    return load(EMBEDDINGS_DIR / f"{model}_embed_initial.npy")


def context_path(filename: str, subdir: str = CONTEXT_SUBDIR):
    """Path to one contextual embedding file."""
    return CONTEXT_DIR / subdir / filename


def contextual(filename: str, scales=None, subdir: str = CONTEXT_SUBDIR) -> jnp.ndarray:
    """One contextual embedding table, optionally rescaled to `scales`."""
    E = load(context_path(filename, subdir))
    return E if scales is None else rescale_embeddings(E, scales)


def initial_scales(model: str = "gemma4") -> jnp.ndarray:
    """Row lengths of the initial embedding table."""
    return jnp.linalg.norm(initial(model), axis=-1)


def sequence(filenames, scales=None, subdir: str = CONTEXT_SUBDIR):
    """Load a list of contextual tables in order, rescaled to `scales`."""
    return [contextual(f, scales, subdir) for f in filenames]
