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
    """Path to one contextual embedding file.

    Falls back to matching on the leading index when the exact name is not
    present. The names in lift.study record how these files were originally
    written, and small differences creep in -- a missing hyphen in
    "17bosThere_..." , a different rendering of the BOS token, a reworded
    context -- none of which change which step of the study a file belongs
    to. The index is what identifies it.
    """
    directory = CONTEXT_DIR / subdir
    exact = directory / filename
    if exact.exists() or not directory.is_dir():
        return exact

    wanted = _leading_index(filename)
    if wanted is None:
        return exact

    matches = sorted(
        p for p in directory.glob("*-embed.npy")
        if _leading_index(p.name) == wanted
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"{filename!r} is not in {directory}, and several files share index "
            f"{wanted}: {[p.name for p in matches]}. Remove the duplicates."
        )
    return exact


def _leading_index(filename: str) -> int | None:
    """The run of digits a context filename starts with, e.g. 17 for 17bos..."""
    digits = ""
    for ch in filename:
        if not ch.isdigit():
            break
        digits += ch
    return int(digits) if digits else None


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
