#!/usr/bin/env python3
"""Track the Gram-matrix eigenvalues of the four sense words through context.

Takes the submatrix of the four disambiguating words (" garden", " back",
" operation", " equation") at each step of the context, forms D @ D.T and
plots how its eigenvalues move. A collapsing spectrum means the four senses
are being pulled onto a lower-dimensional subspace as the context resolves
the ambiguity.

Was main() in analysis.py.

Usage:
    python experiments/eigenvalues.py
"""

import argparse

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lift.embeddings import contextual, initial, initial_scales
from lift.paths import output
from lift.study import BRANCH_FILES, PREFIX_FILES, WORDS_OF_INTEREST


@jax.jit
def gram_eigenvalues(D):
    """Eigenvalues of D @ D.T, ascending."""
    return jnp.linalg.eigvalsh(jnp.dot(D, D.T))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="eigs.png", help="filename under outputs/")
    args = ap.parse_args()

    indices = jnp.array([tid for _, tid in WORDS_OF_INTEREST])
    print(f"tracking {[w for w, _ in WORDS_OF_INTEREST]} -> {indices}")

    e_init = initial()
    scales = initial_scales()

    # Step 0 is the untouched embedding table, then one step per context
    # prefix, then the first branch.
    submatrices = [e_init[indices]]

    print("getting context submatrices")
    for filename in PREFIX_FILES + [BRANCH_FILES[0]]:
        submatrices.append(contextual(filename, scales)[indices])

    eigs = jnp.array([gram_eigenvalues(D) for D in submatrices])

    for i in range(eigs.shape[1]):
        plt.plot(eigs[:, i])
    plt.xlabel("Steps (Context Number)")
    plt.ylabel("Eigenvalues")
    out = output(args.out)
    plt.savefig(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
