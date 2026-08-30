#!/usr/bin/env python3
"""List a token's nearest neighbours under cosine similarity and L2.

The two metrics disagree once the transformer has rescaled the space, which
is the point: cosine sees direction only, L2 sees the magnitude change too.

Was temp() in analysis.py.

Usage:
    python experiments/neighbors.py
    python experiments/neighbors.py --token 7972 --step 9 --k 20
"""

import argparse

import jax
import jax.numpy as jnp

from lift.embeddings import contextual
from lift.study import BRANCH_FILES, PREFIX_FILES, WORDS_OF_INTEREST


@jax.jit
def distances(X, v):
    """Euclidean distance and cosine similarity from every row of X to v."""
    euclidean = jnp.sqrt(jnp.sum((X - v) ** 2, axis=-1))

    dot = jnp.dot(X, v)
    norm_X = jnp.linalg.norm(X, axis=-1)
    norm_v = jnp.linalg.norm(v)
    # eps guards against a zero row
    cosine = dot / (norm_X * norm_v + 1e-8)

    return euclidean, cosine


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", type=int, default=WORDS_OF_INTEREST[0][1],
                    help=f"token id to centre on (default: {WORDS_OF_INTEREST[0][1]}, ' garden')")
    ap.add_argument("--step", type=int, default=1, help="context step, 1..18 (default: 1)")
    ap.add_argument("--k", type=int, default=100, help="how many neighbours to list")
    args = ap.parse_args()

    from lift.models.tokenizer import load_tokenizer

    files = PREFIX_FILES + BRANCH_FILES
    if not 1 <= args.step <= len(files):
        ap.error(f"--step must be between 1 and {len(files)}")

    tokenizer = load_tokenizer()
    X = contextual(files[args.step - 1])

    dist, sim = distances(X, X[args.token])

    _, by_cosine = jax.lax.top_k(sim, k=args.k)
    # top_k wants the largest values, so negate for the smallest distances
    _, by_l2 = jax.lax.top_k(-dist, k=args.k)

    print(f"neighbours of {tokenizer.decode([args.token])!r} at context step {args.step}\n")
    print(f"{'cosine':<24} {'L2':<24}")
    for i in range(args.k):
        t1 = tokenizer.decode([int(by_cosine[i])])
        t2 = tokenizer.decode([int(by_l2[i])])
        print(f"{t1!r:<24} {t2!r:<24}")


if __name__ == "__main__":
    main()
