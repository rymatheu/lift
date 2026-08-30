#!/usr/bin/env python3
"""Estimate the intrinsic dimension of embedding spaces with the VGT method.

Three modes, which were estimate_dim(), estimate_dim2() and estimate_dim3()
in analysis.py plus main() in test_dim.py:

  radius   sweep the VGT radius for one token in a few embedding spaces, to
           find the plateau where the estimate is stable
  context  hold the radius sweep fixed and walk the context, showing how one
           word's local dimension moves as context accumulates
  words    sweep the radius for each of the four sense words in a single
           embedding space

Usage:
    python experiments/intrinsic_dim.py radius
    python experiments/intrinsic_dim.py context --word " equation"
    python experiments/intrinsic_dim.py words --step 18
"""

import argparse

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lift.embeddings import contextual, initial, initial_scales
from lift.geometry.dic_jax import estimate_global_parameters
from lift.paths import output
from lift.study import BRANCH_FILES, PREFIX_FILES, WORDS_OF_INTEREST

# vmap over r_max, the last argument of estimate_global_parameters(X, i, r_min, r_max)
sweep_radius = jax.jit(jax.vmap(estimate_global_parameters, in_axes=(None, None, None, 0)))


def word_id(name: str) -> int:
    for w, tid in WORDS_OF_INTEREST:
        if w.strip() == name.strip():
            return tid
    raise SystemExit(f"unknown word {name!r}; known: {[w for w, _ in WORDS_OF_INTEREST]}")


def mode_radius(args):
    """Compare the radius sweep across initial, mid-context and full-context."""
    r_max = jnp.linspace(args.r_min_sweep, args.r_max_sweep, args.n_radii)

    X_init = initial()
    scales = jnp.linalg.norm(X_init, axis=-1)
    X_mid = contextual(PREFIX_FILES[3], scales)
    X_cont = contextual(BRANCH_FILES[2], scales)

    tid = args.token if args.token is not None else int(np.random.randint(0, 200_000))
    print(f"token id {tid}")

    n_init, _, _ = sweep_radius(X_init, tid, 0.0, r_max)
    n_mid, _, _ = sweep_radius(X_mid, tid, 0.0, r_max)
    n_cont, _, _ = sweep_radius(X_cont, tid, 0.0, r_max)

    plt.plot(r_max, n_init, label="init_embed")
    plt.plot(r_max, n_mid, label="context_mid")
    plt.plot(r_max, n_cont, label="context_embed")
    plt.legend()
    plt.xlabel("r_max")
    plt.ylabel("estimated local dim")
    out = output("figs", "dim.png")
    plt.savefig(out)
    print(f"saved {out}")


def mode_context(args):
    """Walk the context, plotting the median dimension estimate at each step."""
    e_init = initial()
    scales = jnp.linalg.norm(e_init, axis=-1)

    E = [e_init]
    for filename in PREFIX_FILES + [BRANCH_FILES[0]]:
        E.append(contextual(filename, scales))
    E = jnp.array(E)
    print(f"embedding stack {E.shape}")

    # vmap over the context axis as well as the radius axis
    sweep_all = jax.jit(jax.vmap(sweep_radius, in_axes=(0, None, None, None)))

    r_max = jnp.linspace(args.r_min_sweep, args.r_max_sweep, args.n_radii)
    tid = word_id(args.word)

    # Split in half; the full stack OOMs.
    half = E.shape[0] // 2
    n1, _, _ = sweep_all(E[:half], tid, 0.0, r_max)
    n2, _, _ = sweep_all(E[half:], tid, 0.0, r_max)
    n = jnp.concatenate([n1, n2], axis=0)

    n = jnp.median(n, axis=-1)
    plt.plot(n)
    plt.xlabel("Steps (Context Number)")
    plt.ylabel(f"Median estimated dim for {args.word!r}")
    out = output("figs", "dimm.png")
    plt.savefig(out)
    print(f"saved {out}")


def mode_words(args):
    """Sweep the radius for all four sense words in one embedding space."""
    files = PREFIX_FILES + BRANCH_FILES
    if not 1 <= args.step <= len(files):
        raise SystemExit(f"--step must be between 1 and {len(files)}")
    X = contextual(files[args.step - 1])

    r = jnp.linspace(args.r_min_sweep, args.r_max_sweep, args.n_radii)
    for word, tid in WORDS_OF_INTEREST:
        dim, _, _ = sweep_radius(X, tid, 0.0, r)
        plt.plot(r, jnp.clip(dim, 0.0), label=word.strip())

    plt.legend()
    plt.xlabel("VGT Radius")
    plt.ylabel("Estimated Dim")
    plt.title(f"VGT Sweep for Context {args.step}")
    out = output("figs", "dim", f"{args.step}-dim.png")
    plt.savefig(out)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    radius = sub.add_parser("radius", help="radius sweep for one token across three spaces")
    radius.add_argument("--token", type=int, help="token id (default: random)")
    radius.set_defaults(func=mode_radius, r_min_sweep=0.6, r_max_sweep=1.8, n_radii=500)

    context = sub.add_parser("context", help="dimension of one word through the context")
    context.add_argument("--word", default=" equation",
                         help="which sense word to track (default: ' equation')")
    context.set_defaults(func=mode_context, r_min_sweep=0.6, r_max_sweep=1.8, n_radii=500)

    words = sub.add_parser("words", help="radius sweep for all four words in one space")
    words.add_argument("--step", type=int, default=18, help="context step, 1..18")
    words.set_defaults(func=mode_words, r_min_sweep=50.0, r_max_sweep=1000.0, n_radii=500)

    for p in (radius, context, words):
        p.add_argument("--r-min-sweep", type=float, dest="r_min_sweep")
        p.add_argument("--r-max-sweep", type=float, dest="r_max_sweep")
        p.add_argument("--n-radii", type=int, dest="n_radii")

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
