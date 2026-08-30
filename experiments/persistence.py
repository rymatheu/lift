#!/usr/bin/env python3
"""H_0 persistence of a token's normalized neighbourhood.

Takes the k nearest neighbours of a token, punctures the neighbourhood at
that token (subtract it, then normalize), and computes the degree-zero
persistence diagram of the result via a minimum spanning tree. The
q-Wasserstein norm of that diagram summarises how clustered the
neighbourhood is -- a single number to compare across contexts.

Was p() in analysis.py.

Usage:
    python experiments/persistence.py --embeddings data/embeddings/semantics/initial.npy
    python experiments/persistence.py --token-text "<bos>" --k 50
"""

import argparse

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

from lift.embeddings import load
from lift.paths import DATA_DIR, output


def compute_h0_persistence(points) -> np.ndarray:
    """0D persistence diagram for (k, d) points, as (k, 2) birth/death pairs.

    Uses scipy for the minimum spanning tree: it is a discrete greedy
    algorithm, non-differentiable and a poor fit for JIT.
    """
    X_np = np.asarray(points)
    dist_matrix = squareform(pdist(X_np, metric="euclidean"))

    mst = minimum_spanning_tree(dist_matrix)

    # Non-zero MST entries are the merge distances, i.e. the death times
    death_times = np.sort(mst.data)

    k = X_np.shape[0]
    diagram = np.zeros((k, 2))
    # All components are born at 0; k-1 die at the MST edge weights and the
    # last global component lives forever.
    diagram[:-1, 1] = death_times
    diagram[-1, 1] = np.inf
    return diagram


def plot_h0_persistence(diagram: np.ndarray, out_path):
    births, deaths = diagram[:, 0], diagram[:, 1]
    finite = np.isfinite(deaths)

    max_finite = np.max(deaths[finite]) if np.any(finite) else 1.0
    # Draw the infinite component 20% above the highest finite death
    inf_y = max_finite * 1.2

    fig, ax = plt.subplots(figsize=(6, 6))
    limit = inf_y * 1.1
    ax.plot([0, limit], [0, limit], color="gray", linestyle="--", alpha=0.7, label="y = x")

    ax.scatter(births[finite], deaths[finite], color="royalblue", s=50, ec="k",
               zorder=3, label="$H_0$ Finite Merges")
    ax.scatter(births[~finite], [inf_y] * int((~finite).sum()), color="crimson",
               marker="^", s=60, ec="k", zorder=3, label=r"Global Component ($\infty$)")
    ax.axhline(inf_y, color="crimson", linestyle=":", alpha=0.5)

    ax.set_xlim(-0.05, limit)
    ax.set_ylim(-0.05, limit)

    ticks = [t for t in ax.get_yticks().tolist() if t < inf_y]
    ax.set_yticks(ticks + [inf_y])
    ax.set_yticklabels([f"{t:.2f}" for t in ticks] + [r"$\infty$"])

    ax.set_xlabel(r"Birth Time ($\epsilon$)")
    ax.set_ylabel(r"Death Time ($\epsilon$)")
    ax.set_title("0D ($H_0$) Persistence Diagram of Normalized Set")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def wasserstein_norm(diagram, q: int = 1):
    """q-Wasserstein norm of the diagram, ignoring the infinite component."""
    deaths = jnp.asarray(diagram[:, 1])
    finite_deaths = deaths[jnp.isfinite(deaths)]

    # Orthogonal distance from each (0, d) to the diagonal
    to_diagonal = finite_deaths / jnp.sqrt(2)

    if q == 1:
        return jnp.sum(to_diagonal)
    return jnp.power(jnp.sum(jnp.power(to_diagonal, q)), 1.0 / q)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", default=str(DATA_DIR / "embeddings" / "semantics" / "initial.npy"),
                    help="embedding matrix to read")
    ap.add_argument("--token-text", default="<bos>",
                    help="token to centre the neighbourhood on (default: '<bos>')")
    ap.add_argument("--token", type=int, help="token id, overriding --token-text")
    ap.add_argument("--k", type=int, default=50, help="neighbourhood size")
    ap.add_argument("--q", type=int, default=1, help="Wasserstein order")
    ap.add_argument("--save-neighborhood", help="write the neighbour submatrix here")
    args = ap.parse_args()

    if args.token is not None:
        tid = args.token
    else:
        from lift.models.tokenizer import load_tokenizer
        tid = load_tokenizer()(args.token_text, add_special_tokens=False)["input_ids"][0]
    print(f"centre token id {tid}")

    X = load(args.embeddings)

    # 1. Normalize all word vectors
    X = X / jnp.maximum(jnp.linalg.norm(X, axis=-1, keepdims=True), 1e-12)

    # 2. k closest neighbours, excluding the centre itself
    centre = X[tid]
    d = jnp.sum((X - centre) ** 2, axis=-1)
    _, ids = jax.lax.top_k(-d, k=args.k + 1)
    X = X[ids[1:]]

    if args.save_neighborhood:
        np.save(args.save_neighborhood, np.asarray(X))
        print(f"saved neighbourhood -> {args.save_neighborhood}")

    # 3. Puncture at the centre and renormalize
    M = X - centre
    M = M / jnp.maximum(jnp.linalg.norm(M, axis=-1, keepdims=True), 1e-12)

    # 4. Degree-zero persistence
    diagram = compute_h0_persistence(M)
    plot_h0_persistence(diagram, output("figs", "pers.png"))

    # 5. Wasserstein norm
    print(f"{args.q}-Wasserstein norm: {wasserstein_norm(diagram, args.q)}")


if __name__ == "__main__":
    main()
