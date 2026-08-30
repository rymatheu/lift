#!/usr/bin/env python3
"""Histogram the vector lengths of an embedding space.

The transformer changes embedding magnitudes as well as directions; this
shows the distribution of row norms, which is what motivates rescaling
elsewhere in the study.

Was hist() in analysis.py.

Usage:
    python experiments/embedding_lengths.py --step 10
    python experiments/embedding_lengths.py --initial
"""

import argparse

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lift.embeddings import contextual, initial
from lift.paths import output
from lift.study import BRANCH_FILES, PREFIX_FILES


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, default=10,
                    help="context step to histogram, 1..18 (default: 10)")
    ap.add_argument("--initial", action="store_true",
                    help="histogram the initial embedding table instead")
    args = ap.parse_args()

    if args.initial:
        E = initial()
        label = "Initial Embeddings"
        name = "initial"
    else:
        files = PREFIX_FILES + BRANCH_FILES
        if not 1 <= args.step <= len(files):
            ap.error(f"--step must be between 1 and {len(files)}")
        E = contextual(files[args.step - 1])
        label = f"Context {args.step}"
        name = str(args.step)

    lengths = jnp.linalg.norm(E, axis=1)
    mean, std = float(jnp.mean(lengths)), float(jnp.std(lengths))
    lo, hi = float(jnp.min(lengths)), float(jnp.max(lengths))

    plt.figure(figsize=(8, 5), dpi=150)
    plt.hist(lengths, bins=100, alpha=0.6, label=label, color="#ef4444", edgecolor="none")

    stats_text = (
        f"  Metric  │   Value   \n"
        f"──────────┼───────────\n"
        f"  Mean    │  {mean:8.2f}\n"
        f"  Std Dev │  {std:8.2f}\n"
        f"  Min     │  {lo:8.2f}\n"
        f"  Max     │  {hi:8.2f}"
    )

    ax = plt.gca()
    ax.text(
        0.7, 0.85, stats_text, transform=ax.transAxes, fontsize=9,
        fontfamily="monospace",  # Critical for column alignment
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9),
    )

    plt.title(f"Embedding Vector Lengths ({label})", fontsize=12, fontweight="bold")
    plt.xlabel("Magnitude", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    out = output("figs", f"{name}-embedding_lengths_histogram.png")
    plt.savefig(out)
    print(f"saved {out}")
    print(f"{label} -> Mean: {mean:.4f} | Std: {std:.4f} | Min: {lo:.4f} | Max: {hi:.4f}")


if __name__ == "__main__":
    main()
