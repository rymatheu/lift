#!/usr/bin/env python3
"""UMAP-reduce embedding matrices to 3D and plot them.

Reads (vocab, hidden) matrices, rescales them to the initial embedding
lengths, projects to 3D with cuML's UMAP and writes both the reduced cloud
(data/reduced/) and a scatter plot (outputs/reduced/).

Was main2() in analysis.py plus the __main__ block of the old umap.py, which
between them hardcoded three absolute paths.

Usage:
    python experiments/reduce_embeddings.py --set branch
    python experiments/reduce_embeddings.py --set prefix --no-rescale
    python experiments/reduce_embeddings.py --files data/embeddings/foo.npy
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lift.embeddings import contextual, initial_scales, load
from lift.geometry.dimred import BACKENDS, backend_in_use, reduce
from lift.paths import DATA_DIR, output
from lift.study import BRANCH_FILES, PREFIX_FILES
from lift.viz.plots import plot_3d_umap


def save_and_plot(X_r, name: str):
    out_npy = DATA_DIR / "reduced" / f"{name}.npy"
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, np.asarray(X_r).astype(np.float32))
    print(f"saved {X_r.shape} -> {out_npy}")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    plot_3d_umap(ax, X_r, "")
    plt.tight_layout()
    out_png = output("reduced", f"{name}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["prefix", "branch"], default="branch",
                    help="which group of context files to reduce (default: branch)")
    ap.add_argument("--files", nargs="+",
                    help="explicit .npy paths to reduce instead of a named set")
    ap.add_argument("--no-rescale", action="store_true",
                    help="skip rescaling rows to the initial embedding lengths")
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--backend", choices=BACKENDS, default="auto",
                    help="UMAP backend: cuml (GPU) or umap-learn (CPU); "
                         "auto prefers cuml (default: auto)")
    ap.add_argument("--random-state", type=int,
                    help="seed UMAP for a reproducible layout (slower)")
    args = ap.parse_args()

    print(f"UMAP backend: {backend_in_use(args.backend)}")

    scales = None if args.no_rescale else initial_scales()

    if args.files:
        for path in args.files:
            X = load(path)
            X_r = reduce(X, args.n_neighbors, args.min_dist,
                         backend=args.backend, random_state=args.random_state)
            save_and_plot(X_r, Path(path).stem)
        return

    files = PREFIX_FILES if args.set == "prefix" else BRANCH_FILES
    # Branch files are numbered 10..18 in the study, prefixes 1..9.
    offset = 1 if args.set == "prefix" else 10

    for i, filename in enumerate(files):
        X = contextual(filename, scales)
        X_r = reduce(X, args.n_neighbors, args.min_dist,
                     backend=args.backend, random_state=args.random_state)
        save_and_plot(X_r, str(i + offset))


if __name__ == "__main__":
    main()
