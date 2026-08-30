#!/usr/bin/env python3
"""Plot initial vs contextual embedding spaces side by side under PCA.

Was the __main__ block of plot_embeddings.py.

Usage:
    python experiments/pca_plot.py --model gemma4
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lift.geometry.pca import pca
from lift.paths import EMBEDDINGS_DIR, output
from lift.viz.plots import plot_3d


def _optional(path):
    return np.load(path, allow_pickle=True) if path.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["gemma4", "gemma4-12b", "smollm2"], default="gemma4")
    args = ap.parse_args()

    prefix = EMBEDDINGS_DIR / args.model
    initial_path = prefix.with_name(f"{args.model}_embed_initial.npy")
    contextual_path = prefix.with_name(f"{args.model}_embed_contextual.npy")

    print(f"loading {initial_path}...")
    E0 = np.load(initial_path).astype(np.float32)
    print(f"loading {contextual_path}...")
    Ec = np.load(contextual_path).astype(np.float32)

    top_ids = _optional(prefix.with_name(f"{args.model}_top_tokens.npy"))
    top_text = _optional(prefix.with_name(f"{args.model}_top_tokens_text.npy"))

    print("PCA on initial embeddings...")
    proj0, var0 = pca(E0)
    print("PCA on contextual embeddings...")
    projc, varc = pca(Ec)

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle("Token embedding space: initial vs contextual", fontsize=13)

    ax0 = fig.add_subplot(121, projection="3d")
    plot_3d(ax0, proj0, "Initial (raw embed)", var0, top_ids, top_text)

    ax1 = fig.add_subplot(122, projection="3d")
    plot_3d(ax1, projc, "Contextual (post-transformer)", varc, top_ids, top_text)

    plt.tight_layout()
    out = output("figs", f"{args.model}_embeddings_pca.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
