#!/usr/bin/env python3
"""Align two reduced clouds with a rigid motion and animate the alignment.

UMAP is only determined up to a rigid motion, so two runs of the same data
land in different orientations. This solves for the rotation and translation
that best maps one cloud onto the other (Kabsch), reports the MSE before and
after, and optionally renders the interpolation between the two poses --
rotating along the geodesic via Rodrigues' formula rather than lerping the
matrix.

Was a commented-out block in analysis.py's __main__, using helpers that are
now lift.geometry.transforms.

Usage:
    python experiments/rigid_align.py --a 1_umap --b 2_umap
    python experiments/rigid_align.py --a 1_umap --b 2_umap --animate
"""

import argparse

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lift.embeddings import load
from lift.geometry.transforms import find_rigid_transform, rod_rot
from lift.paths import DATA_DIR, output_dir
from lift.viz.plots import plot_3d_umap


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="source cloud name under data/reduced/")
    ap.add_argument("--b", required=True, help="target cloud name under data/reduced/")
    ap.add_argument("--animate", action="store_true", help="render the interpolation frames")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--name", default="align", help="subdirectory of outputs/ for frames")
    args = ap.parse_args()

    X1 = load(DATA_DIR / "reduced" / f"{args.a}.npy")
    X2 = load(DATA_DIR / "reduced" / f"{args.b}.npy")

    print(f"MSE pre transform:  {jnp.mean((X1 - X2) ** 2)}")

    R, d = find_rigid_transform(X1, X2)
    X1_t = jnp.dot(X1, R.T) + d
    print(f"MSE post transform: {jnp.mean((X1_t - X2) ** 2)}")

    if not args.animate:
        return

    # Recover the axis and angle of R so the two poses can be interpolated
    # along the rotation geodesic instead of through the ambient space.
    _, eigvecs = jnp.linalg.eig(R)
    axis = jnp.real(eigvecs[:, -1])
    angle = jnp.arccos(0.5 * (jnp.linalg.trace(R) - 1))

    shift = jnp.linalg.norm(d)
    direction = d / shift

    out = output_dir(args.name)
    ts = jnp.linspace(0, angle, args.frames)
    ds = jnp.linspace(0, shift, args.frames)

    for i in range(args.frames):
        X1_i = jnp.dot(X1, rod_rot(axis, ts[i]).T) + direction * ds[i]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")
        plot_3d_umap(ax, X1_i, "", point_cmap="winter")
        plot_3d_umap(ax, X2, "", point_cmap="autumn")
        plt.tight_layout()
        plt.savefig(out / f"{i}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"{i + 1}/{args.frames}")


if __name__ == "__main__":
    main()
