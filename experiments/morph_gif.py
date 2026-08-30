#!/usr/bin/env python3
"""Animate the morph between two reduced embedding clouds.

Linearly interpolates between two 3D clouds from data/reduced/, keeping the
four sense words marked and labelled throughout, and writes a gif per pair.
Both clouds are centred first so the animation shows shape change rather
than drift.

Was animate() in analysis.py, driven by a commented-out list of pairs in its
__main__ block.

Usage:
    python experiments/morph_gif.py --pairs 0-1 1-2 2-3
    python experiments/morph_gif.py --consecutive 10 --stitch
"""

import argparse
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from lift.paths import DATA_DIR, output_dir
from lift.study import WORDS_OF_INTEREST
from lift.viz.gif import stitch_gifs

# Fixed so every gif in a series shares one scale and the frames can be
# compared across pairs.
AXIS_RANGE = 8.307863


def morph(a: int, b: int, woi, num_frames: int, out_dir):
    cloud_A = np.load(DATA_DIR / "reduced" / f"{a}.npy")
    cloud_B = np.load(DATA_DIR / "reduced" / f"{b}.npy")

    # Shift centers of gravity of both point clouds to the origin
    cloud_A_centered = cloud_A - np.mean(cloud_A, axis=0)
    cloud_B_centered = cloud_B - np.mean(cloud_B, axis=0)

    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Pin the limits so matplotlib does not relayout the grid every frame
    ax.set_xlim([-AXIS_RANGE, AXIS_RANGE])
    ax.set_ylim([-AXIS_RANGE, AXIS_RANGE])
    ax.set_zlim([-AXIS_RANGE, AXIS_RANGE])
    ax.tick_params(labelsize=6)

    colors = np.arange(cloud_A.shape[0])
    scat = ax.scatter(
        cloud_A_centered[:, 0], cloud_A_centered[:, 1], cloud_A_centered[:, 2],
        c=colors, cmap="turbo", s=4, alpha=0.6, linewidths=0,
    )

    indices = [tid for _, tid in woi]
    marker_scat = ax.scatter(
        cloud_A_centered[indices, 0],
        cloud_A_centered[indices, 1],
        cloud_A_centered[indices, 2],
        color="red", marker="^", s=10, edgecolors="white", linewidths=0.8,
        zorder=15, depthshade=False,
    )

    text_objects = []
    for word, tid in woi:
        x, y, z = cloud_A_centered[tid]
        text_objects.append(ax.text(
            x, y, z, word, fontsize=4, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", ec="black", alpha=0.9),
            zorder=20,
        ))

    frames = []
    for frame in range(num_frames):
        alpha = frame / (num_frames - 1)
        current = (1.0 - alpha) * cloud_A_centered + alpha * cloud_B_centered

        # Update the existing scatter in place rather than redrawing
        scat._offsets3d = (current[:, 0], current[:, 1], current[:, 2])

        tracked = current[indices]
        marker_scat._offsets3d = (tracked[:, 0], tracked[:, 1], tracked[:, 2])
        for i in range(len(indices)):
            text_objects[i].set_position_3d(tuple(tracked[i]))

        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf))

    plt.close(fig)

    out_path = out_dir / f"{a}-{b}.gif"
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=70, loop=1)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", metavar="A-B",
                    help="cloud index pairs to morph between, e.g. 0-1 9-14")
    ap.add_argument("--consecutive", type=int, metavar="N",
                    help="morph 0-1, 1-2, ... up to N")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--stitch", action="store_true",
                    help="also stitch the consecutive gifs into one sequence")
    args = ap.parse_args()

    if not args.pairs and not args.consecutive:
        ap.error("give --pairs or --consecutive")

    pairs = []
    if args.consecutive:
        pairs += [(i, i + 1) for i in range(args.consecutive)]
    if args.pairs:
        for p in args.pairs:
            a, b = p.split("-")
            pairs.append((int(a), int(b)))

    out_dir = output_dir("gifs")
    for a, b in pairs:
        morph(a, b, WORDS_OF_INTEREST, args.frames, out_dir)

    if args.stitch:
        stitch_gifs(out_dir, out_dir / "combined_continuous.gif")


if __name__ == "__main__":
    main()
