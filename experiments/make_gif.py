#!/usr/bin/env python3
"""Assemble a directory of numbered frames into a gif.

Replaces animation/animate.py and temp/animate.py, which were two copies of
this loop sitting inside the directories they wrote to, each with the frame
pattern hardcoded.

Usage:
    python experiments/make_gif.py outputs/animation --pattern "{i}-umap.png"
    python experiments/make_gif.py outputs/temp --duration 70 --out morph.gif
    python experiments/make_gif.py outputs/gifs --stitch
"""

import argparse
from pathlib import Path

from lift.viz.gif import frames_to_gif, numbered_frames, stitch_gifs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="directory holding the frames")
    ap.add_argument("--pattern", default="{i}.png",
                    help="frame filename pattern, with {i} counting from 0")
    ap.add_argument("--count", type=int, default=50, help="highest frame index to look for")
    ap.add_argument("--first", help="an extra frame to place before frame 0")
    ap.add_argument("--out", default="animation.gif", help="output filename inside folder")
    ap.add_argument("--duration", type=int, default=500, help="ms per frame")
    ap.add_argument("--stitch", action="store_true",
                    help="stitch consecutive step gifs (0-1.gif, 1-2.gif, ...) instead")
    args = ap.parse_args()

    if args.stitch:
        stitch_gifs(args.folder, args.folder / "combined_continuous.gif")
        return

    frames = numbered_frames(args.folder, args.pattern, args.count, args.first)
    frames_to_gif(frames, args.folder / args.out, duration=args.duration)


if __name__ == "__main__":
    main()
