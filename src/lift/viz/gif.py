"""Assembling frame sequences into gifs.

Replaces three near-identical copies of the same loop that used to live
inside the output directories they wrote to (animation/animate.py,
temp/animate.py and gifs/gifs.py).
"""

import os
import re
from pathlib import Path

from PIL import Image


def frames_to_gif(
    frames: list[Path],
    out_path: Path,
    duration: int = 500,
    loop: int = 0,
) -> Path | None:
    """Write an ordered list of image files out as a single animated gif.

    Missing files are skipped with a warning. Returns the output path, or
    None if no frame could be read.
    """
    images = []
    for path in frames:
        if os.path.exists(path):
            images.append(Image.open(path))
        else:
            print(f"warning: {path} not found. Skipping.")

    if not images:
        print("error: no images were loaded. GIF not created.")
        return None

    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
    )
    print(f"wrote {out_path} ({len(images)} frames)")
    return out_path


def numbered_frames(
    folder: Path,
    pattern: str = "{i}.png",
    count: int = 50,
    first: Path | None = None,
) -> list[Path]:
    """Frame paths for `folder/pattern` with i counting up from 0.

    `first` prepends one extra frame (the pre-context initial state) when it
    exists.
    """
    frames = []
    if first is not None and os.path.exists(folder / first):
        frames.append(folder / first)
    frames.extend(folder / pattern.format(i=i) for i in range(count))
    return [f for f in frames if os.path.exists(f)]


def stitch_gifs(
    folder: Path,
    out_path: Path,
    standard_duration: int = 70,
    pause_duration: int = 600,
) -> Path | None:
    """Concatenate consecutive step gifs (0-1.gif, 1-2.gif, ...) into one.

    Only files whose two numbers are consecutive are picked up, so the
    non-sequential comparison gifs (9-14.gif and friends) sitting in the same
    directory are left out. Each segment's final frame is held for
    `pause_duration` so the steps stay readable.
    """

    def consecutive_key(filename: str) -> list[int] | None:
        nums = [int(x) for x in re.findall(r"\d+", filename)]
        if len(nums) == 2 and nums[1] == nums[0] + 1:
            return nums
        return None

    folder = Path(folder)
    if not folder.exists():
        print(f"error: the directory '{folder}' does not exist.")
        return None

    keyed = []
    for f in os.listdir(folder):
        if f.endswith(".gif"):
            key = consecutive_key(f)
            if key is not None:
                keyed.append((key, f))

    keyed.sort(key=lambda item: item[0])
    gif_files = [filename for _, filename in keyed]

    if not gif_files:
        print("no matching consecutive step GIFs (like 0-1.gif, 1-2.gif) found.")
        return None

    print(f"found {len(gif_files)} consecutive step GIFs to stitch:")
    for f in gif_files:
        print(f"  - {f}")

    all_frames = []
    frame_durations = []

    for filename in gif_files:
        with Image.open(folder / filename) as img:
            segment = []
            try:
                while True:
                    # Copy the frame out of the file buffer safely
                    segment.append(img.copy().convert("RGBA"))
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

            if segment:
                durations = [standard_duration] * len(segment)
                # Hold the last frame of each segment
                durations[-1] = pause_duration
                all_frames.extend(segment)
                frame_durations.extend(durations)

    if not all_frames:
        print("no frames were successfully extracted.")
        return None

    print(f"saving combined GIF with pauses ({len(all_frames)} total frames)...")
    # Convert frames back to palette mode for standard GIF compatibility
    optimized = [f.convert("P", palette=Image.ADAPTIVE) for f in all_frames]
    optimized[0].save(
        out_path,
        save_all=True,
        append_images=optimized[1:],
        duration=frame_durations,
        loop=0,
    )
    print(f"wrote {out_path}")
    return out_path
