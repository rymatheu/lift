#!/usr/bin/env python3
"""Render one UMAP frame per context prefix, then assemble them into a gif.

Walks the prefixes of a context, computes the contextual embedding space at
each step, reduces it to 3D and plots it -- so the gif shows the whole
embedding space reorganising as the context accumulates. Also prints the
model's next-token prediction at each step.

Was main() and initial_embeddings() in main.py.

Usage:
    python experiments/umap_frames.py
    python experiments/umap_frames.py --context "There is a mole in the garden." --name garden
"""

import argparse

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lift.embeddings import initial
from lift.geometry.dimred import reduce
from lift.paths import output_dir
from lift.study import CONTEXT
from lift.viz.gif import frames_to_gif, numbered_frames
from lift.viz.plots import plot_3d_umap


def render(coords, title, path):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    plot_3d_umap(ax, coords, title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", default=CONTEXT + " your back.")
    ap.add_argument("--name", default="animation",
                    help="subdirectory of outputs/ to write frames into")
    ap.add_argument("--initial-frame", action="store_true",
                    help="also render the untouched embedding table as frame 'init'")
    ap.add_argument("--no-gif", action="store_true", help="render frames only")
    ap.add_argument("--frame-duration", type=int, default=500, help="ms per gif frame")
    ap.add_argument("--model", choices=["gemma4", "gemma4-12b", "smollm2"], default="gemma4")
    args = ap.parse_args()

    from lift.models import load
    mod = load(args.model)
    build_model, get_contextual_embeddings = mod.build_model, mod.get_contextual_embeddings

    out = output_dir(args.name)
    model, tokenizer = build_model()

    if args.initial_frame:
        print("reducing the initial embedding table...")
        render(reduce(initial(), 15, 0.1), "", out / "init-umap.png")

    context_ids = jnp.array(
        [tokenizer.bos_token_id] + tokenizer(args.context, add_special_tokens=False)["input_ids"]
    )
    tokens = [tokenizer.decode([int(i)]) for i in context_ids]
    print(tokens)

    for i in range(context_ids.shape[0]):
        prefix = context_ids[:i + 1]

        embeddings = get_contextual_embeddings(model, prefix, chunk_size=4096, save=False)

        logits = model(prefix)
        probs = jax.nn.softmax(logits[-2:], axis=-1)
        print(f"  next: {tokenizer.decode(jnp.argmax(probs, axis=-1))!r}")

        render(reduce(embeddings, 15, 0.1), tokenizer.decode(prefix), out / f"{i}-umap.png")
        print(f"done loop {i + 1}/{len(tokens)}\n")

    if not args.no_gif:
        frames = numbered_frames(out, "{i}-umap.png", count=len(tokens),
                                 first="init-umap.png" if args.initial_frame else None)
        frames_to_gif(frames, out / "umap_animation.gif", duration=args.frame_duration)


if __name__ == "__main__":
    main()
