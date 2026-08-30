#!/usr/bin/env python3
"""Dump contextual embedding spaces for every prefix of a context.

For a context of n tokens this writes n matrices: the embedding space as the
model sees it after 1 token, after 2 tokens, and so on. With --branch it
instead extends the full context by one word at a time (" the", " your",
" our", ...) and dumps one matrix per branch.

Each matrix is (vocab, hidden) and around 2 GB for Gemma 4, so this writes a
lot of data -- everything lands under data/embeddings/contexts/.

Was main2(), main3() and context_embeddings() in main.py.

Usage:
    python experiments/extract_context_embeddings.py
    python experiments/extract_context_embeddings.py --branch
    python experiments/extract_context_embeddings.py --context "There is a mole" --subdir mole
"""

import argparse

import jax.numpy as jnp

from lift.paths import CONTEXT_DIR
from lift.study import BRANCH_WORDS, CONTEXT, CONTEXT_SUBDIR


def token_table(tokenizer, context_ids):
    """[(decoded token, id), ...] for a context, for eyeballing the split."""
    return [(tokenizer.decode([int(i)]), int(i)) for i in context_ids]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", default=CONTEXT, help=f"context to probe (default: {CONTEXT!r})")
    ap.add_argument("--subdir", default=CONTEXT_SUBDIR,
                    help="subdirectory of data/embeddings/contexts/ to write into")
    ap.add_argument("--branch", action="store_true",
                    help="extend the full context by one word each, instead of "
                         "walking its prefixes")
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--model", choices=["gemma4", "gemma4-12b", "smollm2"], default="gemma4")
    args = ap.parse_args()

    from lift.models import load
    mod = load(args.model)
    build_model, get_contextual_embeddings = mod.build_model, mod.get_contextual_embeddings

    model, tokenizer = build_model()
    out_dir = CONTEXT_DIR / args.subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.branch:
        contexts = [args.context + w for w in BRANCH_WORDS]
    else:
        ids = [tokenizer.bos_token_id] + tokenizer(args.context, add_special_tokens=False)["input_ids"]
        contexts = [tokenizer.decode(ids[1:i + 1]) for i in range(len(ids))]

    for i, ctx in enumerate(contexts):
        context_ids = jnp.array(
            [tokenizer.bos_token_id] + tokenizer(ctx, add_special_tokens=False)["input_ids"]
        )
        print(f"[{i + 1}/{len(contexts)}] {token_table(tokenizer, context_ids)}")

        decoded = tokenizer.decode(context_ids).replace(" ", "_").replace("/", "_")
        path = out_dir / f"{i + 1}-{decoded}-embed.npy"
        get_contextual_embeddings(model, context_ids, chunk_size=args.chunk_size,
                                  save=True, path=str(path))


if __name__ == "__main__":
    main()
