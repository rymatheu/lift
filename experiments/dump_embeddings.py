#!/usr/bin/env python3
"""Dump a model's initial and contextual embedding spaces for one context.

Writes, under data/embeddings/:
    <model>_embed_initial.npy      (vocab, hidden) pre-transformer table
    <model>_embed_contextual.npy   (vocab, hidden) after the context
    <model>_top_tokens.npy         top-50 next-token ids
    <model>_top_probs.npy          their probabilities
    <model>_top_tokens_text.npy    their decoded text
    <model>_all_probs.npy          the full next-token distribution

Was the __main__ block of gemma4.py and smollm2.py, which held two copies of
this and wrote into a relative embeddings/ directory.

Usage:
    python experiments/dump_embeddings.py --model gemma4
    python experiments/dump_embeddings.py --model smollm2 --prompt-index 2
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from lift.paths import EMBEDDINGS_DIR

GETTYSBURG = (
    "Four score and seven years ago our fathers brought forth, upon this "
    "continent, a new nation, conceived in liberty, and dedicated to the "
    "proposition that all men are created equal. Now we are engaged in a great "
    "civil war, testing whether that nation, or any nation so conceived, and so "
    "dedicated, can long endure. We are met on a great battle field of that war. "
    "We come to dedicate a portion of it, as a final resting place for those who "
    "died here, that the nation might live. This we may, in all propriety do. But, "
    "in a larger sense, we can not dedicate we can not consecrate we can not "
    "hallow, this ground The brave men, living and dead, who struggled here, have "
    "hallowed it, far above our poor power to add or detract. The world will "
    "little note, nor long remember what we say here; while it can never forget "
    "what they did here. It is rather for us, the living, we here be dedicated to "
    "the great task remaining before us that, from these honored dead we take "
    "increased devotion to that cause for which they here, gave the last full "
    "measure of devotion that we here highly resolve these dead shall not have "
    "died in vain; that the nation, shall have a new birth of freedom, and that "
    "government of the people, by the people, for the people, shall not perish "
    "from the"
)

PROMPTS = [
    "Four score and seven",
    GETTYSBURG[:GETTYSBURG.index(" equal.")],
    GETTYSBURG,
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["gemma4", "gemma4-12b", "smollm2"], default="gemma4")
    ap.add_argument("--prompt", help="context to condition on (overrides --prompt-index)")
    ap.add_argument("--prompt-index", type=int, default=1,
                    help=f"which built-in prompt to use, 0..{len(PROMPTS) - 1} (default: 1)")
    ap.add_argument("--chunk-size", type=int, default=4096)
    args = ap.parse_args()

    from lift.models import load
    mod = load(args.model)
    build_model = mod.build_model
    get_contextual_embeddings = mod.get_contextual_embeddings
    save_initial_embeddings = mod.save_initial_embeddings

    context = args.prompt if args.prompt else PROMPTS[args.prompt_index]

    model, tokenizer = build_model()

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = EMBEDDINGS_DIR / args.model

    print("saving initial embedding space...")
    save_initial_embeddings(model, path=f"{prefix}_embed_initial.npy")

    print(f"context: {context!r}")
    context_ids = jnp.array(
        [tokenizer.bos_token_id] + tokenizer(context, add_special_tokens=False)["input_ids"]
    )
    print(f"getting contextual embeddings ({context_ids.shape[0]} context tokens)...")
    get_contextual_embeddings(
        model, context_ids,
        chunk_size=args.chunk_size,
        path=f"{prefix}_embed_contextual.npy",
    )

    logits = model(context_ids)
    probs = np.array(jax.nn.softmax(logits[-1]))
    top50 = np.array(jnp.argsort(logits[-1])[::-1][:50], dtype=np.int32)
    np.save(f"{prefix}_top_tokens.npy", top50)
    np.save(f"{prefix}_top_probs.npy", probs[top50].astype(np.float32))
    np.save(f"{prefix}_all_probs.npy", probs.astype(np.float32))
    top50_text = np.array([tokenizer.decode(i) for i in top50])
    np.save(f"{prefix}_top_tokens_text.npy", top50_text)
    print("top 50 tokens:", top50_text.tolist())


if __name__ == "__main__":
    main()
