#!/usr/bin/env python3
"""Print the model's top-k next tokens for a context.

Was temp.py.

Usage:
    python experiments/next_tokens.py --context "There is a mole in the middle of my"
    python experiments/next_tokens.py -k 10
"""

import argparse

import jax
import jax.numpy as jnp

from lift.study import CONTEXT


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", default=CONTEXT + " my")
    ap.add_argument("-k", "--top-k", type=int, default=50)
    ap.add_argument("--model", choices=["gemma4", "gemma4-12b", "smollm2"], default="gemma4")
    args = ap.parse_args()

    from lift.models import load
    build_model = load(args.model).build_model

    model, tokenizer = build_model()

    context_ids = jnp.array(
        [tokenizer.bos_token_id] + tokenizer(args.context, add_special_tokens=False)["input_ids"]
    )
    probs = jax.nn.softmax(model(context_ids)[-1])
    top_probs, top_ids = jax.lax.top_k(probs, k=args.top_k)

    print(f"context: {args.context!r}\n")
    for tid, p in zip(top_ids, top_probs):
        print(f"{tokenizer.decode([int(tid)])!r:<20} {float(p):.4f}")


if __name__ == "__main__":
    main()
