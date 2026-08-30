#!/usr/bin/env python3
"""Greedily generate continuations, redrawing the terminal as it goes.

A sanity check that the JAX model reproduces sensible text -- there is no KV
cache here, so every step re-runs the full forward pass and it is slow by
design.

Was test.py.

Usage:
    python experiments/generate.py
    python experiments/generate.py --context "There is a mole in the middle of the" -n 20
"""

import argparse

import jax.numpy as jnp

DEFAULT_CONTEXT = (
    "I have 2 apples, then I buy 2 more. I bake a pie with 2 of the apples. "
    "After eating half of the pie how many apples do I have left?"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", default=DEFAULT_CONTEXT)
    ap.add_argument("-n", "--max-new-tokens", type=int, default=50)
    ap.add_argument("--model", choices=["gemma4", "gemma4-12b", "smollm2"], default="gemma4")
    ap.add_argument("--quiet", action="store_true", help="only print the final result")
    args = ap.parse_args()

    from lift.models import load
    build_model = load(args.model).build_model

    model, tokenizer = build_model()

    context_ids = [tokenizer.bos_token_id] + tokenizer(args.context, add_special_tokens=False)["input_ids"]
    generated = list(context_ids)

    for _ in range(args.max_new_tokens):
        logits = model(jnp.array(generated))
        next_id = int(jnp.argmax(logits[-1]))
        generated.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
        if not args.quiet:
            print("\033[H\033[J", end="")  # clear screen, home cursor
            print(args.context)
            print()
            print(tokenizer.decode(generated[len(context_ids):]))

    new_ids = generated[len(context_ids):]
    print(new_ids)
    print(tokenizer.decode(new_ids))


if __name__ == "__main__":
    main()
