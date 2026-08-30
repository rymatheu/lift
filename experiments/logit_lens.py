#!/usr/bin/env python3
"""Print a logit-lens table: the top-k predictions read off each layer.

BROKEN: this imports logit_lens from lift.models.gemma4, which has never
existed -- the function was never written, so this script has never run.
Everything below is the table renderer, which is complete; what is missing
is a forward pass that collects the hidden state after each decoder layer,
applies the final norm and the tied LM head, and returns
(NUM_LAYERS + 1, S, top_k) arrays of ids and probabilities.

Was logit_lense.py (sic).

Usage:
    python experiments/logit_lens.py --context "Chemistry. There is a mole in the middle of my"
"""

import argparse
import os

import jax.numpy as jnp
import wcwidth

TOK_W = 12  # fixed *display* width for the decoded-token field within each cell


def _fit(text: str, width: int) -> str:
    """Pad/truncate to exactly `width` terminal columns (CJK etc. render as 2 columns
    each, so plain len()-based padding misaligns as soon as a wide glyph shows up)."""
    w = wcwidth.wcswidth(text)
    if w is None or w < 0:
        w = len(text)
    if w <= width:
        return text + " " * (width - w)
    out, cur = [], 0
    for ch in text:
        cw = wcwidth.wcwidth(ch)
        cw = cw if (cw is not None and cw >= 0) else 1
        if cur + cw > width - 1:
            break
        out.append(ch)
        cur += cw
    out.append("\u2026")
    cur += 1
    return "".join(out) + " " * max(0, width - cur)


def cell(tokenizer, tok_id: int, prob: float) -> str:
    text = tokenizer.decode([tok_id]).replace("\n", "\\n").replace("\t", "\\t").replace(" ", "_")
    return f"{_fit(text, TOK_W)}{prob:>6.1%}"


def render(tokenizer, top_ids, top_probs, position: int, top_k: int, label_w: int = 8):
    header = f"{'layer':>{label_w}} | " + " | ".join(
        f"{'top-' + str(k + 1):<{TOK_W + 6}}" for k in range(top_k)
    )
    print()
    print(header)
    print("-" * len(header))
    for layer in range(1, top_ids.shape[0]):
        cells = [
            cell(tokenizer, int(top_ids[layer, position, k]), float(top_probs[layer, position, k]))
            for k in range(top_k)
        ]
        print(f"{'L' + str(layer):>{label_w}} | " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", default="Chemistry. There is a mole in the middle of my")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--position", type=int, default=-1,
                    help="sequence position to inspect; -1 = the next-token prediction")
    ap.add_argument("--gpu", default="1", help="value for CUDA_VISIBLE_DEVICES")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from lift.models.gemma4 import build_model, logit_lens

    model, tokenizer = build_model()

    input_ids = jnp.array(
        [tokenizer.bos_token_id] + tokenizer(args.context, add_special_tokens=False)["input_ids"]
    )
    tokens = [tokenizer.decode([int(t)]) for t in input_ids]
    print(f"context: {args.context!r}")
    print(f"tokens ({len(tokens)}): {tokens}")

    top_ids, top_probs = logit_lens(model, input_ids, top_k=args.top_k)
    render(tokenizer, top_ids, top_probs, args.position, args.top_k)


if __name__ == "__main__":
    main()
