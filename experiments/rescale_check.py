#!/usr/bin/env python3
"""Check what rescaling does to one token's embedding.

Prints the initial and rescaled lengths for a token alongside its cosine
similarity to the initial embedding, before and after rescaling. The two
cosines should agree: rescaling changes magnitude only, so it is the
directional change that the rest of the study measures.

Was analysis() in main.py.

Usage:
    python experiments/rescale_check.py --token 800
"""

import argparse

import jax.numpy as jnp

from lift.embeddings import contextual, initial, initial_scales
from lift.geometry.metrics import cosine_similarity
from lift.study import PREFIX_FILES


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token", type=int, default=800)
    ap.add_argument("--step", type=int, default=1, help="context step, 1..9")
    args = ap.parse_args()

    e_init = initial()
    scales = initial_scales()

    e_cont = contextual(PREFIX_FILES[args.step - 1])
    e_cont_scaled = contextual(PREFIX_FILES[args.step - 1], scales)

    i = args.token
    print(f"initial length:  {scales[i]}")
    print(f"rescaled length: {jnp.linalg.norm(e_cont_scaled[i], axis=-1)}")
    print(f"cosine(init, contextual):          {cosine_similarity(e_init[i], e_cont[i])}")
    print(f"cosine(init, contextual rescaled): {cosine_similarity(e_init[i], e_cont_scaled[i])}")


if __name__ == "__main__":
    main()
