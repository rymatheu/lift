#!/usr/bin/env python3
"""Print the token ids for a list of words.

Words that do not tokenize to a single token are flagged -- the study tracks
individual rows of the embedding table, so a multi-token word cannot be
followed through the context.

Was main4() in main.py.

Usage:
    python experiments/token_ids.py
    python experiments/token_ids.py --words " garden" " back" " mole"
"""

import argparse

from lift.study import RELATED_WORDS, WORDS_OF_INTEREST


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--words", nargs="+",
                    help="words to look up (default: the study's words of interest)")
    args = ap.parse_args()

    from lift.models.tokenizer import load_tokenizer

    words = args.words or [w for w, _ in WORDS_OF_INTEREST] + RELATED_WORDS
    tokenizer = load_tokenizer()

    for w in words:
        ids = tokenizer(w, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            print(f'"{w}": {ids}  <- not a single token')
        else:
            print(f'"{w}": {ids[0]}')


if __name__ == "__main__":
    main()
