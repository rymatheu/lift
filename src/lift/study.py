"""Constants describing the "mole" study.

The running context is

    "There is a mole in the middle of ..."

which is ambiguous until its final word: a mole is an animal in a garden, a
blemish on your back, a spy in an operation, or 6.02e23 of something in an
equation. The study tracks what the transformer does to the embeddings of
those four disambiguating words as the context accumulates.

Token ids are Gemma 4 E2B ids; they do not transfer to SmolLM2.
"""

CONTEXT = "There is a mole in the middle of"

# (token text, gemma4 token id) for the four senses.
WORDS_OF_INTEREST = [
    (" garden",    7972),
    (" back",      1063),
    (" operation", 5585),
    (" equation",  4901),
]

# Semantically nearby words, used to check that the effect is not specific to
# the four above.
RELATED_WORDS = [
    " business", " reaction", " experiment", " yard", " neck",
    " room", " office", " stomach", " beaker", " table", " ditch",
]

# Contextual embeddings for each prefix of CONTEXT, one file per prefix
# length: <bos>, <bos>There, <bos>There is, ...
PREFIX_FILES = [
    "1-bos-embed.npy",
    "2-bosThere-embed.npy",
    "3-bosThere_is-embed.npy",
    "4-bosThere_is_a-embed.npy",
    "5-bosThere_is_a_mole-embed.npy",
    "6-bosThere_is_a_mole_in-embed.npy",
    "7-bosThere_is_a_mole_in_the-embed.npy",
    "8-bosThere_is_a_mole_in_the_middle-embed.npy",
    "9-bosThere_is_a_mole_in_the_middle_of-embed.npy",
]

# The same context continued by one more determiner, branching the ambiguity
# in nine directions.
BRANCH_FILES = [
    "10-bosThere_is_a_mole_in_the_middle_of_the-embed.npy",
    "11-bosThere_is_a_mole_in_the_middle_of_your-embed.npy",
    "12-bosThere_is_a_mole_in_the_middle_of_our-embed.npy",
    "13-bosThere_is_a_mole_in_the_middle_of_this-embed.npy",
    "14-bosThere_is_a_mole_in_the_middle_of_my-embed.npy",
    "15-bosThere_is_a_mole_in_the_middle_of_his-embed.npy",
    "16-bosThere_is_a_mole_in_the_middle_of_her-embed.npy",
    "17bosThere_is_a_mole_in_the_middle_of_that-embed.npy",
    "18-bosThere_is_a_mole_in_the_middle_of_some-embed.npy",
]

BRANCH_WORDS = [" the", " your", " our", " this", " my", " his", " her", " that", " some"]

# Where PREFIX_FILES and BRANCH_FILES live under data/embeddings/contexts/.
CONTEXT_SUBDIR = "There_is_a_mole_in_the_middle_of"
