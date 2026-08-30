# lift

Probing how transformer context shifts token embedding spaces.

The question: when a transformer reads a context, what happens to the
*entire* embedding space, not just to the tokens present in the text? For
every token in the vocabulary, `lift` computes the hidden state that token
would take if it followed the context, giving a (vocab, hidden) matrix per
context prefix — the embedding space as the model sees it at that moment.
Watching that matrix move as the context accumulates is the study.

The running example is the sentence

> There is a mole in the middle of ...

which stays ambiguous until its last word: a mole is an animal in a
*garden*, a blemish on your *back*, a spy in an *operation*, or 6.02 × 10²³
of something in an *equation*. `lift/study.py` pins the four senses and the
context prefixes; the experiments track what the model does to them.

Loads SmolLM2-135M, Gemma 4 E2B or Gemma 4 12B via Equinox + safetensors.

| `--model` | module | notes |
|---|---|---|
| `gemma4` | `lift.models.gemma4` | Gemma 4 E2B, per-layer embeddings, hardcoded config |
| `gemma4-12b` | `lift.models.gemma4_12b` | Gemma 4 12B dense, all shapes read from `config.json` |
| `smollm2` | `lift.models.smollm2` | SmolLM2-135M |

## install

```bash
pip install -e .
```

UMAP reduction runs on the GPU through cuML, and the point-cloud viewers
need their own stack:

```bash
pip install --extra-index-url=https://pypi.nvidia.com -e '.[gpu,viz]'
```

`requirements.txt` is a full freeze of the development environment if you
need to reproduce it exactly. To fetch the third-party Potree viewer used by
`lift-potree`:

```bash
./scripts/fetch_potree.sh
```

## usage

```python
import jax
import jax.numpy as jnp
from lift.models.gemma4 import build_model

model, tokenizer = build_model()
context = "There is a mole"
context_ids = jnp.array(
    [tokenizer.bos_token_id] + tokenizer(context, add_special_tokens=False)["input_ids"]
)
logits = model(context_ids)
next_token_prob = jax.nn.softmax(logits[-1])
top_token_id = jnp.argmax(next_token_prob)
print(tokenizer.decode([top_token_id]))  # decoder expects an array of token ids -> " in"
```

## layout

```
src/lift/          the library
  models/          gemma4 (E2B), gemma4_12b, smollm2, tokenizer
                   layers.py   RMSNorm + RoPE shared by the Gemma models
                   probing.py  the vocabulary probe: mask and RoPE positions
  geometry/        intrinsic dimension (VGT), UMAP, PCA, metrics, rigid motions
  viz/             matplotlib plots, gif assembly, pyvista + Potree viewers
  study.py         the constants describing the mole investigation
  embeddings.py    loading matrices and rescaling them to initial lengths
  paths.py         every data / output / vendor path in one place
experiments/       one script per experiment, each with --help
tests/             tests for the geometry helpers
lorenz/            an unrelated side project (Lorenz attractor + transformer)
scripts/           fetch_potree.sh
docs/contexts/     which context maps to which index in the data
```

Nothing writes next to the source. Everything lands in three directories,
all gitignored, and all overridable with `LIFT_DATA_DIR`, `LIFT_OUTPUT_DIR`
and `LIFT_VENDOR_DIR`:

```
data/      embedding matrices, reduced point clouds
outputs/   figures, gifs, frame sequences, Potree octrees
vendor/    the Potree viewer
```

## experiments

Every script takes `--help`. The usual order:

```bash
# 1. dump the initial and contextual embedding spaces (slow, GPU, ~2 GB per matrix)
python experiments/dump_embeddings.py --model gemma4
python experiments/dump_embeddings.py --model gemma4-12b   # the larger model
python experiments/extract_context_embeddings.py            # one matrix per prefix
python experiments/extract_context_embeddings.py --branch   # one per continuation

# 2. reduce to 3D
python experiments/reduce_embeddings.py --set branch

# 3. look at it
python experiments/eigenvalues.py                  # sense-word spectrum through context
python experiments/intrinsic_dim.py radius         # VGT dimension estimates
python experiments/embedding_lengths.py --step 10  # magnitude distribution
python experiments/neighbors.py --token 7972       # nearest neighbours, cosine vs L2
python experiments/persistence.py                  # H0 persistence of a neighbourhood
python experiments/morph_gif.py --consecutive 9 --stitch
lift-potree data/reduced/10.npy                    # interactive viewer
```

`experiments/logit_lens.py` does not run: it calls `logit_lens` in
`lift.models.gemma4`, which was never written. The table renderer is
complete; the missing piece is a forward pass returning per-layer top-k
predictions.

## tests

```bash
pytest
```

`tests/test_gemma4_12b.py` checks the 12B forward pass against transformers'
own Gemma4Unified implementation. That comparison needs torch and skips
without it:

```bash
pip install -e '.[reference]'
```

## a note on the probe

`get_contextual_embeddings` appends a chunk of vocabulary tokens to the
context in one forward pass, masked so each probe sees the context and
itself but not the other probes. Every probe stands in for the same slot --
the next token after the context -- so they must all share RoPE position
`n`. `lift/models/probing.py` keeps the mask and the positions together for
that reason; letting positions run on as `arange(n + C)` silently corrupts
every row of a chunk except the first.

The E2B path does not restrict its sliding-attention layers to
`SLIDING_WINDOW`, so contexts longer than 512 tokens will not match what the
model really computes. `gemma4_12b` honours the window.
