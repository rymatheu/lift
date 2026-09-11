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

UMAP needs a backend. On an NVIDIA card, cuML from the NVIDIA index; anywhere
else, umap-learn on the CPU:

```bash
pip install --extra-index-url=https://pypi.nvidia.com -e '.[gpu]'   # CUDA
pip install -e '.[cpu]'                                             # CPU
```

The point-cloud viewers are a third extra, and `scripts/fetch_potree.sh`
downloads the Potree assets they serve:

```bash
pip install -e '.[viz]'
./scripts/fetch_potree.sh
```

`requirements.txt` is a full freeze of the development environment if you need
to reproduce it exactly.

### model weights

You do not re-download them. `build_model()` goes through
`huggingface_hub`, which downloads a file only if it is not already in the
local cache — `~/.cache/huggingface/hub` by default, about 2 GB for Gemma 4
E2B. Later runs read from there.

```bash
export HF_HOME=/big/disk/hf        # put the cache somewhere else
export HF_HUB_OFFLINE=1            # never touch the network; cache or fail
```

What *does* happen on every `build_model()` is converting safetensors into
JAX arrays. That cost is per process, not per download, which is the main
argument for working in a notebook or a long-lived session rather than
re-running a script.

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
notebooks/         probe_to_potree.ipynb, the pipeline end to end
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

## from a context to a point cloud

The full path, for one context. `notebooks/probe_to_potree.ipynb` is this
same walk with the output visible at each step, and it runs on a laptop
because it probes a few thousand tokens rather than all 262k.

### 1. probe

For every token in the vocabulary, compute the hidden state it would take
*if it came next* after the context. One matrix per context prefix:

```bash
python experiments/extract_context_embeddings.py \
    --context "There is a mole in the middle of"
```

That writes `data/embeddings/contexts/<subdir>/1-bos-embed.npy`,
`2-bosThere-embed.npy`, and so on — one file per prefix, each
`vocab x hidden`. For Gemma 4 E2B that is about **1.5 GB per file**, so nine
prefixes is ~14 GB. Check your disk first.

To branch the finished context by one more word instead of walking its
prefixes:

```bash
python experiments/extract_context_embeddings.py --branch
```

Branch files continue the numbering after the prefixes (10, 11, ...), so the
two runs together form one sequence.

### 2. reduce to 3D

```bash
python experiments/reduce_embeddings.py --set prefix
python experiments/reduce_embeddings.py --set branch
```

Writes `data/reduced/1.npy` ... `data/reduced/18.npy` and a matplotlib
preview of each under `outputs/reduced/`. Add `--backend umap-learn` to force
the CPU path, or `--random-state 0` for a reproducible layout.

### 3. view

```bash
lift-potree data/reduced/10.npy
```

Builds a Potree octree next to the `.npy`, serves it, and prints a URL.
Click a point and the overlay names the token; the search box takes a word
and flies to it. Re-serving an octree you already built skips the conversion:

```bash
lift-potree --serve-only --out-dir data/reduced/10_potree
```

`lift-pointcloud` is the lighter alternative — pyvista, no converter needed,
fine up to a few hundred thousand points.

### PotreeConverter

`lift-potree` needs two things in `vendor/potree/`. `scripts/fetch_potree.sh`
gets the **viewer assets**. The **converter binary** is a separate C++ build
and is not fetched for you:

```bash
git clone https://github.com/potree/PotreeConverter
cd PotreeConverter && mkdir build && cd build
cmake .. && make -j
cp PotreeConverter <repo>/vendor/potree/PotreeConverter
```

It links against laszip; if the binary cannot find `liblaszip.so` at run
time, put it next to the binary — `lift-potree` adds that directory to
`LD_LIBRARY_PATH` before calling it.

Without the converter you can still use `--serve-only` on an octree built
elsewhere, and `lift-pointcloud` and the matplotlib previews work regardless.

## on a MacBook

Everything except CUDA works. The honest summary: **run JAX on the CPU**.

| | on Apple Silicon |
|---|---|
| JAX | CPU only in practice — see below |
| UMAP | `umap-learn` (`.[cpu]`), not cuML |
| Potree viewer | works; build PotreeConverter from source |
| `lift-pointcloud` | works |

```bash
pip install -e '.[cpu,viz]'
./scripts/fetch_potree.sh
LIFT_NOTEBOOK_SMOKE=1 jupyter lab notebooks/probe_to_potree.ipynb   # plumbing check
jupyter lab notebooks/probe_to_potree.ipynb                          # the real thing
```

### about jax-metal

Apple's `jax-metal` plugin is not a dependable path. As of late 2025 the JAX
maintainers closed the open jax-metal issues citing no active development,
and the newest macOS it supports is Sonoma. Two community projects have
picked it up — `metaljax`, a PJRT plugin passing ~99.5% of the JAX 0.11 test
suite, and `jax-mps`, pinned to `jaxlib==0.9.0` — but both are experimental,
neither supports float64, and both are single-device only.

CPU JAX is the reliable choice, and for this workload it is not as bad as it
sounds: probing is dominated by a few large matmuls over a 262k-row
embedding table, and Apple's Accelerate-backed BLAS handles those reasonably.
Expect a full-vocabulary probe to take hours rather than minutes, which is
why the notebook probes a subset.

If you want to try Metal anyway, do it in a throwaway environment and check
`jax.devices()` before trusting any output:

```bash
python -c "import jax; print(jax.devices(), jax.default_backend())"
```

### memory

The matrices are the real constraint, not the model. One contextual
embedding matrix for Gemma 4 E2B is `262144 x 1536` float32 — about 1.5 GB,
held in RAM before it is written. On a 16 GB machine, probe a subset
(`token_ids=` in `get_contextual_embeddings`, or the notebook's `N_PROBE`)
rather than the whole vocabulary. The 12B is wider still, which is why its
default `chunk_size` is smaller.

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
