import os
# os.environ["JAX_PLATFORMS"] = "cpu"

import json
import equinox as eqx
import jax
import jax.numpy as jnp
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

from lift.models.layers import RMSNorm, apply_rope as _apply_rope, make_rope_cache as _make_rope_cache, rotate_half as _rotate_half
from lift.models.probing import contextual_embeddings

MODEL_ID = "principled-intelligence/gemma-4-E2B-it-text-only"

# ── Config constants ──────────────────────────────────────────────────────────
HIDDEN          = 1536
SLIDING_HD      = 256    # head_dim for sliding-attention layers
FULL_HD         = 512    # head_dim for full-attention layers
NUM_HEADS       = 8
NUM_KV_HEADS    = 1
INTERMEDIATE_S  = 6144   # MLP width for layers 0-14
INTERMEDIATE_L  = 12288  # MLP width for layers 15-34 (double-wide)
PER_LAYER_DIM   = 256    # hidden_size_per_layer_input
NUM_LAYERS      = 35
VOCAB           = 262144
SLIDING_WINDOW  = 512
THETA_SLIDING   = 10_000.0
THETA_FULL      = 1_000_000.0
PARTIAL_ROT     = 0.25   # fraction of full-attention head_dim that gets rotated
RMS_EPS         = 1e-6
LOGIT_SOFTCAP   = 30.0

# Layer pattern: four sliding then one full, repeated seven times.
LAYER_TYPES = (["sliding"] * 4 + ["full"]) * 7  # len == 35
FIRST_SHARED    = 15     # layers >= this reuse KV from layers 13 / 14
LAST_SLIDING_NS = 13     # last non-shared sliding layer (stores KV)
LAST_FULL_NS    = 14     # last non-shared full layer (stores KV)


# ── Modules ───────────────────────────────────────────────────────────────────

class Attention(eqx.Module):
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear
    q_norm: RMSNorm
    k_norm: RMSNorm
    head_dim:  int  = eqx.field(static=True)
    is_sliding: bool = eqx.field(static=True)

    def __init__(self, is_sliding: bool, key):
        hd = SLIDING_HD if is_sliding else FULL_HD
        ks = jax.random.split(key, 4)
        self.q_proj = eqx.nn.Linear(HIDDEN, NUM_HEADS * hd,    use_bias=False, key=ks[0])
        self.k_proj = eqx.nn.Linear(HIDDEN, NUM_KV_HEADS * hd, use_bias=False, key=ks[1])
        self.v_proj = eqx.nn.Linear(HIDDEN, NUM_KV_HEADS * hd, use_bias=False, key=ks[2])
        self.o_proj = eqx.nn.Linear(NUM_HEADS * hd, HIDDEN,    use_bias=False, key=ks[3])
        self.q_norm = RMSNorm(hd)
        self.k_norm = RMSNorm(hd)
        self.head_dim   = hd
        self.is_sliding = is_sliding

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        kv_override: tuple[jax.Array, jax.Array] | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """Returns (attn_output, (k_rope, v)) for optional KV sharing."""
        S  = x.shape[0]
        hd = self.head_dim

        q = (x @ self.q_proj.weight.T).reshape(S, NUM_HEADS, hd)
        q = self.q_norm(q)                          # RMSNorm over HD dim
        q = _apply_rope(q, cos[:S], sin[:S])

        if kv_override is not None:
            k_rope, v = kv_override
        else:
            k = (x @ self.k_proj.weight.T).reshape(S, NUM_KV_HEADS, hd)
            k = self.k_norm(k)
            k_rope = _apply_rope(k, cos[:S], sin[:S])

            v = (x @ self.v_proj.weight.T).reshape(S, NUM_KV_HEADS, hd)
            v_rms = jnp.sqrt(jnp.mean(v * v, axis=-1, keepdims=True) + RMS_EPS)
            v = v / v_rms                           # V norm: no learnable scale

        # GQA: tile K,V from NUM_KV_HEADS to NUM_HEADS
        groups = NUM_HEADS // NUM_KV_HEADS
        k_exp = jnp.repeat(k_rope, groups, axis=1)  # (S, H, HD)
        v_exp = jnp.repeat(v,      groups, axis=1)

        q_t = q.transpose(1, 0, 2)    # (H, S, HD)
        k_t = k_exp.transpose(1, 0, 2)
        v_t = v_exp.transpose(1, 0, 2)

        # scaling = 1.0 because Q/K are unit-norm after q_norm/k_norm
        scores = jnp.einsum("hqd,hkd->hqk", q_t, k_t)

        mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        if self.is_sliding:
            local = jnp.triu(jnp.ones((S, S), dtype=jnp.bool_), -(SLIDING_WINDOW - 1))
            mask  = mask & local

        scores = jnp.where(mask[None], scores, jnp.finfo(scores.dtype).min)
        out = jnp.einsum("hqk,hkd->hqd", jax.nn.softmax(scores, axis=-1), v_t)
        out = out.transpose(1, 0, 2).reshape(S, -1)  # (S, H*HD)

        return out @ self.o_proj.weight.T, (k_rope, v)


class MLP(eqx.Module):
    gate_proj: eqx.nn.Linear
    up_proj:   eqx.nn.Linear
    down_proj: eqx.nn.Linear

    def __init__(self, intermediate: int, key):
        ks = jax.random.split(key, 3)
        self.gate_proj = eqx.nn.Linear(HIDDEN,        intermediate, use_bias=False, key=ks[0])
        self.up_proj   = eqx.nn.Linear(HIDDEN,        intermediate, use_bias=False, key=ks[1])
        self.down_proj = eqx.nn.Linear(intermediate,  HIDDEN,       use_bias=False, key=ks[2])

    def __call__(self, x: jax.Array) -> jax.Array:
        # SwiGLU-style with GELU (approximate = tanh) as activation
        gate = jax.nn.gelu(x @ self.gate_proj.weight.T, approximate=True)
        up   = x @ self.up_proj.weight.T
        return (gate * up) @ self.down_proj.weight.T


class DecoderLayer(eqx.Module):
    self_attn:                Attention
    mlp:                      MLP
    input_layernorm:          RMSNorm
    post_attention_layernorm: RMSNorm
    pre_feedforward_layernorm:  RMSNorm
    post_feedforward_layernorm: RMSNorm
    per_layer_input_gate:     eqx.nn.Linear   # HIDDEN → PER_LAYER_DIM
    per_layer_projection:     eqx.nn.Linear   # PER_LAYER_DIM → HIDDEN
    post_per_layer_input_norm: RMSNorm
    layer_scalar:             jax.Array        # shape (1,)

    def __init__(self, layer_idx: int, key):
        is_sliding   = LAYER_TYPES[layer_idx] == "sliding"
        intermediate = INTERMEDIATE_S if layer_idx < FIRST_SHARED else INTERMEDIATE_L
        k0, k1, k2, k3 = jax.random.split(key, 4)
        self.self_attn                  = Attention(is_sliding, k0)
        self.mlp                        = MLP(intermediate, k1)
        self.input_layernorm            = RMSNorm(HIDDEN)
        self.post_attention_layernorm   = RMSNorm(HIDDEN)
        self.pre_feedforward_layernorm  = RMSNorm(HIDDEN)
        self.post_feedforward_layernorm = RMSNorm(HIDDEN)
        self.per_layer_input_gate       = eqx.nn.Linear(HIDDEN,         PER_LAYER_DIM, use_bias=False, key=k2)
        self.per_layer_projection       = eqx.nn.Linear(PER_LAYER_DIM,  HIDDEN,        use_bias=False, key=k3)
        self.post_per_layer_input_norm  = RMSNorm(HIDDEN)
        self.layer_scalar               = jnp.ones(1)

    def __call__(
        self,
        x: jax.Array,
        pli: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        kv_override: tuple[jax.Array, jax.Array] | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """
        pli: (S, PER_LAYER_DIM) per-layer input for this layer.
        Returns (hidden_state, (k_rope, v)) so the model can implement KV sharing.
        """
        # Attention block (pre-norm + post-norm residual)
        residual = x
        h = self.input_layernorm(x)
        h, kv = self.self_attn(h, cos, sin, kv_override=kv_override)
        h = self.post_attention_layernorm(h)
        x = residual + h

        # MLP block (pre-norm + post-norm residual)
        residual = x
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        x = residual + h

        # Per-layer embedding injection
        residual = x
        h = x @ self.per_layer_input_gate.weight.T   # (S, PER_LAYER_DIM)
        h = jax.nn.gelu(h, approximate=True)
        h = h * pli                                   # elementwise gate by PLE vector
        h = h @ self.per_layer_projection.weight.T    # (S, HIDDEN)
        h = self.post_per_layer_input_norm(h)
        x = residual + h

        x = x * self.layer_scalar
        return x, kv


# ── Model ─────────────────────────────────────────────────────────────────────

class Gemma4(eqx.Module):
    embed:                  jax.Array   # (VOCAB, HIDDEN)
    embed_per_layer:        jax.Array   # (VOCAB, NUM_LAYERS * PER_LAYER_DIM)
    per_layer_model_proj:   jax.Array   # (NUM_LAYERS * PER_LAYER_DIM, HIDDEN)
    per_layer_proj_norm:    RMSNorm     # normalises each PER_LAYER_DIM slice
    layers:                 list
    norm:                   RMSNorm
    cos_sliding:            jax.Array   # (max_seq, SLIDING_HD)
    sin_sliding:            jax.Array
    cos_full:               jax.Array   # (max_seq, FULL_HD)
    sin_full:               jax.Array

    def __init__(self, max_seq: int, raw: dict, key):
        """
        raw: pre-loaded weight dict (numpy arrays from safe_open).
        Large embedding tables are taken directly from raw to avoid double allocation.
        """
        ks = jax.random.split(key, NUM_LAYERS + 1)
        # Large tables: convert from numpy once, no random allocation
        self.embed               = jnp.array(raw["embed_tokens.weight"])
        self.embed_per_layer     = jnp.array(raw["embed_tokens_per_layer.weight"])
        self.per_layer_model_proj = jnp.array(raw["per_layer_model_projection.weight"])
        self.per_layer_proj_norm = RMSNorm(PER_LAYER_DIM)
        self.layers              = [DecoderLayer(i, ks[i]) for i in range(NUM_LAYERS)]
        self.norm                = RMSNorm(HIDDEN)
        self.cos_sliding, self.sin_sliding = _make_rope_cache(max_seq, SLIDING_HD, THETA_SLIDING)
        self.cos_full,    self.sin_full    = _make_rope_cache(
            max_seq, FULL_HD, THETA_FULL, partial_factor=PARTIAL_ROT
        )

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        S = input_ids.shape[0]

        x = self.embed[input_ids] * (HIDDEN ** 0.5)  # (S, HIDDEN)

        # Per-layer embedding pipeline (computed once from initial embeddings)
        tok_pli  = self.embed_per_layer[input_ids] * (PER_LAYER_DIM ** 0.5)  # (S, L*D)
        tok_pli  = tok_pli.reshape(S, NUM_LAYERS, PER_LAYER_DIM)

        proj_pli = (x @ self.per_layer_model_proj.T) * (HIDDEN ** -0.5)      # (S, L*D)
        proj_pli = proj_pli.reshape(S, NUM_LAYERS, PER_LAYER_DIM)
        proj_pli = self.per_layer_proj_norm(proj_pli)  # RMSNorm over last dim

        pli = (proj_pli + tok_pli) * (2.0 ** -0.5)   # (S, NUM_LAYERS, PER_LAYER_DIM)

        # Decoder stack with KV sharing (layers 15-34 reuse KV from layers 13/14)
        shared_kv: dict[str, tuple[jax.Array, jax.Array] | None] = {
            "sliding": None,
            "full":    None,
        }
        for i, layer in enumerate(self.layers):
            lt      = LAYER_TYPES[i]
            cos     = self.cos_sliding if lt == "sliding" else self.cos_full
            sin     = self.sin_sliding if lt == "sliding" else self.sin_full
            kv_ovr  = shared_kv[lt] if i >= FIRST_SHARED else None

            x, kv = layer(x, pli[:, i, :], cos, sin, kv_override=kv_ovr)

            if i == LAST_SLIDING_NS or i == LAST_FULL_NS:
                shared_kv[lt] = kv

        x = self.norm(x)
        logits = x @ self.embed.T                              # tied LM head (S, VOCAB)
        logits = jnp.tanh(logits / LOGIT_SOFTCAP) * LOGIT_SOFTCAP
        return logits


# ── Weight loading ─────────────────────────────────────────────────────────────

def load_weights(model: Gemma4, raw: dict) -> Gemma4:
    """Load per-layer weights into an already-constructed Gemma4.

    The large embedding tables (embed, embed_per_layer, per_layer_model_proj)
    are already set during Gemma4.__init__; this function only loads the smaller
    per-layer weights (norms, attention projections, MLP, PLE gate/proj).
    """
    model = eqx.tree_at(lambda m: m.per_layer_proj_norm.weight, model, jnp.array(raw["per_layer_projection_norm.weight"]))
    model = eqx.tree_at(lambda m: m.norm.weight,                model, jnp.array(raw["norm.weight"]))

    def w(key): return jnp.array(raw[key])

    for i in range(NUM_LAYERS):
        p = f"layers.{i}"
        a = f"{p}.self_attn"
        model = eqx.tree_at(lambda m, i=i: m.layers[i].self_attn.q_proj.weight,            model, w(f"{a}.q_proj.weight"))
        model = eqx.tree_at(lambda m, i=i: m.layers[i].self_attn.k_proj.weight,            model, w(f"{a}.k_proj.weight"))
        model = eqx.tree_at(lambda m, i=i: m.layers[i].self_attn.v_proj.weight,            model, w(f"{a}.v_proj.weight"))
        model = eqx.tree_at(lambda m, i=i: m.layers[i].self_attn.o_proj.weight,            model, w(f"{a}.o_proj.weight"))
        model = eqx.tree_at(lambda m, i=i: m.layers[i].self_attn.q_norm.weight,            model, w(f"{a}.q_norm.weight"))
        model = eqx.tree_at(lambda m, i=i: m.layers[i].self_attn.k_norm.weight,            model, w(f"{a}.k_norm.weight"))
        model = eqx.tree_at(lambda m, i=i: m.layers[i].mlp.gate_proj.weight,               model, w(f"{p}.mlp.gate_proj.weight"))
        model = eqx.tree_at(lambda m, i=i: m.layers[i].mlp.up_proj.weight,                 model, w(f"{p}.mlp.up_proj.weight"))
        model = eqx.tree_at(lambda m, i=i: m.layers[i].mlp.down_proj.weight,               model, w(f"{p}.mlp.down_proj.weight"))
        model = eqx.tree_at(lambda m, i=i: m.layers[i].input_layernorm.weight,             model, w(f"{p}.input_layernorm.weight"))
        model = eqx.tree_at(lambda m, i=i: m.layers[i].post_attention_layernorm.weight,    model, w(f"{p}.post_attention_layernorm.weight"))
        model = eqx.tree_at(lambda m, i=i: m.layers[i].pre_feedforward_layernorm.weight,   model, w(f"{p}.pre_feedforward_layernorm.weight"))
        model = eqx.tree_at(lambda m, i=i: m.layers[i].post_feedforward_layernorm.weight,  model, w(f"{p}.post_feedforward_layernorm.weight"))
        model = eqx.tree_at(lambda m, i=i: m.layers[i].per_layer_input_gate.weight,        model, w(f"{p}.per_layer_input_gate.weight"))
        model = eqx.tree_at(lambda m, i=i: m.layers[i].per_layer_projection.weight,        model, w(f"{p}.per_layer_projection.weight"))
        model = eqx.tree_at(lambda m, i=i: m.layers[i].post_per_layer_input_norm.weight,   model, w(f"{p}.post_per_layer_input_norm.weight"))
        model = eqx.tree_at(lambda m, i=i: m.layers[i].layer_scalar,                       model, w(f"{p}.layer_scalar"))

    return model


# ── Embedding utilities ────────────────────────────────────────────────────────

def save_initial_embeddings(model: Gemma4, path: str = "embed_initial.npy"):
    import numpy as np
    np.save(path, np.array(model.embed.astype(jnp.float32)))
    print(f"saved {model.embed.shape} → {path}")


@eqx.filter_jit
def _forward_masked(
    model: Gemma4,
    input_ids: jax.Array,
    mask: jax.Array,
    position_ids: jax.Array | None = None,
) -> jax.Array:
    """Forward pass with a custom attention mask (no causal or sliding-window enforcement).

    position_ids: (S,) RoPE position per token. Defaults to arange(S) (normal
    sequential positions). Pass explicit positions when tokens in input_ids
    don't represent a single contiguous sequence — e.g. in
    get_contextual_embeddings, every appended probe token stands in for "the
    next token after the context" and must share one position, not increment.
    """
    S = input_ids.shape[0]
    if position_ids is None:
        position_ids = jnp.arange(S)
    x = model.embed[input_ids] * (HIDDEN ** 0.5)

    tok_pli  = model.embed_per_layer[input_ids] * (PER_LAYER_DIM ** 0.5)
    tok_pli  = tok_pli.reshape(S, NUM_LAYERS, PER_LAYER_DIM)
    proj_pli = (x @ model.per_layer_model_proj.T) * (HIDDEN ** -0.5)
    proj_pli = proj_pli.reshape(S, NUM_LAYERS, PER_LAYER_DIM)
    proj_pli = model.per_layer_proj_norm(proj_pli)
    pli = (proj_pli + tok_pli) * (2.0 ** -0.5)

    shared_kv: dict = {"sliding": None, "full": None}

    for i, layer in enumerate(model.layers):
        lt  = LAYER_TYPES[i]
        cos = model.cos_sliding if lt == "sliding" else model.cos_full
        sin = model.sin_sliding if lt == "sliding" else model.sin_full

        kv_ovr = shared_kv[lt] if i >= FIRST_SHARED else None

        # Inline layer forward with custom mask
        residual = x
        h = layer.input_layernorm(x)

        hd = SLIDING_HD if lt == "sliding" else FULL_HD
        q = (h @ layer.self_attn.q_proj.weight.T).reshape(S, NUM_HEADS, hd)
        q = layer.self_attn.q_norm(q)
        q = _apply_rope(q, cos[position_ids], sin[position_ids])

        if kv_ovr is not None:
            k_rope, v = kv_ovr
        else:
            k = (h @ layer.self_attn.k_proj.weight.T).reshape(S, NUM_KV_HEADS, hd)
            k = layer.self_attn.k_norm(k)
            k_rope = _apply_rope(k, cos[position_ids], sin[position_ids])
            v = (h @ layer.self_attn.v_proj.weight.T).reshape(S, NUM_KV_HEADS, hd)
            v_rms = jnp.sqrt(jnp.mean(v * v, axis=-1, keepdims=True) + RMS_EPS)
            v = v / v_rms

        groups = NUM_HEADS // NUM_KV_HEADS
        k_exp = jnp.repeat(k_rope, groups, axis=1)
        v_exp = jnp.repeat(v,      groups, axis=1)

        q_t, k_t, v_t = (t.transpose(1, 0, 2) for t in (q, k_exp, v_exp))
        scores = jnp.einsum("hqd,hkd->hqk", q_t, k_t)
        scores = jnp.where(mask[None], scores, jnp.finfo(scores.dtype).min)
        attn_out = jnp.einsum("hqk,hkd->hqd", jax.nn.softmax(scores, axis=-1), v_t)
        attn_out = attn_out.transpose(1, 0, 2).reshape(S, -1) @ layer.self_attn.o_proj.weight.T

        h = layer.post_attention_layernorm(attn_out)
        x = residual + h

        residual = x
        h = layer.pre_feedforward_layernorm(x)
        h = layer.mlp(h)
        h = layer.post_feedforward_layernorm(h)
        x = residual + h

        residual = x
        h = x @ layer.per_layer_input_gate.weight.T
        h = jax.nn.gelu(h, approximate=True)
        h = h * pli[:, i, :]
        h = h @ layer.per_layer_projection.weight.T
        h = layer.post_per_layer_input_norm(h)
        x = residual + h
        x = x * layer.layer_scalar

        if i == LAST_SLIDING_NS or i == LAST_FULL_NS:
            shared_kv[lt] = (k_rope, v)

    return model.norm(x)


def get_contextual_embeddings(
    model: Gemma4,
    context_ids: jax.Array,
    chunk_size: int = 2048,
    path: str = "embed_contextual.npy",
    save: bool = True,
    token_ids=None,
):
    """Context-conditioned embedding for every vocabulary token.

    Each vocab token attends to the full context and only to itself (not to
    other vocab tokens in the chunk), producing independent context-conditioned
    representations. See lift.models.probing for the mask and the RoPE
    positions, which are shared with the 12B implementation.

    Note: the sliding-attention layers are not restricted to SLIDING_WINDOW
    here, so a context longer than 512 tokens will not reproduce the model's
    real behaviour. lift.models.gemma4_12b honours the window.
    """
    V, H = model.embed.shape

    def forward(input_ids, mask, position_ids):
        return _forward_masked(model, input_ids, mask, position_ids=position_ids)

    return contextual_embeddings(
        forward, context_ids,
        vocab_size=V, hidden_size=H,
        chunk_size=chunk_size, path=path, save=save,
        sliding_window=None, token_ids=token_ids,
    )


def build_model(seed: int = 0):
    """Return a JAX Gemma 4 E2B model and tokenizer.

    Example usage:
        model, tokenizer = build_model()
        context = "There is a mole"
        context_ids = jnp.array([tokenizer.bos_token_id] + tokenizer(context, add_special_tokens=False)["input_ids"])
        logits = model(context_ids)
        next_token_prob = jax.nn.softmax(logits[-1])

    """

    config_path = hf_hub_download(MODEL_ID, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    wt_path = hf_hub_download(MODEL_ID, "model.safetensors")
    print("loading weights from safetensors...")
    raw = {}
    with safe_open(wt_path, framework="numpy") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k)

    print("building model...")
    model = Gemma4(cfg["max_position_embeddings"], raw, jax.random.PRNGKey(0))
    print("injecting per-layer weights...")
    model = load_weights(model, raw)
    del raw  # free numpy copies

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    return model, tokenizer
