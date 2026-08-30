#

import os
# os.environ["JAX_PLATFORMS"] = "cpu"

import json

import equinox as eqx
import jax
import jax.numpy as jnp
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer


# ── RoPE ─────────────────────────────────────────────────────────────────────

def _rotate_half(x):
    h = x.shape[-1] // 2
    return jnp.concatenate([-x[..., h:], x[..., :h]], axis=-1)

def _apply_rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin

def make_rope_cache(head_dim: int, max_seq: int, theta: float):
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = jnp.outer(jnp.arange(max_seq, dtype=jnp.float32), inv_freq)
    freqs = jnp.concatenate([freqs, freqs], axis=-1)  # (max_seq, head_dim)
    return jnp.cos(freqs), jnp.sin(freqs)


# ── modules ──────────────────────────────────────────────────────────────────

class RMSNorm(eqx.Module):
    weight: jax.Array
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = jnp.ones(dim)
        self.eps = eps

    def __call__(self, x: jax.Array) -> jax.Array:
        rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class Attention(eqx.Module):
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, hidden: int, num_heads: int, num_kv_heads: int, key):
        hd = hidden // num_heads
        ks = jax.random.split(key, 4)
        self.q_proj = eqx.nn.Linear(hidden, num_heads * hd, use_bias=False, key=ks[0])
        self.k_proj = eqx.nn.Linear(hidden, num_kv_heads * hd, use_bias=False, key=ks[1])
        self.v_proj = eqx.nn.Linear(hidden, num_kv_heads * hd, use_bias=False, key=ks[2])
        self.o_proj = eqx.nn.Linear(num_heads * hd, hidden, use_bias=False, key=ks[3])
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hd

    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
        S = x.shape[0]
        q = (x @ self.q_proj.weight.T).reshape(S, self.num_heads, self.head_dim)
        k = (x @ self.k_proj.weight.T).reshape(S, self.num_kv_heads, self.head_dim)
        v = (x @ self.v_proj.weight.T).reshape(S, self.num_kv_heads, self.head_dim)

        c, s = cos[:S, None, :], sin[:S, None, :]  # (S, 1, head_dim)
        q = _apply_rope(q, c, s)
        k = _apply_rope(k, c, s)

        # GQA: tile k/v to match num_heads
        groups = self.num_heads // self.num_kv_heads
        k = jnp.repeat(k, groups, axis=1)
        v = jnp.repeat(v, groups, axis=1)

        q, k, v = (t.transpose(1, 0, 2) for t in (q, k, v))  # (H, S, D)
        scores = jnp.einsum("hqd,hkd->hqk", q, k) * (self.head_dim ** -0.5)
        mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        scores = jnp.where(mask[None], scores, jnp.finfo(scores.dtype).min)
        out = jnp.einsum("hqk,hkd->hqd", jax.nn.softmax(scores, axis=-1), v)

        out = out.transpose(1, 0, 2).reshape(S, -1)  # (S, H*D)
        return out @ self.o_proj.weight.T


class MLP(eqx.Module):
    gate_proj: eqx.nn.Linear
    up_proj: eqx.nn.Linear
    down_proj: eqx.nn.Linear

    def __init__(self, hidden: int, intermediate: int, key):
        ks = jax.random.split(key, 3)
        self.gate_proj = eqx.nn.Linear(hidden, intermediate, use_bias=False, key=ks[0])
        self.up_proj   = eqx.nn.Linear(hidden, intermediate, use_bias=False, key=ks[1])
        self.down_proj = eqx.nn.Linear(intermediate, hidden, use_bias=False, key=ks[2])

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.silu(x @ self.gate_proj.weight.T) * (x @ self.up_proj.weight.T) @ self.down_proj.weight.T


class DecoderLayer(eqx.Module):
    self_attn: Attention
    mlp: MLP
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm

    def __init__(self, hidden: int, num_heads: int, num_kv_heads: int,
                 intermediate: int, rms_eps: float, key):
        ks = jax.random.split(key, 2)
        self.self_attn = Attention(hidden, num_heads, num_kv_heads, ks[0])
        self.mlp = MLP(hidden, intermediate, ks[1])
        self.input_layernorm = RMSNorm(hidden, rms_eps)
        self.post_attention_layernorm = RMSNorm(hidden, rms_eps)

    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Llama(eqx.Module):
    embed: jax.Array  # (vocab, hidden)
    layers: list
    norm: RMSNorm
    cos_cache: jax.Array
    sin_cache: jax.Array

    def __init__(self, cfg: dict, key):
        H = cfg["hidden_size"]
        ks = jax.random.split(key, cfg["num_hidden_layers"] + 1)
        self.embed = jax.random.normal(ks[0], (cfg["vocab_size"], H)) * 0.02
        self.layers = [
            DecoderLayer(H, cfg["num_attention_heads"], cfg["num_key_value_heads"],
                         cfg["intermediate_size"], cfg["rms_norm_eps"], ks[i + 1])
            for i in range(cfg["num_hidden_layers"])
        ]
        self.norm = RMSNorm(H, cfg["rms_norm_eps"])
        hd = H // cfg["num_attention_heads"]
        self.cos_cache, self.sin_cache = make_rope_cache(
            hd, cfg["max_position_embeddings"], cfg["rope_theta"]
        )

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        x = self.embed[input_ids]  # (seq, hidden)
        for layer in self.layers:
            x = layer(x, self.cos_cache, self.sin_cache)
        x = self.norm(x)
        return x @ self.embed.T  # tied lm head → (seq, vocab)


# ── weight loading ────────────────────────────────────────────────────────────

def load_weights(model: Llama, paths: list[str]) -> Llama:
    raw = {}
    for path in paths:
        with safe_open(path, framework="numpy") as f:
            for k in f.keys():
                raw[k] = jnp.array(f.get_tensor(k))

    model = eqx.tree_at(lambda m: m.embed, model, raw["model.embed_tokens.weight"])

    for i in range(len(model.layers)):
        p = f"model.layers.{i}"
        attn = f"{p}.self_attn"
        model = eqx.tree_at(lambda m, i=i: m.layers[i].self_attn.q_proj.weight,              model, raw[f"{attn}.q_proj.weight"])
        model = eqx.tree_at(lambda m, i=i: m.layers[i].self_attn.k_proj.weight,              model, raw[f"{attn}.k_proj.weight"])
        model = eqx.tree_at(lambda m, i=i: m.layers[i].self_attn.v_proj.weight,              model, raw[f"{attn}.v_proj.weight"])
        model = eqx.tree_at(lambda m, i=i: m.layers[i].self_attn.o_proj.weight,              model, raw[f"{attn}.o_proj.weight"])
        model = eqx.tree_at(lambda m, i=i: m.layers[i].mlp.gate_proj.weight,                 model, raw[f"{p}.mlp.gate_proj.weight"])
        model = eqx.tree_at(lambda m, i=i: m.layers[i].mlp.up_proj.weight,                   model, raw[f"{p}.mlp.up_proj.weight"])
        model = eqx.tree_at(lambda m, i=i: m.layers[i].mlp.down_proj.weight,                 model, raw[f"{p}.mlp.down_proj.weight"])
        model = eqx.tree_at(lambda m, i=i: m.layers[i].input_layernorm.weight,               model, raw[f"{p}.input_layernorm.weight"])
        model = eqx.tree_at(lambda m, i=i: m.layers[i].post_attention_layernorm.weight,      model, raw[f"{p}.post_attention_layernorm.weight"])

    model = eqx.tree_at(lambda m: m.norm.weight, model, raw["model.norm.weight"])
    return model


# ── embedding space analysis ─────────────────────────────────────────────────

def save_initial_embeddings(model: Llama, path: str = "embed_initial.npy"):
    """Save model.embed — the raw (vocab_size, hidden) token embedding matrix."""
    import numpy as np
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, np.array(model.embed.astype(jnp.float32)))
    print(f"saved {model.embed.shape} → {path}")


def _attn_with_mask(attn: Attention, x: jax.Array, cos: jax.Array, sin: jax.Array, mask: jax.Array) -> jax.Array:
    S = x.shape[0]
    q = (x @ attn.q_proj.weight.T).reshape(S, attn.num_heads, attn.head_dim)
    k = (x @ attn.k_proj.weight.T).reshape(S, attn.num_kv_heads, attn.head_dim)
    v = (x @ attn.v_proj.weight.T).reshape(S, attn.num_kv_heads, attn.head_dim)
    c, s = cos[:, None, :], sin[:, None, :]
    q = _apply_rope(q, c, s)
    k = _apply_rope(k, c, s)
    groups = attn.num_heads // attn.num_kv_heads
    k = jnp.repeat(k, groups, axis=1)
    v = jnp.repeat(v, groups, axis=1)
    q, k, v = (t.transpose(1, 0, 2) for t in (q, k, v))
    scores = jnp.einsum("hqd,hkd->hqk", q, k) * (attn.head_dim ** -0.5)
    scores = jnp.where(mask[None], scores, jnp.finfo(scores.dtype).min)
    out = jnp.einsum("hqk,hkd->hqd", jax.nn.softmax(scores, axis=-1), v)
    return out.transpose(1, 0, 2).reshape(S, -1) @ attn.o_proj.weight.T


@eqx.filter_jit
def _forward_masked(model: Llama, input_ids: jax.Array, mask: jax.Array) -> jax.Array:
    x = model.embed[input_ids]
    S = x.shape[0]
    cos, sin = model.cos_cache[:S], model.sin_cache[:S]
    for layer in model.layers:
        x = x + _attn_with_mask(layer.self_attn, layer.input_layernorm(x), cos, sin, mask)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return model.norm(x)


def get_contextual_embeddings(
    model: Llama,
    context_ids: jax.Array,
    chunk_size: int = 4096,
    path: str = "embed_contextual.npy",
):
    """
    For every token in the vocabulary, compute its hidden state when it follows context_ids.
    Saves a (vocab_size, hidden) matrix — the contextual embedding space — to path.

    Each vocab token attends to the full context but not to other vocab tokens in the chunk,
    so every row is an independent context-conditioned representation.
    """
    import numpy as np

    V, H = model.embed.shape
    n = context_ids.shape[0]
    out = np.zeros((V, H), dtype=np.float32)

    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        C = end - start
        chunk_ids = jnp.arange(start, end)
        input_ids = jnp.concatenate([context_ids, chunk_ids])

        # mask: context is causal; each vocab token attends to all context + only itself
        top = jnp.concatenate([jnp.tril(jnp.ones((n, n), jnp.bool_)),
                                jnp.zeros((n, C), jnp.bool_)], axis=1)
        bot = jnp.concatenate([jnp.ones((C, n), jnp.bool_),
                                jnp.eye(C, dtype=jnp.bool_)], axis=1)
        mask = jnp.concatenate([top, bot], axis=0)  # (n+C, n+C)

        hidden = _forward_masked(model, input_ids, mask)  # (n+C, H)
        out[start:end] = np.array(hidden[n:])

        if start % (chunk_size * 20) == 0:
            print(f"  {end}/{V}")

    np.save(path, out)
    print(f"saved contextual embeddings ({V}, {H}) → {path}")
    return out


# ── model construction ───────────────────────────────────────────────────────

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def build_model(seed: int = 0):
    """Return a JAX SmolLM2-135M model and tokenizer.

    Mirrors lift.models.gemma4.build_model so experiments can take either
    model without special-casing.
    """
    config_path = hf_hub_download(MODEL_ID, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    print("building model...")
    model = Llama(cfg, jax.random.PRNGKey(seed))
    print("loading weights...")
    model = load_weights(model, [hf_hub_download(MODEL_ID, "model.safetensors")])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    return model, tokenizer
