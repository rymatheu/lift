"""Gemma 4 12B ("unified") in JAX + Equinox, text tower only.

The 12B is a dense model and differs from the E2B implementation in
lift.models.gemma4 in several ways:

  * no per-layer embeddings (PLE). E2B carries an embed_tokens_per_layer
    table, a per-layer projection and a gate in every block; the 12B decoder
    layer is just attention + MLP + the layer scalar.
  * head_dim varies by layer type -- `head_dim` for sliding-attention layers,
    `global_head_dim` for full-attention layers -- and the full layers may
    also carry their own `num_global_key_value_heads`.
  * `attention_k_eq_v` lets full-attention layers reuse the key projection as
    the value projection, so those layers have no v_proj at all.
  * KV sharing is expressed as `num_kv_shared_layers` counting back from the
    end, rather than E2B's hardcoded layer indices.
  * `use_double_wide_mlp` widens the MLP on the KV-sharing layers.
  * logit softcapping is optional (`final_logit_softcapping`, often null).

Every shape is read from the checkpoint's config.json rather than hardcoded,
so this module tracks whatever Google actually shipped instead of a set of
constants transcribed by hand.

Mirrors gemma4.py's public API: build_model(), save_initial_embeddings(),
get_contextual_embeddings().
"""

import json
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

from lift.models.layers import RMSNorm, apply_rope, make_rope_cache
from lift.models.probing import contextual_embeddings, probe_mask

MODEL_ID = "google/gemma-4-12B-it"

SLIDING = "sliding_attention"
FULL = "full_attention"


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Gemma4Config:
    """The subset of config.json this implementation needs.

    Field names and defaults follow transformers' Gemma4UnifiedTextConfig.
    """

    vocab_size: int = 262_144
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 30
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    global_head_dim: int = 512
    num_global_key_value_heads: int | None = None
    rms_norm_eps: float = 1e-6
    sliding_window: int = 1024
    max_position_embeddings: int = 262_144
    layer_types: tuple[str, ...] = ()
    rope_theta_sliding: float = 10_000.0
    rope_theta_full: float = 1_000_000.0
    partial_rotary_factor: float = 0.25
    final_logit_softcapping: float | None = None
    attention_k_eq_v: bool = False
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False

    @classmethod
    def from_dict(cls, cfg: dict) -> "Gemma4Config":
        """Parse a config.json, descending into text_config when present."""
        if "text_config" in cfg:
            cfg = cfg["text_config"]

        n_layers = cfg.get("num_hidden_layers", 30)

        layer_types = cfg.get("layer_types")
        if not layer_types:
            # transformers' default: 5 sliding then 1 full, last layer forced full
            layer_types = [
                SLIDING if (i + 1) % 6 else FULL for i in range(n_layers)
            ]
        layer_types = list(layer_types)
        if layer_types[-1] != FULL:
            layer_types[-1] = FULL

        rope = cfg.get("rope_parameters") or {}
        sliding_rope = rope.get(SLIDING) or {}
        full_rope = rope.get(FULL) or {}

        return cls(
            vocab_size=cfg.get("vocab_size", 262_144),
            hidden_size=cfg.get("hidden_size", 2304),
            intermediate_size=cfg.get("intermediate_size", 9216),
            num_hidden_layers=n_layers,
            num_attention_heads=cfg.get("num_attention_heads", 8),
            num_key_value_heads=cfg.get("num_key_value_heads", 4),
            head_dim=cfg.get("head_dim", 256),
            global_head_dim=cfg.get("global_head_dim", 512),
            num_global_key_value_heads=cfg.get("num_global_key_value_heads"),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            sliding_window=cfg.get("sliding_window", 1024),
            max_position_embeddings=cfg.get("max_position_embeddings", 262_144),
            layer_types=tuple(layer_types),
            rope_theta_sliding=sliding_rope.get("rope_theta", 10_000.0),
            rope_theta_full=full_rope.get("rope_theta", 1_000_000.0),
            partial_rotary_factor=full_rope.get("partial_rotary_factor", 0.25),
            final_logit_softcapping=cfg.get("final_logit_softcapping"),
            attention_k_eq_v=cfg.get("attention_k_eq_v", False),
            num_kv_shared_layers=cfg.get("num_kv_shared_layers", 0),
            use_double_wide_mlp=cfg.get("use_double_wide_mlp", False),
        )

    # ── derived, per layer ────────────────────────────────────────────────

    @property
    def first_kv_shared_layer(self) -> int:
        return self.num_hidden_layers - self.num_kv_shared_layers

    def is_sliding(self, i: int) -> bool:
        return self.layer_types[i] == SLIDING

    def is_kv_shared(self, i: int) -> bool:
        first = self.first_kv_shared_layer
        return first >= 0 and i >= first

    def layer_head_dim(self, i: int) -> int:
        return self.head_dim if self.is_sliding(i) else self.global_head_dim

    def layer_kv_heads(self, i: int) -> int:
        """Full-attention layers may override the KV head count."""
        if (not self.is_sliding(i)
                and self.attention_k_eq_v
                and self.num_global_key_value_heads is not None):
            return self.num_global_key_value_heads
        return self.num_key_value_heads

    def k_eq_v(self, i: int) -> bool:
        """Full-attention layers may reuse the key projection as the value."""
        return self.attention_k_eq_v and not self.is_sliding(i)

    def layer_intermediate(self, i: int) -> int:
        first = self.first_kv_shared_layer
        double = self.use_double_wide_mlp and i >= first > 0
        return self.intermediate_size * (2 if double else 1)

    def stores_shared_kv(self, i: int) -> bool:
        """True for the last non-shared layer of each attention type.

        Those layers' K/V are what every later same-type layer reuses.
        """
        if self.is_kv_shared(i):
            return False
        prev = self.layer_types[:self.first_kv_shared_layer]
        want = self.layer_types[i]
        if want not in prev:
            return False
        return i == len(prev) - 1 - prev[::-1].index(want)


# ── Modules ───────────────────────────────────────────────────────────────────

class Attention(eqx.Module):
    q_proj: jax.Array
    o_proj: jax.Array
    q_norm: RMSNorm
    # Absent on KV-sharing layers, which reuse an earlier layer's K/V.
    k_proj: jax.Array | None
    k_norm: RMSNorm | None
    # Absent when the layer reuses k_proj as the value projection.
    v_proj: jax.Array | None
    v_norm: RMSNorm | None

    n_heads: int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    is_sliding: bool = eqx.field(static=True)
    is_kv_shared: bool = eqx.field(static=True)

    def __call__(self, x, cos, sin, position_ids, mask, kv_override):
        """Returns (attn_out, (k_rope, v)) so callers can share K/V."""
        S = x.shape[0]

        q = (x @ self.q_proj.T).reshape(S, self.n_heads, self.head_dim)
        q = self.q_norm(q)
        q = apply_rope(q, cos[position_ids], sin[position_ids])

        if self.is_kv_shared:
            k_rope, v = kv_override
        else:
            kv_shape = (S, self.n_kv_heads, self.head_dim)
            k_raw = (x @ self.k_proj.T).reshape(kv_shape)
            # k_eq_v layers take the value from the key projection, before
            # k_norm and RoPE are applied to the key.
            v_raw = k_raw if self.v_proj is None else (x @ self.v_proj.T).reshape(kv_shape)

            k_rope = apply_rope(self.k_norm(k_raw), cos[position_ids], sin[position_ids])
            v = self.v_norm(v_raw)

        groups = self.n_heads // self.n_kv_heads
        k_exp = jnp.repeat(k_rope, groups, axis=1)
        v_exp = jnp.repeat(v, groups, axis=1)

        q_t, k_t, v_t = (t.transpose(1, 0, 2) for t in (q, k_exp, v_exp))

        # scaling is 1.0: q_norm / k_norm already normalise Q and K
        scores = jnp.einsum("hqd,hkd->hqk", q_t, k_t)
        scores = jnp.where(mask[None], scores, jnp.finfo(scores.dtype).min)
        out = jnp.einsum("hqk,hkd->hqd", jax.nn.softmax(scores, axis=-1), v_t)
        out = out.transpose(1, 0, 2).reshape(S, -1)

        return out @ self.o_proj.T, (k_rope, v)


class MLP(eqx.Module):
    gate_proj: jax.Array
    up_proj: jax.Array
    down_proj: jax.Array

    def __call__(self, x):
        return (jax.nn.gelu(x @ self.gate_proj.T, approximate=True)
                * (x @ self.up_proj.T)) @ self.down_proj.T


class DecoderLayer(eqx.Module):
    self_attn: Attention
    mlp: MLP
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm
    pre_feedforward_layernorm: RMSNorm
    post_feedforward_layernorm: RMSNorm
    layer_scalar: jax.Array

    def __call__(self, x, cos, sin, position_ids, mask, kv_override):
        residual = x
        h, kv = self.self_attn(self.input_layernorm(x), cos, sin,
                               position_ids, mask, kv_override)
        x = residual + self.post_attention_layernorm(h)

        residual = x
        h = self.mlp(self.pre_feedforward_layernorm(x))
        x = residual + self.post_feedforward_layernorm(h)

        return x * self.layer_scalar, kv


class Gemma4_12B(eqx.Module):
    embed: jax.Array
    layers: list
    norm: RMSNorm
    cos_sliding: jax.Array
    sin_sliding: jax.Array
    cos_full: jax.Array
    sin_full: jax.Array
    config: Gemma4Config = eqx.field(static=True)

    def rope_for(self, i: int):
        if self.config.is_sliding(i):
            return self.cos_sliding, self.sin_sliding
        return self.cos_full, self.sin_full

    def hidden_states(self, input_ids, masks, position_ids):
        """Final normed hidden states, (S, hidden).

        masks: {"sliding": (S, S) bool, "full": (S, S) bool}
        """
        cfg = self.config
        x = self.embed[input_ids] * (cfg.hidden_size ** 0.5)

        shared_kv: dict = {SLIDING: None, FULL: None}

        for i, layer in enumerate(self.layers):
            lt = cfg.layer_types[i]
            cos, sin = self.rope_for(i)
            kv_ovr = shared_kv[lt] if cfg.is_kv_shared(i) else None
            mask = masks["sliding" if lt == SLIDING else "full"]

            x, kv = layer(x, cos, sin, position_ids, mask, kv_ovr)

            if cfg.stores_shared_kv(i):
                shared_kv[lt] = kv

        return self.norm(x)

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        """Ordinary causal forward pass, returning logits (S, vocab)."""
        S = input_ids.shape[0]
        causal = jnp.tril(jnp.ones((S, S), jnp.bool_))
        window = jnp.triu(jnp.ones((S, S), jnp.bool_), -(self.config.sliding_window - 1))
        masks = {"full": causal, "sliding": causal & window}

        h = self.hidden_states(input_ids, masks, jnp.arange(S))
        logits = h @ self.embed.T  # tied LM head

        cap = self.config.final_logit_softcapping
        if cap is not None:
            logits = jnp.tanh(logits / cap) * cap
        return logits


# ── Weight loading ────────────────────────────────────────────────────────────

def _find_prefix(keys) -> str:
    """Locate the text tower inside a checkpoint.

    Text-only re-uploads use bare names ("layers.0..."); the released
    multimodal checkpoints nest them under "model." or
    "model.language_model.".
    """
    suffix = "layers.0.self_attn.q_proj.weight"
    for key in keys:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    raise KeyError(
        "could not find the text decoder in this checkpoint: no key ends with "
        f"{suffix!r}. Keys present: {sorted(keys)[:20]}"
    )


def _require(raw: dict, key: str) -> jnp.ndarray:
    if key not in raw:
        raise KeyError(
            f"missing weight {key!r}. This usually means the checkpoint's "
            "architecture does not match the parsed config.json -- check "
            "attention_k_eq_v, num_kv_shared_layers and layer_types."
        )
    return jnp.asarray(raw[key])


def from_weights(cfg: Gemma4Config, raw: dict) -> Gemma4_12B:
    """Build a model directly from a safetensors weight dict."""
    p = _find_prefix(raw.keys())

    layers = []
    for i in range(cfg.num_hidden_layers):
        lp = f"{p}layers.{i}"
        eps = cfg.rms_norm_eps
        shared = cfg.is_kv_shared(i)

        attn = Attention(
            q_proj=_require(raw, f"{lp}.self_attn.q_proj.weight"),
            o_proj=_require(raw, f"{lp}.self_attn.o_proj.weight"),
            q_norm=_norm_from(raw, f"{lp}.self_attn.q_norm.weight", eps),
            k_proj=None if shared else _require(raw, f"{lp}.self_attn.k_proj.weight"),
            k_norm=None if shared else _norm_from(raw, f"{lp}.self_attn.k_norm.weight", eps),
            v_proj=(None if shared or cfg.k_eq_v(i)
                    else _require(raw, f"{lp}.self_attn.v_proj.weight")),
            # V is normalised without a learnable scale, so there is no weight
            v_norm=None if shared else RMSNorm(cfg.layer_head_dim(i), eps, with_scale=False),
            n_heads=cfg.num_attention_heads,
            n_kv_heads=cfg.layer_kv_heads(i),
            head_dim=cfg.layer_head_dim(i),
            is_sliding=cfg.is_sliding(i),
            is_kv_shared=shared,
        )

        mlp = MLP(
            gate_proj=_require(raw, f"{lp}.mlp.gate_proj.weight"),
            up_proj=_require(raw, f"{lp}.mlp.up_proj.weight"),
            down_proj=_require(raw, f"{lp}.mlp.down_proj.weight"),
        )

        layers.append(DecoderLayer(
            self_attn=attn,
            mlp=mlp,
            input_layernorm=_norm_from(raw, f"{lp}.input_layernorm.weight", eps),
            post_attention_layernorm=_norm_from(raw, f"{lp}.post_attention_layernorm.weight", eps),
            pre_feedforward_layernorm=_norm_from(raw, f"{lp}.pre_feedforward_layernorm.weight", eps),
            post_feedforward_layernorm=_norm_from(raw, f"{lp}.post_feedforward_layernorm.weight", eps),
            layer_scalar=_require(raw, f"{lp}.layer_scalar"),
        ))

    max_seq = cfg.max_position_embeddings
    cos_s, sin_s = make_rope_cache(max_seq, cfg.head_dim, cfg.rope_theta_sliding)
    cos_f, sin_f = make_rope_cache(max_seq, cfg.global_head_dim, cfg.rope_theta_full,
                                   partial_factor=cfg.partial_rotary_factor)

    return Gemma4_12B(
        embed=_require(raw, f"{p}embed_tokens.weight"),
        layers=layers,
        norm=_norm_from(raw, f"{p}norm.weight", cfg.rms_norm_eps),
        cos_sliding=cos_s, sin_sliding=sin_s,
        cos_full=cos_f, sin_full=sin_f,
        config=cfg,
    )


def _norm_from(raw: dict, key: str, eps: float) -> RMSNorm:
    w = _require(raw, key)
    norm = RMSNorm(w.shape[0], eps)
    return eqx.tree_at(lambda n: n.weight, norm, w)


# ── Embedding utilities ───────────────────────────────────────────────────────

def save_initial_embeddings(model: Gemma4_12B, path: str = "embed_initial.npy"):
    np.save(path, np.array(model.embed.astype(jnp.float32)))
    print(f"saved {model.embed.shape} -> {path}")


@eqx.filter_jit
def _forward_masked(model: Gemma4_12B, input_ids, masks, position_ids):
    return model.hidden_states(input_ids, masks, position_ids)


def get_contextual_embeddings(
    model: Gemma4_12B,
    context_ids: jax.Array,
    chunk_size: int = 512,
    path: str = "embed_contextual.npy",
    save: bool = True,
    honour_sliding_window: bool = True,
    token_ids=None,
):
    """Context-conditioned embedding for every vocabulary token.

    Each vocab token attends to the context and to itself only, at the RoPE
    position it would occupy as the next token. See lift.models.probing.

    honour_sliding_window keeps the sliding-attention layers restricted to
    their real window. Turn it off only to reproduce the E2B behaviour, where
    probes see the whole context in every layer.
    """
    cfg = model.config

    if honour_sliding_window:
        def forward(input_ids, masks, position_ids):
            return _forward_masked(model, input_ids, masks, position_ids)
        window = cfg.sliding_window
    else:
        # probing hands back a single mask; every layer type then shares it
        def forward(input_ids, mask, position_ids):
            return _forward_masked(model, input_ids,
                                   {"sliding": mask, "full": mask}, position_ids)
        window = None

    return contextual_embeddings(
        forward,
        context_ids,
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        chunk_size=chunk_size,
        path=path,
        save=save,
        sliding_window=window,
        token_ids=token_ids,
    )


# ── Construction ──────────────────────────────────────────────────────────────

def load_config(model_id: str = MODEL_ID) -> Gemma4Config:
    with open(hf_hub_download(model_id, "config.json")) as f:
        return Gemma4Config.from_dict(json.load(f))


def build_model(model_id: str = MODEL_ID, seed: int = 0):
    """Return a JAX Gemma 4 12B text model and its tokenizer.

    Mirrors lift.models.gemma4.build_model. seed is accepted for signature
    compatibility and unused -- every weight comes from the checkpoint.
    """
    cfg = load_config(model_id)
    print(f"config: {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, "
          f"vocab {cfg.vocab_size}, sliding window {cfg.sliding_window}, "
          f"kv-shared {cfg.num_kv_shared_layers}, k_eq_v {cfg.attention_k_eq_v}")

    print("loading weights from safetensors...")
    raw = {}
    for filename in _weight_files(model_id):
        path = hf_hub_download(model_id, filename)
        with safe_open(path, framework="numpy") as f:
            for k in f.keys():
                # Skip the vision and audio towers; this is the text model.
                if ".vision" in k or ".audio" in k or "embedder" in k:
                    continue
                raw[k] = f.get_tensor(k)

    print("building model...")
    model = from_weights(cfg, raw)
    del raw

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def _weight_files(model_id: str) -> list[str]:
    """Shard filenames, from the safetensors index when the model is sharded."""
    try:
        index_path = hf_hub_download(model_id, "model.safetensors.index.json")
    except Exception:
        return ["model.safetensors"]
    with open(index_path) as f:
        index = json.load(f)
    return sorted(set(index["weight_map"].values()))
