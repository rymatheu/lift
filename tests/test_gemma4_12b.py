"""Tests for the Gemma 4 12B implementation.

The config tests run anywhere. The reference test compares the JAX forward
pass against transformers' own Gemma4Unified model and needs torch:

    pip install -e '.[reference]'
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lift.models.gemma4_12b import FULL, SLIDING, Gemma4Config, from_weights
from lift.models.probing import probe_mask, probe_positions

BASE = dict(
    vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=12,
    num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=16,
    sliding_window=4, max_position_embeddings=64,
)


# ── config parsing ────────────────────────────────────────────────────────────

def test_default_layer_pattern_is_five_sliding_then_one_full():
    cfg = Gemma4Config.from_dict({"num_hidden_layers": 12})
    assert list(cfg.layer_types) == [SLIDING] * 5 + [FULL] + [SLIDING] * 5 + [FULL]


def test_last_layer_is_forced_to_full_attention():
    cfg = Gemma4Config.from_dict({"num_hidden_layers": 3, "layer_types": [SLIDING] * 3})
    assert cfg.layer_types[-1] == FULL


def test_text_config_is_unwrapped():
    cfg = Gemma4Config.from_dict({"text_config": {"hidden_size": 999}})
    assert cfg.hidden_size == 999


def test_head_dim_varies_by_layer_type():
    cfg = Gemma4Config.from_dict({**BASE})
    for i, lt in enumerate(cfg.layer_types):
        assert cfg.layer_head_dim(i) == (8 if lt == SLIDING else 16)


def test_kv_sharing_counts_back_from_the_end():
    cfg = Gemma4Config.from_dict({**BASE, "num_kv_shared_layers": 4})
    assert cfg.first_kv_shared_layer == 8
    assert [i for i in range(12) if cfg.is_kv_shared(i)] == [8, 9, 10, 11]


def test_no_kv_sharing_by_default():
    """No layer reuses another's KV.

    The last layer of each type is still flagged as storing its KV -- the
    reference implementation does the same -- but with nothing sharing, that
    stored state is never read.
    """
    cfg = Gemma4Config.from_dict({**BASE})
    assert not any(cfg.is_kv_shared(i) for i in range(12))


def test_the_storing_layer_is_the_last_unshared_one_of_each_type():
    cfg = Gemma4Config.from_dict({**BASE, "num_kv_shared_layers": 4})
    storing = [i for i in range(12) if cfg.stores_shared_kv(i)]
    # layers 0..7 are unshared: last sliding is 7, last full is 5
    assert storing == [5, 7]
    for i in storing:
        assert not cfg.is_kv_shared(i)


def test_k_eq_v_applies_only_to_full_attention_layers():
    cfg = Gemma4Config.from_dict({**BASE, "attention_k_eq_v": True})
    for i, lt in enumerate(cfg.layer_types):
        assert cfg.k_eq_v(i) == (lt == FULL)


def test_global_kv_heads_override_needs_k_eq_v():
    with_flag = Gemma4Config.from_dict(
        {**BASE, "attention_k_eq_v": True, "num_global_key_value_heads": 1})
    without = Gemma4Config.from_dict({**BASE, "num_global_key_value_heads": 1})
    full = with_flag.layer_types.index(FULL)
    assert with_flag.layer_kv_heads(full) == 1
    assert without.layer_kv_heads(full) == 2


def test_double_wide_mlp_only_on_kv_shared_layers():
    cfg = Gemma4Config.from_dict(
        {**BASE, "num_kv_shared_layers": 4, "use_double_wide_mlp": True})
    for i in range(12):
        assert cfg.layer_intermediate(i) == (128 if i >= 8 else 64)


def test_double_wide_mlp_is_off_without_kv_sharing():
    cfg = Gemma4Config.from_dict({**BASE, "use_double_wide_mlp": True})
    assert all(cfg.layer_intermediate(i) == 64 for i in range(12))


def test_config_is_hashable_so_it_can_be_a_static_jit_field():
    hash(Gemma4Config.from_dict({**BASE}))


# ── forward pass against transformers' reference implementation ───────────────

REFERENCE_CASES = [
    ("plain dense", {}),
    ("kv sharing", {"num_kv_shared_layers": 4}),
    ("k_eq_v", {"attention_k_eq_v": True}),
    ("k_eq_v + global kv heads", {"attention_k_eq_v": True, "num_global_key_value_heads": 1}),
    ("double-wide mlp", {"num_kv_shared_layers": 4, "use_double_wide_mlp": True}),
    ("all features", {"num_kv_shared_layers": 4, "use_double_wide_mlp": True,
                      "attention_k_eq_v": True, "num_global_key_value_heads": 1}),
    ("mqa", {"num_key_value_heads": 1}),
]


def _build_pair(overrides):
    """Random weights loaded into both the reference model and ours."""
    torch = pytest.importorskip("torch")
    mod = pytest.importorskip("transformers.models.gemma4_unified.modeling_gemma4_unified")
    conf = pytest.importorskip("transformers.models.gemma4_unified.configuration_gemma4_unified")

    torch.manual_seed(0)
    kw = {**BASE, "use_bidirectional_attention": None, "attn_implementation": "eager",
          **overrides}
    cfg_t = conf.Gemma4UnifiedTextConfig(**kw)
    ref = mod.Gemma4UnifiedTextModel(cfg_t).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.4)
        for name, buf in ref.named_buffers():
            if name.endswith("layer_scalar"):
                buf.normal_(1.0, 0.1)

    raw = {k: v.detach().numpy().astype(np.float32) for k, v in ref.state_dict().items()}
    cfg_j = Gemma4Config.from_dict({**kw, "layer_types": list(cfg_t.layer_types)})
    return ref, from_weights(cfg_j, raw), cfg_j, torch


@pytest.mark.parametrize("name,overrides", REFERENCE_CASES, ids=[c[0] for c in REFERENCE_CASES])
def test_forward_matches_transformers_reference(name, overrides):
    ref, jm, cfg, torch = _build_pair(overrides)

    ids = np.array([3, 9, 1, 40, 7, 22, 15, 2, 61, 30])   # longer than the window
    with torch.no_grad():
        expected = ref(input_ids=torch.tensor(ids)[None]).last_hidden_state[0].numpy()

    S = len(ids)
    causal = jnp.tril(jnp.ones((S, S), jnp.bool_))
    window = jnp.triu(jnp.ones((S, S), jnp.bool_), -(cfg.sliding_window - 1))
    got = np.array(jm.hidden_states(jnp.array(ids),
                                    {"full": causal, "sliding": causal & window},
                                    jnp.arange(S)))

    np.testing.assert_allclose(got, expected, rtol=0, atol=2e-4 * np.abs(expected).max())


@pytest.mark.parametrize("chunk,slot", [(8, 0), (8, 5), (16, 15), (33, 30)])
def test_probe_row_equals_a_true_causal_forward(chunk, slot):
    """The invariant the whole probe rests on, for the 12B path."""
    _, jm, cfg, _ = _build_pair({"num_kv_shared_layers": 4, "attention_k_eq_v": True})

    ctx = jnp.array([3, 9, 1, 40])
    n = ctx.shape[0]
    probes = jnp.arange(chunk) % cfg.vocab_size
    token = int(probes[slot])

    masks = {"sliding": probe_mask(n, chunk, cfg.sliding_window),
             "full": probe_mask(n, chunk)}
    got = jm.hidden_states(jnp.concatenate([ctx, probes]), masks,
                           probe_positions(n, chunk))[n + slot]

    S = n + 1
    causal = jnp.tril(jnp.ones((S, S), jnp.bool_))
    window = jnp.triu(jnp.ones((S, S), jnp.bool_), -(cfg.sliding_window - 1))
    expected = jm.hidden_states(jnp.concatenate([ctx, jnp.array([token])]),
                                {"full": causal, "sliding": causal & window},
                                jnp.arange(S))[-1]

    np.testing.assert_allclose(np.array(got), np.array(expected), atol=1e-4)


def test_probe_honours_the_sliding_window():
    """Dropping the window changes the answer once the context outruns it."""
    _, jm, cfg, _ = _build_pair({})

    ctx = jnp.arange(1, 13) % 64          # n = 12, window = 4
    n, C = ctx.shape[0], 4
    probes = jnp.arange(C)

    def probe(window):
        masks = {"sliding": probe_mask(n, C, window), "full": probe_mask(n, C)}
        return jm.hidden_states(jnp.concatenate([ctx, probes]), masks,
                                probe_positions(n, C))[n]

    S = n + 1
    causal = jnp.tril(jnp.ones((S, S), jnp.bool_))
    win = jnp.triu(jnp.ones((S, S), jnp.bool_), -(cfg.sliding_window - 1))
    expected = jm.hidden_states(jnp.concatenate([ctx, jnp.array([0])]),
                                {"full": causal, "sliding": causal & win},
                                jnp.arange(S))[-1]

    np.testing.assert_allclose(np.array(probe(cfg.sliding_window)), np.array(expected), atol=1e-4)
    assert not np.allclose(np.array(probe(None)), np.array(expected), atol=1e-3)
