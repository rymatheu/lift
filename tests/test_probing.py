"""Tests for the vocabulary-probe mask and RoPE positions.

The invariant these protect: every probe token stands in for "the next token
after the context", so they all share RoPE position n and none of them can
see another probe. Getting the positions wrong silently corrupts every row
of the output except the first of each chunk.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lift.models.probing import probe_mask, probe_positions


def test_positions_pin_every_probe_at_n():
    pos = np.array(probe_positions(4, 3))
    np.testing.assert_array_equal(pos[:4], [0, 1, 2, 3])
    np.testing.assert_array_equal(pos[4:], [4, 4, 4])


@pytest.mark.parametrize("n,C", [(1, 1), (4, 3), (9, 64), (37, 5)])
def test_positions_length_and_range(n, C):
    pos = np.array(probe_positions(n, C))
    assert pos.shape == (n + C,)
    assert pos.max() == n


def test_context_is_causal_and_blind_to_probes():
    n, C = 5, 4
    m = np.array(probe_mask(n, C))
    ctx = m[:n, :n]
    np.testing.assert_array_equal(ctx, np.tril(np.ones((n, n), bool)))
    # context must never attend to a probe, or its representation would
    # depend on which chunk it was batched with
    assert not m[:n, n:].any()


def test_each_probe_sees_the_context_and_only_itself():
    n, C = 5, 4
    m = np.array(probe_mask(n, C))
    probes = m[n:]
    assert probes[:, :n].all()                       # whole context visible
    np.testing.assert_array_equal(probes[:, n:], np.eye(C, dtype=bool))


@pytest.mark.parametrize("C", [1, 2, 16, 100])
def test_every_probe_attends_to_the_same_number_of_positions(C):
    """Row sums must not depend on chunk size, or results would drift with it."""
    n = 6
    m = np.array(probe_mask(n, C))
    assert set(m[n:].sum(axis=1)) == {n + 1}


def test_sliding_window_limits_context_rows():
    n, C, w = 6, 2, 3
    m = np.array(probe_mask(n, C, sliding_window=w))
    for i in range(n):
        visible = np.flatnonzero(m[i, :n])
        assert visible.max() == i                    # causal
        assert visible.min() == max(0, i - (w - 1))  # windowed


def test_sliding_window_limits_probe_rows():
    """A probe sits at position n, so it sees the last w-1 context tokens."""
    n, C, w = 6, 2, 3
    m = np.array(probe_mask(n, C, sliding_window=w))
    for row in m[n:]:
        visible = np.flatnonzero(row[:n])
        np.testing.assert_array_equal(visible, [n - 2, n - 1])


def test_window_wider_than_context_is_a_no_op():
    n, C = 4, 3
    np.testing.assert_array_equal(
        np.array(probe_mask(n, C, sliding_window=1000)),
        np.array(probe_mask(n, C)),
    )


def test_mask_is_square_and_boolean():
    m = probe_mask(3, 5)
    assert m.shape == (8, 8)
    assert m.dtype == jnp.bool_
