import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt


def sinusoidal_time_encoding(ts, d_model):
    """Encode scalar time values into d_model-dim sinusoidal vectors.

    Uses actual time values (not position indices) so variable time windows
    are handled correctly.

    Args:
        ts:      (T,) actual time values
        d_model: embedding dimension (must be even)
    Returns:
        (T, d_model)
    """
    d_half = (d_model + 1) // 2  # ceil so concat covers >= d_model cols
    log_timescale = jnp.log(10000.0) / max(d_half - 1, 1)
    freqs = jnp.exp(-jnp.arange(d_half) * log_timescale)   # (d_half,)
    args  = ts[:, None] * freqs[None, :]                    # (T, d_half)
    enc   = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)  # (T, 2*d_half)
    return enc[:, :d_model]                                 # (T, d_model)


def causal_mask(T):
    """Lower-triangular boolean mask — True where attention is allowed."""
    return jnp.tril(jnp.ones((T, T), dtype=bool))


class MultiHeadAttention(eqx.Module):
    q_proj:   eqx.nn.Linear
    k_proj:   eqx.nn.Linear
    v_proj:   eqx.nn.Linear
    out_proj: eqx.nn.Linear
    num_heads: int = eqx.field(static=True)
    head_dim:  int = eqx.field(static=True)

    def __init__(self, d_model, num_heads, *, key):
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.q_proj   = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k1)
        self.k_proj   = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k2)
        self.v_proj   = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k3)
        self.out_proj = eqx.nn.Linear(d_model, d_model, key=k4)

    def __call__(self, x, mask=None):
        # x: (T, d_model)
        # returns: (T, d_model), weights (num_heads, T, T)
        T, d = x.shape
        H, Hd = self.num_heads, self.head_dim

        q = jax.vmap(self.q_proj)(x).reshape(T, H, Hd).transpose(1, 0, 2)  # (H, T, Hd)
        k = jax.vmap(self.k_proj)(x).reshape(T, H, Hd).transpose(1, 0, 2)
        v = jax.vmap(self.v_proj)(x).reshape(T, H, Hd).transpose(1, 0, 2)

        attn = jnp.einsum("hid,hjd->hij", q, k) * (Hd ** -0.5)  # (H, T, T)

        if mask is not None:
            attn = jnp.where(mask[None], attn, jnp.finfo(attn.dtype).min)

        weights = jax.nn.softmax(attn, axis=-1)                              # (H, T, T)
        out = jnp.einsum("hij,hjd->hid", weights, v).transpose(1, 0, 2).reshape(T, d)
        return jax.vmap(self.out_proj)(out), weights


class TransformerBlock(eqx.Module):
    attn:  MultiHeadAttention
    mlp:   eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(self, d_model, num_heads, mlp_dim, *, key):
        k1, k2 = jax.random.split(key)
        self.attn  = MultiHeadAttention(d_model, num_heads, key=k1)
        self.mlp   = eqx.nn.MLP(d_model, d_model, mlp_dim, depth=1,
                                 activation=jax.nn.gelu, key=k2)
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.norm2 = eqx.nn.LayerNorm(d_model)

    def __call__(self, x, mask=None):
        # Pre-norm residual
        attn_out, weights = self.attn(jax.vmap(self.norm1)(x), mask=mask)
        x = x + attn_out
        x = x + jax.vmap(self.mlp)(jax.vmap(self.norm2)(x))
        return x, weights


class Transformer(eqx.Module):
    blocks: list
    d_model: int = eqx.field(static=True)

    def __init__(self, d_model=3, num_heads=3, num_layers=2, mlp_dim=12, *, key):
        keys = jax.random.split(key, num_layers)
        self.d_model = d_model
        self.blocks = [
            TransformerBlock(d_model, num_heads, mlp_dim, key=keys[i])
            for i in range(num_layers)
        ]

    def __call__(self, x, ts, mask=None):
        # x:  (T, 3) — tokens live in attractor space throughout
        # ts: (T,)
        # returns: predictions (T, 3), attn_weights list[(num_heads, T, T)]
        h = x + sinusoidal_time_encoding(ts, self.d_model)
        attn_weights = []
        for block in self.blocks:
            h, w = block(h, mask=mask)
            attn_weights.append(w)
        return h, attn_weights


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def loss_fn(model, x, ts):
    # x: (T, 3), ts: (T,)  — single trajectory, causal next-step MSE
    T    = x.shape[0]
    mask = causal_mask(T - 1)
    pred, _ = model(x[:-1], ts[:-1], mask=mask)  # (T-1, 3)
    return jnp.mean((pred - x[1:]) ** 2)


def save_checkpoint(model, path):
    eqx.tree_serialise_leaves(path, model)


def load_checkpoint(model, path):
    return eqx.tree_deserialise_leaves(path, model)


def _latest_checkpoint(checkpoint_dir):
    """Return (path, epoch) of the latest epoch-numbered checkpoint, or (None, 0)."""
    import glob, os, re
    ckpts = glob.glob(os.path.join(checkpoint_dir, "model_epoch*.eqx"))
    if not ckpts:
        return None, 0
    def epoch_num(p):
        m = re.search(r"model_epoch(\d+)\.eqx", p)
        return int(m.group(1)) if m else 0
    latest = max(ckpts, key=epoch_num)
    return latest, epoch_num(latest)


def train(model, data_path, *, num_epochs=200, batch_size=16, lr=1e-3,
          checkpoint_dir="checkpoints", checkpoint_every=50, resume=False, key):
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    if resume:
        ckpt_path, start_epoch = _latest_checkpoint(checkpoint_dir)
        if ckpt_path:
            model = load_checkpoint(model, ckpt_path)
            print(f"Resumed from {ckpt_path} (epoch {start_epoch})")
        else:
            print("No checkpoint found, starting from scratch")

    data = np.load(data_path)
    xs = jnp.array(data["trajectories"])   # (N, T, 3)
    ts = jnp.array(data["ts"])             # (N, T)
    N  = xs.shape[0]

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, x_batch, ts_batch, opt_state):
        def batch_loss(model):
            return jnp.mean(
                jax.vmap(lambda x, t: loss_fn(model, x, t))(x_batch, ts_batch)
            )
        loss, grads = eqx.filter_value_and_grad(batch_loss)(model)
        updates, new_opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        return eqx.apply_updates(model, updates), new_opt_state, loss

    for epoch in range(start_epoch, start_epoch + num_epochs):
        key, k = jax.random.split(key)
        perm = jax.random.permutation(k, N)
        total_loss, n_batches = 0.0, 0
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            model, opt_state, loss = step(model, xs[idx], ts[idx], opt_state)
            total_loss += float(loss)
            n_batches  += 1
        epoch_loss = total_loss / n_batches
        print(f"epoch {epoch + 1:4d}  loss={epoch_loss:.6f}")

        if (epoch + 1) % checkpoint_every == 0:
            path = os.path.join(checkpoint_dir, f"model_epoch{epoch + 1:04d}.eqx")
            save_checkpoint(model, path)
            print(f"  saved {path}")

    save_checkpoint(model, os.path.join(checkpoint_dir, "model_final.eqx"))
    return model


def evaluate(data_path="lorenz_dataset.npz", checkpoint_dir="checkpoints",
             traj_idx=0):
    """Load the latest checkpoint and plot transformer predictions vs ground truth."""
    import glob
    import os

    # Latest checkpoint by modification time
    ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "*.eqx")), key=os.path.getmtime)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest = ckpts[-1]
    print(f"Loading {latest}")

    model = load_checkpoint(Transformer(key=jax.random.PRNGKey(0)), latest)

    data = np.load(data_path)
    xs = jnp.array(data["trajectories"])  # (N, T, 3)
    ts = jnp.array(data["ts"])            # (N, T)

    x  = xs[traj_idx]   # (T, 3)
    t  = ts[traj_idx]   # (T,)
    T  = x.shape[0]

    pred, _ = model(x[:-1], t[:-1], mask=causal_mask(T - 1))  # (T-1, 3)
    pred = np.array(pred)
    true = np.array(x[1:])  # ground truth targets

    labels = ["x", "y", "z"]
    t_axis = np.array(t[1:])

    fig = plt.figure(figsize=(14, 8))
    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.plot(true[:, 0], true[:, 1], true[:, 2], lw=0.5, alpha=0.8, label="true")
    ax3d.plot(pred[:, 0], pred[:, 1], pred[:, 2], lw=0.5, alpha=0.8,
              linestyle="--", label="pred")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("Phase portrait")
    ax3d.legend(fontsize=7)

    for i, lbl in enumerate(labels):
        ax = fig.add_subplot(3, 2, 2 * (i + 1))
        ax.plot(t_axis, true[:, i], lw=0.7, label="true")
        ax.plot(t_axis, pred[:, i], lw=0.7, linestyle="--", label="pred")
        ax.set_ylabel(lbl)
        if i == 0:
            ax.legend(fontsize=7)
        if i == 2:
            ax.set_xlabel("t")
        else:
            ax.set_xticklabels([])

    plt.tight_layout()
    plt.savefig("evaluation.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    import sys
    key = jax.random.PRNGKey(np.random.randint(999))
    if "--eval" in sys.argv:
        evaluate(traj_idx=np.random.randint(999))
    else:
        resume = "--resume" in sys.argv
        model  = Transformer(key=key)
        model  = train(model, "lorenz_dataset.npz", num_epochs=500, checkpoint_every=100, resume=resume, key=key)
