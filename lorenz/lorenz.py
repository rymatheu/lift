#

import jax
import jax.numpy as jnp
import equinox
import diffrax
import matplotlib.pyplot as plt


def f(t, x, args = (10., 28., 8./3.)):
    sigma = args[0]
    rho = args[1]
    beta = args[2]

    dx = sigma * (x[1] - x[0])
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]

    return jnp.array([dx, dy, dz])

def build_solver(args = (10.0, 28.0, 8.0/3.0)):
    """Generates a continuous trajectory of R^3 vectors using Diffrax."""
    # Define the ODE term
    term = diffrax.ODETerm(f)
    solver = diffrax.Dopri5()
    dt0 = None
    stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)
    
    
    def sim(ts, x0):
        sol = diffrax.diffeqsolve(
            terms=term,
            solver=solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=x0,
            args=args,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=stepsize_controller
        )

        trajectory = sol.ys

        return trajectory

    # Batched version: (N, T) ts and (N, 3) x0s -> (N, T, 3)
    sim_batch = jax.vmap(sim, in_axes=(0, 0))

    return sim_batch


def save_dataset(xs, ts_batch, path="lorenz_dataset.npz"):
    """Normalize globally and save trajectories to .npz.

    Normalization is per spatial dimension across all N*T points so the
    attractor geometry is preserved. Stats are saved for inversion.

    Args:
        xs:       (N, T, 3) raw trajectories
        ts_batch: (N, T)    time arrays
        path:     output file path
    """
    import numpy as np

    mean = xs.mean(axis=(0, 1))   # (3,)
    std  = xs.std(axis=(0, 1))    # (3,)
    xs_norm = (xs - mean) / std

    np.savez(
        path,
        trajectories=np.array(xs_norm),
        ts=np.array(ts_batch),
        mean=np.array(mean),
        std=np.array(std),
    )
    print(f"Saved {xs.shape[0]} trajectories to {path}")
    print(f"  mean: {mean},  std: {std}")


def main():
    import numpy as np

    key = jax.random.PRNGKey(np.random.randint(0, 1000))
    N = 64
    T = 1000

    sim_batch = jax.jit(build_solver())

    # Random initial conditions
    key, k1, k2 = jax.random.split(key, 3)
    x0s = jax.random.uniform(k1, (N, 3),
                             minval=jnp.array([-20., -27.,  2.]),
                             maxval=jnp.array([ 20.,  27., 47.]))

    # Different time windows, all with T points
    t_starts = jnp.zeros((N,)) # jax.random.uniform(k2, (N,), minval=0.0, maxval=5.0)
    t_ends = jnp.ones((N,)) * 10. # t_starts + jax.random.uniform(k2, (N,), minval=5.0, maxval=10.0)
    ts_batch = jax.vmap(lambda t0, t1: jnp.linspace(t0, t1, T))(t_starts, t_ends)

    # xs shape: (N, T, 3)
    xs = sim_batch(ts_batch, x0s)

    save_dataset(xs, ts_batch)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(N):
        ax.plot(xs[i, :, 0], xs[i, :, 1], xs[i, :, 2], lw=0.3, alpha=0.4)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Lorenz Attractor — {N} trajectories")
    plt.tight_layout()
    plt.savefig("lorenz.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()