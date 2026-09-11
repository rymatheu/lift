"""UMAP dimensionality reduction.

Two backends, picked at call time:

  cuml    GPU, what the study was run on. Fast enough to reduce a
          (262144, hidden) matrix in a reasonable time.
  umap    umap-learn on the CPU. The fallback for machines without CUDA,
          notably Apple Silicon. Same algorithm, considerably slower on a
          full vocabulary.

Neither is imported at module load, so importing this module -- and running
--help on the experiments that use it -- needs no GPU and no UMAP install.

This module was context/umap.py, whose name shadowed umap-learn for anything
in the process doing `import umap`. That is also why the CPU backend can be
imported here at all.
"""

import numpy as np

BACKENDS = ("auto", "cuml", "umap-learn")


def _resolve(backend: str):
    """Return (name, UMAP class) for the requested backend."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")

    if backend in ("auto", "cuml"):
        try:
            from cuml.manifold import UMAP
            return "cuml", UMAP
        except ImportError:
            if backend == "cuml":
                raise ImportError(
                    "backend='cuml' needs the GPU stack:\n"
                    "    pip install --extra-index-url=https://pypi.nvidia.com 'lift[gpu]'"
                ) from None

    try:
        from umap import UMAP
        return "umap-learn", UMAP
    except ImportError:
        raise ImportError(
            "no UMAP backend available. Install one:\n"
            "    pip install --extra-index-url=https://pypi.nvidia.com 'lift[gpu]'  (CUDA)\n"
            "    pip install 'lift[cpu]'                                            (umap-learn)"
        ) from None


def reduce(X, n_neighbors=15, min_dist=0.1, n_components=3, metric="cosine",
           backend="auto", random_state=None, verbose=False):
    """Project X to n_components dimensions with UMAP.

    backend: "auto" prefers cuml and falls back to umap-learn.
    random_state makes umap-learn deterministic, at the cost of parallelism;
    cuML accepts it too.
    """
    name, UMAP = _resolve(backend)
    X = np.asarray(X).astype(np.float32)

    kwargs = dict(
        n_components=n_components,
        metric=metric,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    if name == "cuml":
        kwargs["output_type"] = "numpy"
    if random_state is not None:
        kwargs["random_state"] = random_state
    if verbose:
        kwargs["verbose"] = True
        print(f"reducing {X.shape} with {name}...")

    return np.asarray(UMAP(**kwargs).fit_transform(X))


def backend_in_use(backend: str = "auto") -> str:
    """Which backend reduce() would pick, without running it."""
    return _resolve(backend)[0]
