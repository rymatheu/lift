"""UMAP dimensionality reduction.

cuML is imported lazily so that importing this module -- and running
--help on the experiments that use it -- does not require a GPU.

Was context/umap.py, which shadowed the umap-learn package name for anything
in the process doing `import umap`.
"""

import numpy as np


def reduce(X, n_neighbors=15, min_dist=0.1, n_components=3, metric="cosine"):
    """Project X to n_components dimensions with cuML's UMAP."""
    try:
        from cuml.manifold import UMAP
    except ImportError as e:  # pragma: no cover - depends on the GPU stack
        raise ImportError(
            "UMAP reduction needs cuML, which is GPU-only:\n"
            "    pip install --extra-index-url=https://pypi.nvidia.com 'lift[gpu]'"
        ) from e

    X = np.asarray(X).astype(np.float32)

    umap_3d = UMAP(
        n_components=n_components,
        metric=metric,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        output_type="numpy",
    )

    return umap_3d.fit_transform(X)
