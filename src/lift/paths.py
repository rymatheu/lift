"""Filesystem layout.

Every path the project reads or writes resolves through here, so nothing in
the codebase hardcodes an absolute path and experiments can be run from any
working directory.

Layout, relative to the repo root:

    data/       embedding matrices (.npy) and reduced point clouds
    outputs/    figures, gifs, frame sequences, Potree octree exports
    vendor/     third-party assets (the Potree viewer)

Override any of them with the LIFT_DATA_DIR, LIFT_OUTPUT_DIR or
LIFT_VENDOR_DIR environment variables -- useful when the embeddings live on
a scratch disk rather than next to the source.
"""

import os
from pathlib import Path

# src/lift/paths.py -> src/lift -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def _dir(env_var: str, default: str) -> Path:
    return Path(os.environ.get(env_var, REPO_ROOT / default))


DATA_DIR   = _dir("LIFT_DATA_DIR",   "data")
OUTPUT_DIR = _dir("LIFT_OUTPUT_DIR", "outputs")
VENDOR_DIR = _dir("LIFT_VENDOR_DIR", "vendor")

# Embeddings, by kind.
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
# Initial (pre-transformer) embedding table, per model.
INITIAL_EMBEDDINGS = EMBEDDINGS_DIR / "{model}_embed_initial.npy"
# Contextual embeddings for one running context, one file per prefix length.
CONTEXT_DIR = EMBEDDINGS_DIR / "contexts"
# UMAP/PCA-reduced point clouds.
REDUCED_DIR = DATA_DIR / "reduced"

POTREE_DIR       = VENDOR_DIR / "potree"
POTREE_CONVERTER = POTREE_DIR / "PotreeConverter"
POTREE_BUILD     = POTREE_DIR / "viewer_build"
POTREE_LIBS      = POTREE_DIR / "viewer_libs"


def data(*parts) -> Path:
    """Path inside data/, creating the parent directory."""
    p = DATA_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def output(*parts) -> Path:
    """Path inside outputs/, creating the parent directory."""
    p = OUTPUT_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def output_dir(*parts) -> Path:
    """Directory inside outputs/, created if missing."""
    p = OUTPUT_DIR.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
