"""3D scatter plots of embedding spaces.

plot_3d and plot_3d_umap differ only in how the axes are labelled -- PCA
axes carry their explained-variance ratio, UMAP axes are just components --
so they share one scatter implementation.
"""

import numpy as np


def _scatter_3d(
    ax,
    coords: np.ndarray,
    title: str,
    axis_labels: tuple[str, str, str],
    highlight: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    n_labels: int = 5,
    point_colors: np.ndarray | None = None,
    point_cmap: str = "turbo",
):
    colors = point_colors if point_colors is not None else np.arange(len(coords))
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=colors, cmap=point_cmap, s=4, alpha=0.6, linewidths=0)
    if highlight is not None:
        h = coords[highlight]
        ax.scatter(h[:, 0], h[:, 1], h[:, 2],
                   c="red", s=120, marker="^", edgecolors="white", linewidths=0.8, zorder=5)
        if labels is not None:
            for i in range(n_labels):
                x, y, z = h[i]
                ax.text(x, y, z, f" {str(labels[i])}", fontsize=9, color="black",
                        fontweight="bold", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.2", fc="yellow", ec="black", alpha=0.9))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(axis_labels[0], fontsize=8)
    ax.set_ylabel(axis_labels[1], fontsize=8)
    ax.set_zlabel(axis_labels[2], fontsize=8)
    ax.tick_params(labelsize=6)


def plot_3d(
    ax,
    coords: np.ndarray,
    title: str,
    var: np.ndarray,
    highlight: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    n_labels: int = 5,
    point_colors: np.ndarray | None = None,
    point_cmap: str = "turbo",
):
    """Scatter PCA coordinates, labelling each axis with its variance ratio."""
    axis_labels = (
        f"PC1 ({var[0]:.1%})",
        f"PC2 ({var[1]:.1%})",
        f"PC3 ({var[2]:.1%})",
    )
    _scatter_3d(ax, coords, title, axis_labels, highlight, labels, n_labels,
                point_colors, point_cmap)


def plot_3d_umap(
    ax,
    coords: np.ndarray,
    title: str,
    highlight: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    n_labels: int = 5,
    point_colors: np.ndarray | None = None,
    point_cmap: str = "turbo",
):
    """Scatter UMAP coordinates, whose axes carry no variance interpretation."""
    axis_labels = ("Component 1", "Component 2", "Component 3")
    _scatter_3d(ax, coords, title, axis_labels, highlight, labels, n_labels,
                point_colors, point_cmap)
