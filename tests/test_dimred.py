"""Tests for UMAP backend selection."""

import builtins

import pytest

from lift.geometry import dimred


def test_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        dimred.backend_in_use("nope")


def test_auto_falls_back_to_cpu_when_cuml_is_absent(monkeypatch):
    real_import = builtins.__import__

    def no_cuml(name, *args, **kwargs):
        if name.startswith("cuml"):
            raise ImportError("no cuml here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cuml)
    pytest.importorskip("umap")
    assert dimred.backend_in_use("auto") == "umap-learn"


def test_explicit_cuml_does_not_silently_fall_back(monkeypatch):
    real_import = builtins.__import__

    def no_cuml(name, *args, **kwargs):
        if name.startswith("cuml"):
            raise ImportError("no cuml here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cuml)
    with pytest.raises(ImportError, match="pypi.nvidia.com"):
        dimred.backend_in_use("cuml")


def test_error_names_both_installs_when_nothing_is_available(monkeypatch):
    real_import = builtins.__import__

    def nothing(name, *args, **kwargs):
        if name.startswith(("cuml", "umap")):
            raise ImportError("absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", nothing)
    with pytest.raises(ImportError) as e:
        dimred.backend_in_use("auto")
    assert "lift[gpu]" in str(e.value) and "lift[cpu]" in str(e.value)


@pytest.mark.parametrize("n_components", [2, 3])
def test_reduce_shape_and_dtype(n_components):
    pytest.importorskip("umap")
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 12))
    Y = dimred.reduce(X, n_neighbors=5, n_components=n_components,
                      backend="umap-learn", random_state=0)
    assert Y.shape == (60, n_components)
    assert np.isfinite(Y).all()
