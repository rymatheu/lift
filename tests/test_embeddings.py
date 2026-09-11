"""Tests for locating contextual embedding files on disk.

The names in lift.study record how the study's files were originally
written. Small differences creep in between what a generator produces and
what is recorded -- a missing hyphen, a different rendering of the BOS
token, a reworded context -- none of which change which step of the study a
file belongs to. The leading index is what identifies it.
"""

import importlib

import pytest


@pytest.fixture
def contexts(tmp_path, monkeypatch):
    """A data directory, with lift.paths and lift.embeddings pointed at it."""
    monkeypatch.setenv("LIFT_DATA_DIR", str(tmp_path))
    import lift.embeddings
    import lift.paths
    importlib.reload(lift.paths)
    importlib.reload(lift.embeddings)
    from lift.study import CONTEXT_SUBDIR

    directory = tmp_path / "embeddings" / "contexts" / CONTEXT_SUBDIR
    directory.mkdir(parents=True)
    yield directory, lift.embeddings

    # leave the modules pointing back at the real defaults
    monkeypatch.delenv("LIFT_DATA_DIR", raising=False)
    importlib.reload(lift.paths)
    importlib.reload(lift.embeddings)


def test_exact_name_wins(contexts):
    directory, embeddings = contexts
    (directory / "1-bos-embed.npy").touch()
    assert embeddings.context_path("1-bos-embed.npy").name == "1-bos-embed.npy"


def test_falls_back_to_the_leading_index(contexts):
    """A file written with different text still resolves by its index."""
    directory, embeddings = contexts
    (directory / "10-bosSOMETHING_ELSE-embed.npy").touch()
    got = embeddings.context_path("10-bosThere_is_a_mole_in_the_middle_of_the-embed.npy")
    assert got.name == "10-bosSOMETHING_ELSE-embed.npy"


def test_resolves_the_name_with_no_hyphen(contexts):
    """17bos... (sic) is what is actually on disk for that step."""
    directory, embeddings = contexts
    odd = "17bosThere_is_a_mole_in_the_middle_of_that-embed.npy"
    (directory / odd).touch()
    assert embeddings.context_path(odd).name == odd


def test_index_match_is_exact_not_a_prefix(contexts):
    """Index 1 must not match 10, 11, ..."""
    directory, embeddings = contexts
    (directory / "10-bosTen-embed.npy").touch()
    (directory / "11-bosEleven-embed.npy").touch()
    assert not embeddings.context_path("1-bos-embed.npy").exists()


def test_ambiguous_index_is_an_error(contexts):
    directory, embeddings = contexts
    (directory / "3-bosOne-embed.npy").touch()
    (directory / "3-bosTwo-embed.npy").touch()
    with pytest.raises(FileNotFoundError, match="share index 3"):
        embeddings.context_path("3-bosThere_is-embed.npy")


def test_missing_index_returns_the_exact_path(contexts):
    """So the caller's error names the file they asked for."""
    _, embeddings = contexts
    got = embeddings.context_path("99-nothing-embed.npy")
    assert got.name == "99-nothing-embed.npy"
    assert not got.exists()


def test_missing_directory_is_not_an_error(contexts):
    directory, embeddings = contexts
    for child in directory.iterdir():
        child.unlink()
    directory.rmdir()
    assert not embeddings.context_path("1-bos-embed.npy").exists()


@pytest.mark.parametrize("name,expected", [
    ("1-bos-embed.npy", 1),
    ("17bosThere-embed.npy", 17),
    ("10-x-embed.npy", 10),
    ("bos-embed.npy", None),
    ("", None),
])
def test_leading_index(contexts, name, expected):
    _, embeddings = contexts
    assert embeddings._leading_index(name) == expected
