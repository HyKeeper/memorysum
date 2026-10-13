from __future__ import annotations

from pathlib import Path

import pytest

from memorysum.settings import RuntimeSettings


def test_runtime_loads_bearer_tokens_from_local_file(tmp_path: Path, monkeypatch):
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text(
        "# active callers\nfirst-secret-token\n\nsecond-secret-token\nfirst-secret-token\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORYSUM_API_KEYS_FILE", "keys.txt")
    monkeypatch.setenv("MEMORYSUM_DEBUG", "false")

    settings = RuntimeSettings.from_environment()

    assert settings.api_auth_scheme == "bearer"
    assert tuple(token.get_secret_value() for token in settings.api_auth_tokens) == (
        "first-secret-token",
        "second-secret-token",
    )


def test_runtime_requires_nonempty_bearer_key_file(tmp_path: Path, monkeypatch):
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("# no active keys\n\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORYSUM_API_KEYS_FILE", str(keys_file))
    with pytest.raises(ValueError, match="contains no tokens"):
        RuntimeSettings.from_environment()


def test_runtime_rejects_tokens_with_whitespace(tmp_path: Path, monkeypatch):
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("bad token\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORYSUM_API_KEYS_FILE", str(keys_file))
    with pytest.raises(ValueError, match="printable ASCII"):
        RuntimeSettings.from_environment()


def test_pipeline_channel_weights_reference_known_channels():
    from memorysum.settings import load_pipeline_config

    pipeline = load_pipeline_config()
    known = {"leaf_dense", "context_dense", "english_fts", "chinese_fts", "options_lexical"}
    weights = pipeline.retrieval.channel_weights
    assert set(weights) <= known, f"unknown channel in channel_weights: {set(weights) - known}"
    assert all(value > 0 for value in weights.values()), "channel weights must be positive"


def test_pipeline_falls_back_to_base_weight_for_unlisted_channels():
    from memorysum.settings import load_pipeline_config

    pipeline = load_pipeline_config()
    unlisted = {
        "leaf_dense",
        "context_dense",
        "english_fts",
        "chinese_fts",
        "options_lexical",
    } - set(pipeline.retrieval.channel_weights)
    for name in unlisted:
        assert (
            pipeline.retrieval.channel_weights.get(name, pipeline.retrieval.base_channel_weight) > 0
        )
