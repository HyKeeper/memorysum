from __future__ import annotations

from memorysum.core.chunking import build_windows
from memorysum.settings import load_pipeline_config


def test_windows_keep_exact_source_offsets_and_both_scales():
    source = "This is a first sentence.  Here is another sentence; with a condition."
    windows = build_windows([("user", source)], load_pipeline_config())

    assert {window.scale for window in windows} == {"leaf", "context"}
    assert all(source[window.start_char : window.end_char] == window.raw_text for window in windows)
    assert all(window.embedding_text.endswith(window.raw_text) for window in windows)
    assert all(window.proxy_token_count > 0 for window in windows)


def test_long_text_chunks_overlap_without_losing_source_text():
    source = "Memory facts should remain verbatim. " * 800
    windows = build_windows([("assistant", source)], load_pipeline_config())
    for scale in ("leaf", "context"):
        current = [window for window in windows if window.scale == scale]
        assert len(current) > 1
        assert current[0].start_char == 0
        assert current[-1].end_char == len(source.rstrip())
        assert all(
            source[window.start_char : window.end_char] == window.raw_text for window in current
        )
        assert all(
            left.start_char < right.start_char
            for left, right in zip(current, current[1:], strict=False)
        )
