from __future__ import annotations

from memorysum.core.chunking import build_ingest_plan, build_windows
from memorysum.settings import load_pipeline_config


def test_context_windows_group_only_ordered_messages_from_the_same_add():
    messages = [
        ("user", "I travelled to Kyoto in May 2024."),
        ("assistant", "You mentioned a quiet place near the station."),
        ("user", "The cafe was named Blue Lantern."),
    ]
    pipeline = load_pipeline_config()
    plan = build_ingest_plan(messages, pipeline)
    context_windows = [window for window in plan.windows if window.scale == "context"]

    assert any(len(window.source_spans) > 1 for window in context_windows)
    assert all(len(window.source_spans) <= pipeline.groups.max_spans for window in context_windows)
    for window in context_windows:
        for span in window.source_spans:
            source = messages[span.message_ordinal][1]
            assert source[span.start_char : span.end_char]
        if len(window.source_spans) > 1:
            assert "Source role: user" in window.embedding_text
            assert "Source role: assistant" in window.embedding_text


def test_window_atoms_are_mapped_to_exact_parent_spans():
    source = "I prefer tea. This is important."
    plan = build_ingest_plan([("user", source)], load_pipeline_config())
    atoms = [atom for atom in plan.atoms if atom.message_ordinal == 0]
    leaf = next(window for window in plan.windows if window.scale == "leaf")

    assert atoms
    assert leaf.atom_refs
    for atom_ordinal in leaf.source_spans[0].atom_ordinals:
        atom = next(atom for atom in atoms if atom.ordinal == atom_ordinal)
        assert source[atom.start_char : atom.end_char]


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
            source[span.start_char : span.end_char]
            for window in current
            for span in window.source_spans
        )
        assert all(
            left.start_char < right.start_char
            for left, right in zip(current, current[1:], strict=False)
        )
