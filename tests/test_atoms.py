from __future__ import annotations

from memorysum.core.atoms import atomize_messages


def test_atom_spans_preserve_markdown_structure_and_exact_offsets():
    source = (
        "# Preferences\n"
        "- Tea is my favorite.\n"
        "| Rule | Condition |\n"
        "| --- | --- |\n"
        "| R1 | unless I say otherwise |\n"
        "> quoted user text\n"
        "```text\n"
        "do not split this code line.\n"
        "```\n"
    )
    atoms = atomize_messages([("user", source)])
    kinds = {atom.kind for atom in atoms}

    assert {
        "heading",
        "list_item",
        "table_header",
        "table_separator",
        "table_row",
        "quote",
        "code",
    } <= kinds
    assert [atom.ordinal for atom in atoms] == list(range(len(atoms)))
    assert all(source[atom.start_char : atom.end_char] for atom in atoms)
    assert all(atom.raw_hash and len(atom.raw_hash) == 32 for atom in atoms)
