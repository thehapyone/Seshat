"""Structural chunking: provenance prefixes and row-safe table splitting."""

import pytest
from llama_index.core.schema import (
    MetadataMode,
    NodeRelationship,
    ObjectType,
    RelatedNodeInfo,
    TextNode,
)

from app.embeddings.chunking import (
    StructuralChunker,
    build_canonical_blocks,
    context_header,
    split_table,
)
from app.parsing import DocumentSegment, DocumentTooLargeError
from app.search_text import search_body

DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"

TABLE = "\n".join(
    ["| Diagnostic | Cause | Action |", "| --- | --- | --- |"]
    + [f"| D-{code} | Cause {code} | Replace S-{code} |" for code in range(100, 160)]
)


def node(text: str, **metadata: object) -> TextNode:
    """An ingest-shaped input node: located, with metadata kept out of the text."""
    full = {
        "document_id": DOCUMENT_ID,
        "title": "Equipment Service Manual",
        "section": "4.2 Diagnostics",
        "page": 112,
        **metadata,
    }
    built = TextNode(text=text, metadata=full)
    built.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
        node_id=DOCUMENT_ID, node_type=ObjectType.DOCUMENT
    )
    built.excluded_embed_metadata_keys = list(full)
    built.excluded_llm_metadata_keys = list(full)
    return built


def contents(chunks: list) -> list[str]:
    return [chunk.get_content(metadata_mode=MetadataMode.NONE) for chunk in chunks]


def test_context_header_names_the_document_and_page() -> None:
    assert (
        context_header(
            {
                "title": "Equipment Manual",
                "section": "4.2 Diagnostics",
                "page": 112,
            }
        )
        == "[Equipment Manual > 4.2 Diagnostics - page 112]"
    )


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({}, ""),
        ({"title": "Equipment Manual"}, "[Equipment Manual]"),
        ({"section": "7.4"}, "[7.4]"),
        ({"page": 9}, "[page 9]"),
        ({"title": "", "section": "", "page": None}, ""),
        # A boolean is an int in Python; it is not a page.
        ({"page": True}, ""),
    ],
)
def test_context_header_omits_what_is_not_known(metadata: dict, expected: str) -> None:
    assert context_header(metadata) == expected


def test_context_header_does_not_repeat_a_heading_the_converter_already_added() -> None:
    """Docling contextualizes its own chunks with their headings."""
    header = context_header(
        {"title": "Equipment Manual", "section": "4.2 Diagnostics", "page": 112},
        text="4.2 Diagnostics\nWhen the fault is active, check the speed sensor.",
    )
    assert header == "[Equipment Manual - page 112]"


def test_a_legacy_context_header_can_be_removed_before_metadata_refresh() -> None:
    metadata = {"title": "Old title", "section": "Old section", "page": 4}
    text = "[Old title > Old section - page 4]\nBattery management guidance."

    assert search_body(text, "", legacy_metadata=metadata) == (
        "Battery management guidance."
    )


def test_every_chunk_carries_its_provenance() -> None:
    chunker = StructuralChunker(chunk_size=64, chunk_overlap=8)
    chunks = chunker([node("Purge the circuit. " * 200)])

    assert len(chunks) > 1
    for text in contents(chunks):
        assert text.startswith(
            "[Equipment Service Manual > 4.2 Diagnostics - page 112]\n"
        )


def test_provenance_stays_out_of_the_metadata_text() -> None:
    """The prefix is content; metadata itself must not leak into embeddings."""
    chunker = StructuralChunker(chunk_size=256, chunk_overlap=16)
    chunk = chunker([node("Purge the circuit.")])[0]

    embedded = chunk.get_content(metadata_mode=MetadataMode.EMBED)
    assert embedded.startswith("[Equipment Service Manual")
    assert "document_id" not in embedded


def test_chunks_keep_the_document_as_their_source() -> None:
    """Deletion is scoped by ref_doc_id, so every chunk must keep the document id."""
    chunker = StructuralChunker(chunk_size=64, chunk_overlap=8)
    chunks = chunker([node(TABLE, is_table=True)]) + chunker([node("Prose. " * 100)])

    assert len(chunks) > 2
    assert {chunk.ref_doc_id for chunk in chunks} == {DOCUMENT_ID}


def test_table_chunks_keep_the_metadata_retrieval_depends_on() -> None:
    """A chunk without collection_id is dropped by the isolation guard at search."""
    chunker = StructuralChunker(chunk_size=200, chunk_overlap=16)
    chunks = chunker([node(TABLE, is_table=True, collection_id="manuals")])

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["collection_id"] == "manuals"
        assert chunk.metadata["document_id"] == DOCUMENT_ID
        assert chunk.metadata["title"] == "Equipment Service Manual"
        assert chunk.metadata["page"] == 112
        assert chunk.excluded_embed_metadata_keys
        assert "collection_id" not in chunk.get_content(metadata_mode=MetadataMode.EMBED)


def test_an_empty_segment_produces_no_chunks() -> None:
    chunker = StructuralChunker(chunk_size=256, chunk_overlap=16)
    assert chunker([node("   \n  ")]) == []


def test_a_table_the_converter_flagged_repeats_its_header_in_every_part() -> None:
    chunker = StructuralChunker(chunk_size=200, chunk_overlap=16)
    chunks = chunker([node(TABLE, is_table=True)])

    assert len(chunks) > 1
    for text in contents(chunks):
        body = text.split("\n", 1)[1]
        assert body.startswith("| Diagnostic | Cause | Action |\n| --- | --- | --- |")


def test_table_rows_are_never_split() -> None:
    chunker = StructuralChunker(chunk_size=200, chunk_overlap=16)
    chunks = chunker([node(TABLE, is_table=True)])

    rows = [
        line for text in contents(chunks) for line in text.split("\n") if line.startswith("| D-")
    ]
    assert rows == [line for line in TABLE.split("\n") if line.startswith("| D-")]


def test_a_table_that_fits_stays_whole() -> None:
    small = "| Diagnostic | Action |\n| --- | --- |\n| D-142 | Replace sensor S-3 |"
    chunker = StructuralChunker(chunk_size=800, chunk_overlap=120)
    chunks = chunker([node(small, is_table=True)])

    assert len(chunks) == 1
    assert "table_part" not in chunks[0].metadata


def test_split_table_parts_are_numbered() -> None:
    chunker = StructuralChunker(chunk_size=200, chunk_overlap=16)
    chunks = chunker([node(TABLE, is_table=True)])

    total = len(chunks)
    assert [chunk.metadata["table_part"] for chunk in chunks] == list(range(1, total + 1))
    assert {chunk.metadata["table_parts"] for chunk in chunks} == {total}


def test_table_metadata_never_reaches_the_embedding() -> None:
    chunker = StructuralChunker(chunk_size=200, chunk_overlap=16)
    chunk = chunker([node(TABLE, is_table=True)])[0]

    assert "table_part" not in chunk.get_content(metadata_mode=MetadataMode.EMBED)
    assert "table_part" not in chunk.get_content(metadata_mode=MetadataMode.LLM)


def test_the_converters_caption_is_carried_into_every_part() -> None:
    chunker = StructuralChunker(chunk_size=200, chunk_overlap=16)
    chunks = chunker(
        [node(TABLE, is_table=True, caption="Table 4-2 Diagnostic codes")]
    )

    assert len(chunks) > 1
    for text in contents(chunks):
        assert "Table 4-2 Diagnostic codes" in text


def test_a_caption_already_in_the_text_is_not_repeated() -> None:
    caption = "Table 4-2 Diagnostic codes"
    chunker = StructuralChunker(chunk_size=800, chunk_overlap=120)
    small = f"{caption}\n| Diagnostic | Action |\n| --- | --- |\n| D-142 | Replace sensor S-3 |"
    chunk = chunker([node(small, is_table=True, caption=caption)])[0]

    assert chunk.get_content(metadata_mode=MetadataMode.NONE).count(caption) == 1


def test_an_unflagged_table_goes_to_the_sentence_splitter() -> None:
    """Structure comes from the converter: tabular-looking text is not a table."""
    chunker = StructuralChunker(chunk_size=200, chunk_overlap=16)
    chunks = chunker([node(TABLE)])

    assert len(chunks) > 1
    assert "table_part" not in chunks[0].metadata
    bodies = [text.split("\n", 1)[1] for text in contents(chunks)]
    assert not all(body.startswith("| Diagnostic | Cause | Action |") for body in bodies)


def test_split_table_keeps_an_oversized_row_intact() -> None:
    table = "| A | B |\n| --- | --- |\n| " + "x" * 400 + " | y |"
    parts = split_table(table, budget=8, tokenizer=lambda text: text.split())

    assert len(parts) == 1
    assert parts[0] == table


def test_canonical_prose_blocks_are_bounded_contiguous_and_non_overlapping() -> None:
    text = "First paragraph.\n\nSecond paragraph contains café and more words."

    blocks = build_canonical_blocks(
        [DocumentSegment(text=text, page=2, page_end=3, section_path=("Manual", "Use"))],
        max_rendered_bytes=24,
    )

    assert "".join(block.text for block in blocks) == text
    assert [block.ordinal for block in blocks] == list(range(len(blocks)))
    assert all(len(block.rendered_text.encode("utf-8")) <= 24 for block in blocks)
    assert {block.section_path for block in blocks} == {("Manual", "Use")}
    assert {(block.page_start, block.page_end) for block in blocks} == {(2, 3)}


def test_canonical_table_parts_repeat_context_and_store_each_row_once() -> None:
    segment = DocumentSegment(
        text=TABLE,
        page=112,
        section_path=("Diagnostics",),
        is_table=True,
        table_ref="#/tables/0",
        caption="Table 4-2 Diagnostic codes",
    )

    blocks = build_canonical_blocks([segment], max_rendered_bytes=300)

    assert len(blocks) > 1
    assert [block.part for block in blocks] == list(range(1, len(blocks) + 1))
    assert {block.parts for block in blocks} == {len(blocks)}
    assert {block.table_ref for block in blocks} == {"#/tables/0"}
    assert all(len(block.rendered_text.encode("utf-8")) <= 300 for block in blocks)
    assert all(
        block.rendered_text.startswith(
            "Table 4-2 Diagnostic codes\n| Diagnostic | Cause | Action |\n| --- | --- | --- |"
        )
        for block in blocks
    )
    rows = [
        line
        for block in blocks
        for line in block.text.splitlines()
        if line.startswith("| D-")
    ]
    assert rows == [line for line in TABLE.splitlines() if line.startswith("| D-")]


def test_a_caption_only_table_remains_searchable_without_repeating_its_caption() -> None:
    caption = "Table 4-2 Diagnostic codes"
    blocks = build_canonical_blocks(
        [DocumentSegment(text=caption, is_table=True, caption=caption)],
        max_rendered_bytes=64,
    )

    assert len(blocks) == 1
    assert blocks[0].text == caption
    assert blocks[0].rendered_text == caption


def test_canonical_table_rejects_a_row_that_cannot_fit_with_context() -> None:
    segment = DocumentSegment(
        text="| A | B |\n| --- | --- |\n| " + "x" * 80 + " | y |",
        is_table=True,
        caption="Caption",
    )

    with pytest.raises(DocumentTooLargeError, match="max_canonical_block_bytes"):
        build_canonical_blocks([segment], max_rendered_bytes=64)


def test_search_chunks_record_their_contributing_canonical_ordinal() -> None:
    source = node("Purge the circuit. " * 200, block_ordinals=[7])

    chunks = StructuralChunker(chunk_size=64, chunk_overlap=8)([source])

    assert len(chunks) > 1
    assert {tuple(chunk.metadata["block_ordinals"]) for chunk in chunks} == {(7,)}


def test_search_overlap_bridges_adjacent_canonical_blocks_in_the_same_section() -> None:
    source = "x" * 63_991 + " battery management"
    blocks = build_canonical_blocks(
        [
            DocumentSegment(
                text=source,
                section_path=("Manual", "Maintenance"),
                section_ref="#/groups/maintenance",
            )
        ],
        max_rendered_bytes=64_000,
    )
    nodes = [
        node(
            block.text,
            section="Maintenance",
            section_path=list(block.section_path),
            source_section_ref=block.source_section_ref,
            block_ordinals=[block.ordinal],
            page=block.ordinal + 1,
        )
        for block in blocks
    ]

    chunks = StructuralChunker(chunk_size=800, chunk_overlap=120)(nodes)
    bridges = [
        chunk
        for chunk in chunks
        if chunk.metadata["block_ordinals"] == [0, 1]
    ]

    assert len(bridges) == 1
    assert bridges[0].metadata["page_end"] == 2
    assert "page_end" not in bridges[0].get_content(metadata_mode=MetadataMode.EMBED)
    assert "battery management" in bridges[0].get_content(
        metadata_mode=MetadataMode.NONE
    )


def test_search_overlap_does_not_cross_a_recognized_section_boundary() -> None:
    left = node(
        "battery ",
        section="Battery",
        section_path=["Manual", "Battery"],
        source_section_ref="#/groups/battery",
        block_ordinals=[0],
    )
    right = node(
        "management",
        section="Management",
        section_path=["Manual", "Management"],
        source_section_ref="#/groups/management",
        block_ordinals=[1],
    )

    chunks = StructuralChunker(chunk_size=64, chunk_overlap=8)([left, right])

    assert all(chunk.metadata["block_ordinals"] != [0, 1] for chunk in chunks)
    assert all(
        "battery management"
        not in chunk.get_content(metadata_mode=MetadataMode.NONE)
        for chunk in chunks
    )
