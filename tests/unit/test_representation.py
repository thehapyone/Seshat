"""Current canonical section and block representation."""

from uuid import uuid4

from app.embeddings.chunking import CanonicalBlock
from app.representation import build_document_structure


def test_structure_preserves_hierarchy_ranges_and_repeated_heading_identity() -> None:
    document_id = uuid4()
    revision_id = uuid4()
    blocks = (
        CanonicalBlock(
            ordinal=0,
            kind="text",
            text="First maintenance section.",
            page_start=2,
            page_end=3,
            section_path=("Manual", "Maintenance"),
            source_section_ref="#/groups/a",
        ),
        CanonicalBlock(
            ordinal=1,
            kind="text",
            text="More from the first section.",
            page_start=4,
            page_end=4,
            section_path=("Manual", "Maintenance"),
            # Some converter chunks omit a repeated structural reference. The
            # contiguous path still belongs to the active section above.
            source_section_ref=None,
        ),
        CanonicalBlock(
            ordinal=2,
            kind="text",
            text="A distinct section with the same heading.",
            page_start=8,
            page_end=8,
            section_path=("Manual", "Maintenance"),
            source_section_ref="#/groups/b",
        ),
        CanonicalBlock(
            ordinal=3,
            kind="text",
            text="Alarm section.",
            page_start=10,
            page_end=10,
            section_path=("Manual", "Alarms"),
            source_section_ref="#/groups/c",
        ),
    )

    structure = build_document_structure(document_id, revision_id, blocks)

    assert [section.path for section in structure.sections] == [
        (),
        ("Manual",),
        ("Manual", "Maintenance"),
        ("Manual", "Maintenance"),
        ("Manual", "Alarms"),
    ]
    assert [section.source_ref for section in structure.sections] == [
        None,
        None,
        "#/groups/a",
        "#/groups/b",
        "#/groups/c",
    ]
    root, manual, first_maintenance, second_maintenance, alarms = structure.sections
    assert root.is_root is True
    assert (root.first_block_ordinal, root.last_block_ordinal) == (0, 3)
    assert (root.page_start, root.page_end) == (2, 10)
    assert (manual.first_block_ordinal, manual.last_block_ordinal) == (0, 3)
    assert first_maintenance.parent_section_id == manual.section_id
    assert (
        first_maintenance.first_block_ordinal,
        first_maintenance.last_block_ordinal,
    ) == (0, 1)
    assert second_maintenance.section_id != first_maintenance.section_id
    assert alarms.parent_section_id == manual.section_id
    assert [block.section_id for block in structure.blocks] == [
        first_maintenance.section_id,
        first_maintenance.section_id,
        second_maintenance.section_id,
        alarms.section_id,
    ]


def test_unstructured_content_is_attached_only_to_the_private_root() -> None:
    structure = build_document_structure(
        uuid4(),
        uuid4(),
        (CanonicalBlock(ordinal=0, kind="text", text="Unstructured text."),),
    )

    assert len(structure.sections) == 1
    assert structure.sections[0].is_root is True
    assert structure.sections[0].path == ()
    assert structure.blocks[0].section_id == structure.sections[0].section_id
