"""Persistable structure derived from immutable canonical document blocks."""

from dataclasses import dataclass, replace
from uuid import UUID, uuid5

from app.embeddings.chunking import CanonicalBlock


@dataclass(frozen=True, slots=True)
class DocumentSection:
    """One current outline node, including Seshat's private root section."""

    section_id: UUID
    document_id: UUID
    revision_id: UUID
    parent_section_id: UUID | None
    ordinal: int
    title: str
    path: tuple[str, ...]
    source_ref: str | None
    first_block_ordinal: int | None = None
    last_block_ordinal: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    is_root: bool = False


@dataclass(frozen=True, slots=True)
class DocumentBlock:
    """One bounded, ordered, non-overlapping canonical reading unit."""

    document_id: UUID
    revision_id: UUID
    section_id: UUID
    ordinal: int
    kind: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    table_id: str | None = None
    table_caption: str = ""
    table_header: str = ""
    part: int | None = None
    parts: int | None = None

    @property
    def rendered_text(self) -> str:
        """Return the self-contained public text for this canonical block."""
        if self.kind == "text":
            return self.text
        return "\n".join(
            value
            for value in (self.table_caption, self.table_header, self.text)
            if value
        )


@dataclass(frozen=True, slots=True)
class DocumentStructure:
    """Current sections and blocks plus the section assigned to each block."""

    sections: tuple[DocumentSection, ...]
    blocks: tuple[DocumentBlock, ...]
    block_section_ids: dict[int, UUID]


def build_document_structure(
    document_id: UUID,
    revision_id: UUID,
    canonical_blocks: tuple[CanonicalBlock, ...],
) -> DocumentStructure:
    """Build a deterministic outline and attach every block to its deepest section."""
    root_id = uuid5(revision_id, "section:root")
    sections: list[DocumentSection] = [
        DocumentSection(
            section_id=root_id,
            document_id=document_id,
            revision_id=revision_id,
            parent_section_id=None,
            ordinal=0,
            title="",
            path=(),
            source_ref=None,
            is_root=True,
        )
    ]
    parent_by_section: dict[UUID, UUID | None] = {root_id: None}
    section_index: dict[UUID, int] = {root_id: 0}
    block_section_ids: dict[int, UUID] = {}
    active_path: tuple[str, ...] = ()
    active_ids: list[UUID] = [root_id]

    for block in canonical_blocks:
        common_depth = 0
        for depth, (active_title, title) in enumerate(
            zip(active_path, block.section_path, strict=False), start=1
        ):
            if active_title != title:
                break
            active_section = sections[section_index[active_ids[depth]]]
            if (
                depth == len(block.section_path)
                and block.source_section_ref is not None
                and active_section.source_ref not in (None, block.source_section_ref)
            ):
                break
            common_depth = depth

        active_ids = active_ids[: common_depth + 1]
        parent_id = active_ids[-1]
        for depth, title in enumerate(
            block.section_path[common_depth:], start=common_depth + 1
        ):
            source_ref = (
                block.source_section_ref if depth == len(block.section_path) else None
            )
            section_id = uuid5(
                revision_id,
                f"section:{len(sections)}:{parent_id}:{title}:{source_ref or ''}",
            )
            parent_by_section[section_id] = parent_id
            section_index[section_id] = len(sections)
            sections.append(
                DocumentSection(
                    section_id=section_id,
                    document_id=document_id,
                    revision_id=revision_id,
                    parent_section_id=parent_id,
                    ordinal=len(sections),
                    title=title,
                    path=block.section_path[:depth],
                    source_ref=source_ref,
                )
            )
            parent_id = section_id
            active_ids.append(section_id)

        if (
            block.section_path
            and block.source_section_ref is not None
            and common_depth == len(block.section_path)
        ):
            leaf_id = active_ids[-1]
            leaf_index = section_index[leaf_id]
            if sections[leaf_index].source_ref is None:
                sections[leaf_index] = replace(
                    sections[leaf_index], source_ref=block.source_section_ref
                )

        active_path = block.section_path
        block_section_ids[block.ordinal] = active_ids[-1]

    ranges: dict[UUID, list[CanonicalBlock]] = {
        section.section_id: [] for section in sections
    }
    for block in canonical_blocks:
        section_id = block_section_ids[block.ordinal]
        ranges[root_id].append(block)
        current_id: UUID | None = section_id
        while current_id is not None and current_id != root_id:
            ranges[current_id].append(block)
            current_id = parent_by_section[current_id]

    located_sections = tuple(
        _with_range(section, ranges[section.section_id]) for section in sections
    )
    blocks = tuple(
        DocumentBlock(
            document_id=document_id,
            revision_id=revision_id,
            section_id=block_section_ids[block.ordinal],
            ordinal=block.ordinal,
            kind=block.kind,
            text=block.text,
            page_start=block.page_start,
            page_end=block.page_end,
            table_id=block.table_ref,
            table_caption=block.table_caption,
            table_header=block.table_header,
            part=block.part,
            parts=block.parts,
        )
        for block in canonical_blocks
    )
    return DocumentStructure(
        sections=located_sections,
        blocks=blocks,
        block_section_ids=block_section_ids,
    )


def _with_range(
    section: DocumentSection, blocks: list[CanonicalBlock]
) -> DocumentSection:
    if not blocks:
        return section
    pages_start = [block.page_start for block in blocks if block.page_start is not None]
    pages_end = [block.page_end for block in blocks if block.page_end is not None]
    return replace(
        section,
        first_block_ordinal=blocks[0].ordinal,
        last_block_ordinal=blocks[-1].ordinal,
        page_start=min(pages_start) if pages_start else None,
        page_end=max(pages_end) if pages_end else None,
    )
