"""Current-source outline and exhaustive scan operations."""

from uuid import UUID

from app.config import Settings
from app.cursors import InvalidScanCursorError, ScanCursor, ScanCursorCodec
from app.models import (
    ScanItem,
    ScanRequest,
    ScanResponse,
    SourceOutlineResponse,
    SourceOutlineSection,
)
from app.passages import passage_groups
from app.references import section_reference, source_revision_marker
from app.repository import DocumentOutlineRecord, DocumentRecord, Repository
from app.representation import DocumentBlock, DocumentSection

NO_RELIABLE_SECTIONS_REASON = "No reliable section structure was extracted."
NO_SECTIONS_RECOGNIZED_REASON = "No sections were recognized in this source."


class ScanLimitError(ValueError):
    """Raised when a requested page limit exceeds the deployment bound."""


async def get_source_outline(
    repository: Repository, collection_id: str, external_id: str
) -> SourceOutlineResponse | None:
    """Return the persisted outline for one current source, if it exists."""
    record = await repository.get_document_outline(collection_id, external_id)
    if record is None:
        return None

    public_sections = [section for section in record.sections if not section.is_root]
    items = [_outline_section(record, section) for section in public_sections]
    response_fields: dict[str, object] = {
        "collection_id": collection_id,
        "external_id": external_id,
        "page_count": record.document.page_count,
        "recognized_section_count": record.document.recognized_section_count,
        "recognized_table_count": record.document.recognized_table_count,
        "recognized_figure_count": record.document.recognized_figure_count,
        "sections": items,
    }
    if not items:
        response_fields["reason"] = (
            NO_RELIABLE_SECTIONS_REASON
            if record.document.recognized_section_count is None
            else NO_SECTIONS_RECOGNIZED_REASON
        )
    return SourceOutlineResponse(**response_fields)


async def scan_source(
    repository: Repository, request: ScanRequest, settings: Settings
) -> ScanResponse | None:
    """Return one deterministic, byte-bounded page of source passages."""
    limit = request.limit or settings.default_scan_limit
    if limit > settings.max_scan_limit:
        raise ScanLimitError(
            "limit must not exceed the configured maximum of "
            f"{settings.max_scan_limit}."
        )

    codec = ScanCursorCodec(settings.cursor_signing_key)
    cursor = codec.decode(request.cursor) if request.cursor is not None else None
    if cursor is not None and (
        cursor.collection_id != request.collection_id
        or cursor.external_id != request.external_id
        or cursor.section_ref != request.section_ref
    ):
        raise InvalidScanCursorError(
            "The scan cursor does not match the request scope."
        )

    record = await repository.get_document_scan(
        request.collection_id,
        request.external_id,
        section_ref=request.section_ref,
        after_ordinal=cursor.after_ordinal if cursor is not None else None,
        expected_source_marker=cursor.source_marker if cursor is not None else None,
        limit=limit,
    )
    if record is None:
        return None

    section_by_id = {section.section_id: section for section in record.sections}
    selected_blocks: list[DocumentBlock] = []
    payload_bytes = 0
    for block in record.blocks:
        text = block.rendered_text
        block_bytes = len(text.encode("utf-8"))
        separator_bytes = 2 if selected_blocks else 0
        if (
            selected_blocks
            and payload_bytes + separator_bytes + block_bytes
            > settings.max_scan_payload_bytes
        ):
            break
        if block_bytes > settings.max_scan_payload_bytes:
            raise RuntimeError("A canonical block exceeds the scan payload limit.")
        selected_blocks.append(block)
        payload_bytes += separator_bytes + block_bytes

    groups = passage_groups(
        selected_blocks,
        text_of=lambda block: block.rendered_text,
        target_tokens=settings.chunk_size,
    )
    items = [
        _scan_item(record.document, group, section_by_id) for group in groups
    ]

    has_more = record.has_more or len(selected_blocks) < len(record.blocks)
    next_cursor = None
    if has_more:
        if not items:
            raise RuntimeError("A scan page cannot advance without returning a block.")
        revision_id = record.document.current_revision_id
        if revision_id is None:
            raise RuntimeError("A current document must have a revision.")
        next_cursor = codec.encode(
            ScanCursor(
                collection_id=request.collection_id,
                external_id=request.external_id,
                source_marker=source_revision_marker(
                    record.document.checksum, revision_id
                ),
                section_ref=request.section_ref,
                after_ordinal=selected_blocks[-1].ordinal,
            )
        )
    return ScanResponse(
        collection_id=request.collection_id,
        external_id=request.external_id,
        items=items,
        next_cursor=next_cursor,
    )


def _section_reference(document: DocumentRecord, section: DocumentSection) -> str:
    """Build an opaque handle that changes whenever the source is replaced."""
    revision_id = document.current_revision_id
    if (
        revision_id is None
        or section.is_root
        or section.document_id != document.document_id
        or section.revision_id != revision_id
    ):
        raise ValueError("A public section must belong to the current source revision.")
    return section_reference(
        document.collection_id,
        document.external_id,
        revision_id,
        section.section_id,
    )


def _outline_section(
    record: DocumentOutlineRecord, section: DocumentSection
) -> SourceOutlineSection:
    fields: dict[str, object] = {
        "section_ref": _section_reference(record.document, section),
        "section_path": list(section.path),
    }
    if section.page_start is not None:
        fields["page_start"] = section.page_start
    if section.page_end is not None:
        fields["page_end"] = section.page_end
    return SourceOutlineSection(**fields)


def _scan_item(
    document: DocumentRecord,
    blocks: tuple[DocumentBlock, ...],
    section_by_id: dict[UUID, DocumentSection],
) -> ScanItem:
    fields: dict[str, object] = {
        "text": "\n\n".join(block.rendered_text for block in blocks)
    }
    sections = {block.section_id for block in blocks}
    section = section_by_id[next(iter(sections))] if len(sections) == 1 else None
    if section is not None and not section.is_root:
        fields["section_ref"] = _section_reference(document, section)
        fields["section_path"] = list(section.path)
    page_starts = [block.page_start for block in blocks if block.page_start is not None]
    page_ends = [block.page_end for block in blocks if block.page_end is not None]
    if page_starts:
        fields["page_start"] = min(page_starts)
    if page_ends:
        fields["page_end"] = max(page_ends)
    return ScanItem(**fields)
