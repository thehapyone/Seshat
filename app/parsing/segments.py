"""Located pieces of a normalized document."""

from dataclasses import dataclass
from typing import Any

# Matches the section field length the ingest contract accepts.
MAXIMUM_SECTION_CHARACTERS = 512

_CONTROL_CHARACTERS = {code: None for code in range(32) if code not in (9, 10)}


@dataclass(frozen=True, slots=True)
class DocumentSegment:
    """One located run of text."""

    text: str
    page: int | None = None
    page_end: int | None = None
    section: str = ""
    section_path: tuple[str, ...] = ()
    section_ref: str | None = None
    # Read from the converter's document structure, never inferred from the text:
    # how a table is serialized is the converter's choice.
    is_table: bool = False
    table_ref: str | None = None
    caption: str = ""


@dataclass(frozen=True, slots=True)
class ConvertedDocument:
    """Normalized text plus the provenance carried alongside it.

    ``text`` is the whole normalized document as one string: it is what a preview
    shows and what the normalized checksum covers. Source change detection uses
    the caller's actual text or uploaded bytes, independent of converter chunking.
    """

    text: str
    segments: tuple[DocumentSegment, ...]
    page_count: int | None = None
    recognized_section_count: int | None = None
    recognized_table_count: int | None = None
    recognized_figure_count: int | None = None

    @property
    def has_provenance(self) -> bool:
        return any(
            segment.page is not None or segment.section_path or segment.section
            for segment in self.segments
        )


def clean_text(text: str) -> str:
    """Normalize newlines and drop control characters that break chunking."""
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .translate(_CONTROL_CHARACTERS)
        .strip()
    )


_INFER_PAGE_COUNT = object()


def build_converted_document(
    segments: list[DocumentSegment],
    *,
    page_count: int | None | object = _INFER_PAGE_COUNT,
    recognized_section_count: int | None = None,
    recognized_table_count: int | None = None,
    recognized_figure_count: int | None = None,
) -> ConvertedDocument:
    """Clean and bound *segments*, dropping the ones that hold no text."""
    cleaned: list[DocumentSegment] = []
    pages: list[int] = []
    for segment in segments:
        text = clean_text(segment.text)
        if not text:
            continue
        if segment.page is not None:
            pages.append(segment.page)
        if segment.page_end is not None:
            pages.append(segment.page_end)
        section_path = _clean_section_path(segment)
        cleaned.append(
            DocumentSegment(
                text=text,
                page=segment.page,
                page_end=segment.page_end
                if segment.page_end is not None
                else segment.page,
                section=section_path[-1] if section_path else "",
                section_path=section_path,
                section_ref=_optional_clean_text(segment.section_ref),
                is_table=segment.is_table,
                table_ref=_optional_clean_text(segment.table_ref),
                caption=clean_text(segment.caption)[:MAXIMUM_SECTION_CHARACTERS],
            )
        )
    resolved_page_count = max(pages) if pages else None
    if page_count is not _INFER_PAGE_COUNT:
        resolved_page_count = page_count if isinstance(page_count, int) else None
    return ConvertedDocument(
        text="\n\n".join(segment.text for segment in cleaned),
        segments=tuple(cleaned),
        page_count=resolved_page_count,
        recognized_section_count=recognized_section_count,
        recognized_table_count=recognized_table_count,
        recognized_figure_count=recognized_figure_count,
    )


def _clean_section_path(segment: DocumentSegment) -> tuple[str, ...]:
    raw = segment.section_path or ((segment.section,) if segment.section else ())
    return tuple(
        cleaned[:MAXIMUM_SECTION_CHARACTERS]
        for item in raw
        if (cleaned := clean_text(item))
    )


def _optional_clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = clean_text(value)
    return cleaned or None
