"""Minimal client for Azure AI Document Intelligence.

Analysis is asynchronous: submitting a document returns 202 with an
``Operation-Location`` header, and the result is fetched by polling that URL
until the operation reports ``succeeded`` or ``failed``. The prebuilt layout
model is used by default so a scanned PDF gets real OCR instead of the "no
readable text" failure a text-only converter would report, while still
returning the page and heading provenance citations need.

``paragraphs`` in the result carry per-paragraph page numbers and, for
headings, a ``role``. Tables are separate structural objects, so both collections
are merged by their source spans to build the same located text/table segments
Docling produces. Paragraphs fully covered by a table span are omitted because
the rendered table already contains them. When Azure returns no paragraphs,
table spans split the document ``content`` into unlocated prose and structural
table segments.

"""

import asyncio
import json
import time
from itertools import pairwise
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.log import logger
from app.parsing.errors import (
    ConversionFailedError,
    ConversionUnavailableError,
    DocumentTooLargeError,
)
from app.parsing.segments import (
    ConvertedDocument,
    DocumentSegment,
    build_converted_document,
    clean_text,
)

API_VERSION = "2024-11-30"
_ANALYZE_PATH_TEMPLATE = "/documentintelligence/documentModels/{model_id}:analyze"
_POLL_INTERVAL_SECONDS = 2.0
_HEADING_ROLES = frozenset({"title", "sectionHeading"})
_MAX_TABLE_CELLS = 100_000

_FAILED_MESSAGE = "The document converter could not read this file."
_TIMEOUT_MESSAGE = (
    "Document conversion took too long. Try a smaller document or increase the "
    "Azure Document Intelligence conversion timeout."
)
_UNAVAILABLE_MESSAGE = (
    "The document converter is unavailable, so PDF and Office uploads cannot be "
    "processed right now. Plain text and Markdown uploads still work."
)
_NO_TEXT_MESSAGE = "The document converter returned no readable text for this file."


class AzureDocumentIntelligenceClient:
    """Converts one binary document into indexable, located text via Azure."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        client: httpx.AsyncClient,
        *,
        model_id: str,
        timeout_seconds: int,
        max_response_bytes: int,
    ) -> None:
        endpoint = endpoint.rstrip("/")
        self._analyze_url = (
            f"{endpoint}{_ANALYZE_PATH_TEMPLATE.format(model_id=model_id)}"
        )
        # The operation URL Azure hands back is followed with the API key attached;
        # it must resolve to the same endpoint we were configured with, or a
        # misbehaving or compromised responder could redirect the key elsewhere.
        parsed_endpoint = urlsplit(endpoint)
        self._trusted_origin = (parsed_endpoint.scheme, parsed_endpoint.netloc)
        self._api_key = api_key
        self._client = client
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes

    @property
    def name(self) -> str:
        return "azure-document-intelligence"

    async def convert(
        self,
        *,
        filename: str,
        media_type: str,
        content: bytes,
    ) -> ConvertedDocument:
        # One deadline for the whole conversion, started before submission: a
        # slow submit must count against converter.azure.timeout_seconds the same as
        # slow polling does, or the two together could exceed it unnoticed.
        deadline = time.monotonic() + self._timeout_seconds
        operation_url = await self._submit(filename, media_type, content, deadline)
        payload = await self._poll(filename, operation_url, deadline)
        return _extract_segments(payload)

    async def _submit(
        self, filename: str, media_type: str, content: bytes, deadline: float
    ) -> str:
        # Streamed like polling, and for the same reason: a 202 body is normally
        # empty, but an error response from a misbehaving endpoint must still be
        # size-capped before it is read into memory.
        try:
            async with self._client.stream(
                "POST",
                self._analyze_url,
                params={
                    "api-version": API_VERSION,
                    "outputContentFormat": "markdown",
                    "stringIndexType": "unicodeCodePoint",
                },
                headers={
                    "Ocp-Apim-Subscription-Key": self._api_key,
                    "Content-Type": media_type or "application/octet-stream",
                },
                content=content,
                timeout=_remaining(deadline),
            ) as response:
                if response.status_code in (401, 403, 408, 429):
                    logger.warning(
                        "Azure Document Intelligence rejected %s with HTTP %d",
                        filename,
                        response.status_code,
                    )
                    raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE)
                if response.status_code >= 500:
                    raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE)
                await _read_bounded(response, self._max_response_bytes)
                if response.status_code != 202:
                    logger.warning(
                        "Azure Document Intelligence rejected %s with HTTP %d",
                        filename,
                        response.status_code,
                    )
                    raise ConversionFailedError(_FAILED_MESSAGE)
                operation_url = response.headers.get("operation-location")
        except DocumentTooLargeError:
            raise
        except httpx.TimeoutException as exc:
            logger.warning(
                "Azure Document Intelligence submission timed out for %s", filename
            )
            raise ConversionUnavailableError(_TIMEOUT_MESSAGE) from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "Azure Document Intelligence request failed: %s", type(exc).__name__
            )
            raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE) from exc

        if not operation_url or not self._is_trusted(operation_url):
            logger.warning(
                "Azure Document Intelligence accepted %s with no usable operation URL",
                filename,
            )
            raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE)
        return operation_url

    def _is_trusted(self, operation_url: str) -> bool:
        parsed = urlsplit(operation_url)
        return (parsed.scheme, parsed.netloc) == self._trusted_origin

    async def _poll(
        self, filename: str, operation_url: str, deadline: float
    ) -> dict[str, Any]:
        while True:
            if time.monotonic() >= deadline:
                logger.warning(
                    "Azure Document Intelligence timed out analyzing %s", filename
                )
                raise ConversionUnavailableError(_TIMEOUT_MESSAGE)
            body = await self._poll_once(operation_url, deadline)
            payload = _payload(body)
            raw_status = payload.get("status")
            status = raw_status.lower() if isinstance(raw_status, str) else ""
            if status == "succeeded":
                return payload
            if status in ("failed", "canceled", "cancelled"):
                logger.warning(
                    "Azure Document Intelligence failed to analyze %s", filename
                )
                raise ConversionFailedError(_FAILED_MESSAGE)
            if time.monotonic() >= deadline:
                logger.warning(
                    "Azure Document Intelligence timed out analyzing %s", filename
                )
                raise ConversionUnavailableError(_TIMEOUT_MESSAGE)
            await asyncio.sleep(min(_POLL_INTERVAL_SECONDS, _remaining(deadline)))

    async def _poll_once(self, operation_url: str, deadline: float) -> bytes:
        try:
            async with self._client.stream(
                "GET",
                operation_url,
                headers={"Ocp-Apim-Subscription-Key": self._api_key},
                timeout=_remaining(deadline),
            ) as response:
                if response.status_code >= 500:
                    raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE)
                if response.status_code >= 400:
                    logger.warning(
                        "Azure Document Intelligence polling failed with HTTP %d",
                        response.status_code,
                    )
                    raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE)
                return await _read_bounded(response, self._max_response_bytes)
        except DocumentTooLargeError:
            raise
        except httpx.TimeoutException as exc:
            logger.warning("Azure Document Intelligence polling request timed out")
            raise ConversionUnavailableError(_TIMEOUT_MESSAGE) from exc
        except httpx.HTTPError as exc:
            logger.warning(
                "Azure Document Intelligence polling request failed: %s",
                type(exc).__name__,
            )
            raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE) from exc


def _remaining(deadline: float) -> float:
    """Seconds left before *deadline*, floored so httpx never sees zero or negative.

    Used as each request's own timeout, so a single slow request cannot by
    itself push the whole conversion past its configured deadline.
    """
    return max(deadline - time.monotonic(), 0.001)


async def _read_bounded(response: httpx.Response, limit: int) -> bytes:
    chunks: list[bytes] = []
    received = 0
    async for chunk in response.aiter_bytes():
        received += len(chunk)
        if received > limit:
            raise DocumentTooLargeError(
                "The converted document is larger than this service accepts."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _payload(body: bytes) -> dict[str, Any]:
    try:
        payload: Any = json.loads(body)
    except ValueError as exc:
        raise ConversionFailedError(_FAILED_MESSAGE) from exc
    if not isinstance(payload, dict):
        raise ConversionFailedError(_FAILED_MESSAGE)
    return payload


def _extract_segments(payload: dict[str, Any]) -> ConvertedDocument:
    result = payload.get("analyzeResult")
    if not isinstance(result, dict):
        raise ConversionFailedError(_FAILED_MESSAGE)

    raw_paragraphs = result.get("paragraphs")
    raw_tables = result.get("tables")
    if isinstance(raw_paragraphs, list) and raw_paragraphs:
        excluded_paragraph_indexes = _table_covered_paragraph_indexes(
            raw_paragraphs,
            raw_tables if isinstance(raw_tables, list) else [],
        )
        segments = _segments_from_paragraphs(
            raw_paragraphs,
            raw_tables=raw_tables if isinstance(raw_tables, list) else [],
            excluded_paragraph_indexes=excluded_paragraph_indexes,
        )
        if segments:
            return _build_document(
                result,
                segments,
                raw_paragraphs=[
                    paragraph
                    for index, paragraph in enumerate(raw_paragraphs)
                    if index not in excluded_paragraph_indexes
                ],
            )

    content = result.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ConversionFailedError(_NO_TEXT_MESSAGE)
    if isinstance(raw_tables, list) and raw_tables:
        segments = _segments_from_content_and_tables(content, raw_tables)
        if segments:
            return _build_document(
                result,
                segments,
                raw_paragraphs=(
                    raw_paragraphs if isinstance(raw_paragraphs, list) else None
                ),
            )
    return _build_document(
        result,
        [DocumentSegment(text=content)],
        raw_paragraphs=raw_paragraphs if isinstance(raw_paragraphs, list) else None,
    )


def _build_document(
    result: dict[str, Any],
    segments: list[DocumentSegment],
    *,
    raw_paragraphs: list[Any] | None,
) -> ConvertedDocument:
    return build_converted_document(
        segments,
        page_count=_reported_page_count(result),
        recognized_section_count=(
            _recognized_section_count(raw_paragraphs)
            if raw_paragraphs is not None
            else None
        ),
        recognized_table_count=_reported_list_count(result, "tables"),
        recognized_figure_count=_reported_list_count(result, "figures"),
    )


def _segments_from_paragraphs(
    raw_paragraphs: list[Any],
    *,
    raw_tables: list[Any] | None = None,
    excluded_paragraph_indexes: set[int] | None = None,
) -> list[DocumentSegment]:
    """Located paragraphs and tables, carrying forward the current heading.

    Azure reports no chunk hierarchy the way Docling does, so the last heading
    or title paragraph seen is used as the section for the paragraphs that
    follow it -- the same "most specific heading above this text" locator. Source
    spans restore tables to their position among the flat paragraph sequence.
    """
    segments: list[DocumentSegment] = []
    section = ""
    section_ref: str | None = None
    excluded_indexes = excluded_paragraph_indexes or set()
    items: list[tuple[str, int, dict[str, Any]]] = [
        ("paragraph", index, raw)
        for index, raw in enumerate(raw_paragraphs)
        if isinstance(raw, dict) and index not in excluded_indexes
    ]
    items.extend(
        ("table", index, raw)
        for index, raw in enumerate(raw_tables or [])
        if isinstance(raw, dict)
    )
    ordered_items = [
        (offset, item)
        for item in items
        if (offset := _span_offset(item[2])) is not None
    ]
    if items and len(ordered_items) == len(items):
        items = [
            item
            for _, item in sorted(
                ordered_items,
                key=lambda pair: (
                    pair[0],
                    0 if pair[1][0] == "paragraph" else 1,
                    pair[1][1],
                ),
            )
        ]

    for kind, item_index, raw in items:
        if kind == "table":
            text = _table_markdown(raw)
            if not text:
                continue
            page_start, page_end = _page_range(raw.get("boundingRegions"))
            segments.append(
                DocumentSegment(
                    text=text,
                    page=page_start,
                    page_end=page_end,
                    section=section,
                    section_path=(section,) if section else (),
                    section_ref=section_ref,
                    is_table=True,
                    table_ref=f"#/tables/{item_index}",
                    caption=_table_caption(raw),
                )
            )
            continue

        paragraph_text = raw.get("content")
        if not isinstance(paragraph_text, str) or not paragraph_text.strip():
            continue
        if raw.get("role") in _HEADING_ROLES:
            section = paragraph_text.strip()
            # Azure returns a flat ordered paragraph list rather than parent
            # links. Its array position is still a stable identity within this
            # conversion and keeps repeated heading text as distinct sections.
            section_ref = f"#/paragraphs/{item_index}"
        page_start, page_end = _page_range(raw.get("boundingRegions"))
        segments.append(
            DocumentSegment(
                text=paragraph_text,
                page=page_start,
                page_end=page_end,
                section=section,
                section_path=(section,) if section else (),
                section_ref=section_ref,
            )
        )
    return segments


def _table_covered_paragraph_indexes(
    raw_paragraphs: list[Any], raw_tables: list[Any]
) -> set[int]:
    """Find paragraphs whose complete source span is represented by a table."""
    table_ranges = tuple(
        span_range
        for table in raw_tables
        if isinstance(table, dict) and _table_markdown(table)
        for span_range in _valid_span_ranges(table)
    )
    if not table_ranges:
        return set()
    return {
        index
        for index, paragraph in enumerate(raw_paragraphs)
        if isinstance(paragraph, dict) and _spans_are_covered(paragraph, table_ranges)
    }


def _spans_are_covered(
    value: dict[str, Any], covering_ranges: tuple[tuple[int, int], ...]
) -> bool:
    raw_spans = value.get("spans")
    if not isinstance(raw_spans, list) or not raw_spans:
        return False

    ranges: list[tuple[int, int]] = []
    for raw_span in raw_spans:
        if not isinstance(raw_span, dict):
            return False
        offset = _non_negative_int(raw_span.get("offset"))
        length = _positive_int(raw_span.get("length"))
        if offset is None or length is None:
            return False
        ranges.append((offset, offset + length))

    return all(
        any(
            cover_start <= start and end <= cover_end
            for cover_start, cover_end in covering_ranges
        )
        for start, end in ranges
    )


def _valid_span_ranges(value: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    raw_spans = value.get("spans")
    if not isinstance(raw_spans, list):
        return ()
    ranges: list[tuple[int, int]] = []
    for raw_span in raw_spans:
        if not isinstance(raw_span, dict):
            continue
        offset = _non_negative_int(raw_span.get("offset"))
        length = _positive_int(raw_span.get("length"))
        if offset is not None and length is not None:
            ranges.append((offset, offset + length))
    return tuple(ranges)


def _segments_from_content_and_tables(
    content: str, raw_tables: list[Any]
) -> list[DocumentSegment]:
    tables: list[tuple[int, int, int, dict[str, Any], str]] = []
    for table_index, raw in enumerate(raw_tables):
        if not isinstance(raw, dict):
            continue
        text = _table_markdown(raw)
        span = _span_range(raw, content_length=len(content))
        if text and span is not None:
            tables.append((*span, table_index, raw, text))
    if not tables:
        return []

    tables.sort(key=lambda item: (item[0], item[1], item[2]))
    if any(right[0] < left[1] for left, right in pairwise(tables)):
        return []

    segments: list[DocumentSegment] = []
    cursor = 0
    for start, end, table_index, raw, table_text in tables:
        if prose := content[cursor:start].strip():
            segments.append(DocumentSegment(text=prose))
        page_start, page_end = _page_range(raw.get("boundingRegions"))
        segments.append(
            DocumentSegment(
                text=table_text,
                page=page_start,
                page_end=page_end,
                is_table=True,
                table_ref=f"#/tables/{table_index}",
                caption=_table_caption(raw),
            )
        )
        cursor = end
    if prose := content[cursor:].strip():
        segments.append(DocumentSegment(text=prose))
    return segments


def _span_offset(value: dict[str, Any]) -> int | None:
    spans = value.get("spans")
    if not isinstance(spans, list):
        return None
    offsets: list[int] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        offset = _non_negative_int(span.get("offset"))
        if offset is not None:
            offsets.append(offset)
    return min(offsets) if offsets else None


def _span_range(
    value: dict[str, Any], *, content_length: int
) -> tuple[int, int] | None:
    spans = value.get("spans")
    if not isinstance(spans, list):
        return None
    ranges: list[tuple[int, int]] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        offset = _non_negative_int(span.get("offset"))
        length = _positive_int(span.get("length"))
        if offset is None or length is None or offset + length > content_length:
            continue
        ranges.append((offset, offset + length))
    if not ranges:
        return None
    return min(start for start, _ in ranges), max(end for _, end in ranges)


def _table_markdown(table: dict[str, Any]) -> str:
    raw_cells = table.get("cells")
    if not isinstance(raw_cells, list):
        return ""
    cells = [cell for cell in raw_cells if isinstance(cell, dict)]
    located: list[tuple[dict[str, Any], int, int]] = []
    for cell in cells:
        row_index = _non_negative_int(cell.get("rowIndex"))
        column_index = _non_negative_int(cell.get("columnIndex"))
        if row_index is not None and column_index is not None:
            located.append((cell, row_index, column_index))
    if not located:
        return ""

    inferred_rows = max(
        row_index + (_positive_int(cell.get("rowSpan")) or 1)
        for cell, row_index, _ in located
    )
    inferred_columns = max(
        column_index + (_positive_int(cell.get("columnSpan")) or 1)
        for cell, _, column_index in located
    )
    row_count = max(_positive_int(table.get("rowCount")) or 0, inferred_rows)
    column_count = max(_positive_int(table.get("columnCount")) or 0, inferred_columns)
    if (
        row_count > _MAX_TABLE_CELLS
        or column_count > _MAX_TABLE_CELLS
        or row_count * column_count > _MAX_TABLE_CELLS
    ):
        raise ConversionFailedError(_FAILED_MESSAGE)
    rows = [["" for _ in range(column_count)] for _ in range(row_count)]
    has_content = False
    for cell, row_index, column_index in located:
        content = cell.get("content")
        text = _markdown_cell(content) if isinstance(content, str) else ""
        rows[row_index][column_index] = text
        has_content = has_content or bool(text)
    if not has_content:
        return ""

    first_row_is_header = any(
        row_index == 0 and cell.get("kind") == "columnHeader"
        for cell, row_index, _ in located
    )
    header = rows[0] if first_row_is_header else [""] * column_count
    body = rows[1:] if first_row_is_header else rows
    return "\n".join(
        (
            _markdown_row(header),
            _markdown_row(["---"] * column_count),
            *(_markdown_row(row) for row in body),
        )
    )


def _table_caption(table: dict[str, Any]) -> str:
    caption = table.get("caption")
    if isinstance(caption, dict):
        caption = caption.get("content")
    return caption.strip() if isinstance(caption, str) else ""


def _markdown_cell(value: str) -> str:
    return "<br>".join(part.strip() for part in value.splitlines()).replace("|", r"\|")


def _markdown_row(cells: list[str]) -> str:
    return f"| {' | '.join(cells)} |"


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _page_range(value: Any) -> tuple[int | None, int | None]:
    """The first and last pages touched by one paragraph."""
    if not isinstance(value, list):
        return None, None
    pages: list[int] = []
    for region in value:
        if not isinstance(region, dict):
            continue
        page = region.get("pageNumber")
        if isinstance(page, int) and not isinstance(page, bool) and page >= 1:
            pages.append(page)
    return (min(pages), max(pages)) if pages else (None, None)


def _reported_page_count(result: dict[str, Any]) -> int | None:
    pages = result.get("pages")
    return len(pages) if isinstance(pages, list) else None


def _reported_list_count(result: dict[str, Any], key: str) -> int | None:
    value = result.get(key)
    return len(value) if isinstance(value, list) else None


def _recognized_section_count(paragraphs: list[Any]) -> int:
    return sum(
        1
        for paragraph in paragraphs
        if isinstance(paragraph, dict)
        and paragraph.get("role") in _HEADING_ROLES
        and isinstance(paragraph.get("content"), str)
        and bool(clean_text(paragraph["content"]))
    )
