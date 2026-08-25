"""Client for a docling-serve instance, driven through its asynchronous task API.

``POST {base}/v1/chunk/hierarchical/file/async``
    Submits one file and answers ``{"task_id": ..., "task_status": ...}``. This is
    the asynchronous form of the hierarchical chunk route, so it keeps the
    structural provenance the viewer needs -- per-chunk ``page_numbers``,
    ``headings``, ``captions`` and ``doc_items``.

    ``chunking_use_markdown_tables`` is requested because the chunker otherwise
    serializes a table as triplets, which costs a large fraction of a tabular
    document's text. Markdown rows also make an oversized table splittable on row
    boundaries, and ``doc_items`` is what says a chunk is a table in the first
    place: its entries are document self-references such as ``#/tables/0``.

``GET {base}/v1/status/poll/{task_id}``
    Reports ``pending``, ``started``, ``success``, ``partial_success`` or
    ``failure`` for a submitted task. Seshat accepts only complete success.

``GET {base}/v1/result/{task_id}``
    Returns the completed ``{"chunks": [...], "documents": [...]}`` payload.
"""

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from app.log import logger
from app.parsing.errors import (
    ConversionDeadlineExceededError,
    ConversionFailedError,
    ConversionResultUnavailableError,
    ConversionSubmissionError,
    ConversionTaskLostError,
    ConversionUnavailableError,
    DocumentTooLargeError,
)
from app.parsing.segments import (
    ConvertedDocument,
    DocumentSegment,
    build_converted_document,
)

CHUNK_ASYNC_PATH = "/v1/chunk/hierarchical/file/async"
STATUS_PATH = "/v1/status/poll"
RESULT_PATH = "/v1/result"

_TABLE_REFERENCE_PREFIX = "#/tables/"

_PENDING_STATUSES = frozenset({"pending", "started"})
_SUCCESS_STATUS = "success"
# Do not fall back to Markdown when the required asynchronous route is absent;
# that would silently lose page provenance.
_ROUTE_ABSENT_STATUSES = frozenset({404, 405, 501})
# These statuses describe temporary converter unavailability, not a bad document.
_UNAVAILABLE_STATUSES = frozenset({401, 403, 408, 429})
# Bound one continuous converter outage without discarding a long conversion for
# a brief gateway failure. The overall conversion deadline still applies.
_TRANSIENT_FAILURE_GRACE_SECONDS = 300.0
# Submit and status responses are small; cap them before buffering malformed bodies.
_STATUS_RESPONSE_BYTES = 64 * 1024
# Chunk results repeat metadata, so they need a larger cap than converted text.
_RESULT_RESPONSE_FACTOR = 4

_FAILED_MESSAGE = "The document converter could not read this file."
_SUBMIT_REJECTED_MESSAGE = (
    "The document converter refused this file. Check that it is a readable document of "
    "a supported type."
)
_DEADLINE_MESSAGE = (
    "Document conversion did not finish within the conversion deadline. Try a smaller "
    "document or increase the Docling conversion deadline."
)
_TASK_LOST_MESSAGE = (
    "The document converter no longer holds this conversion, so it could not be "
    "finished. Upload the file again to start a new conversion."
)
_RESULT_MESSAGE = (
    "The document converter finished but its result could not be retrieved. Upload the "
    "file again."
)
_UNAVAILABLE_MESSAGE = (
    "The document converter is unavailable, so PDF and Office uploads cannot be "
    "processed right now. Plain text and Markdown uploads still work."
)
_ROUTE_ABSENT_MESSAGE = (
    "The document converter does not offer the asynchronous chunking API this service "
    "requires. Ask the service owner to run a supported docling-serve version."
)
_NO_TEXT_MESSAGE = (
    "The document converter returned no readable text for this file. A scanned "
    "document needs OCR, which this evaluation service does not run."
)


def _monotonic() -> float:
    """Indirection so tests can drive a long conversion without waiting for one."""
    return time.monotonic()


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


class DoclingClient:
    """Converts one binary document into indexable, located text."""

    def __init__(
        self,
        base_url: str,
        client: httpx.AsyncClient,
        *,
        max_response_bytes: int,
        request_timeout_seconds: float,
        deadline_seconds: float,
        poll_interval_seconds: float,
    ) -> None:
        base = base_url.rstrip("/")
        self._submit_url = f"{base}{CHUNK_ASYNC_PATH}"
        self._status_url = f"{base}{STATUS_PATH}"
        self._result_url = f"{base}{RESULT_PATH}"
        self._client = client
        self._max_response_bytes = max_response_bytes
        self._request_timeout_seconds = request_timeout_seconds
        self._deadline_seconds = deadline_seconds
        self._poll_interval_seconds = poll_interval_seconds

    @property
    def name(self) -> str:
        return "docling"

    @property
    def deadline_seconds(self) -> float:
        return self._deadline_seconds

    async def convert(
        self,
        *,
        filename: str,
        media_type: str,
        content: bytes,
    ) -> ConvertedDocument:
        """Submit and follow one conversion without persisting the task id."""
        task_id = await self.submit(
            filename=filename, media_type=media_type, content=content
        )
        return await self.await_result(task_id, submitted_at=datetime.now(UTC))

    async def submit(
        self,
        *,
        filename: str,
        media_type: str,
        content: bytes,
    ) -> str:
        """Enqueue one conversion and return the converter's task id."""
        body = await self._request(
            "POST",
            self._submit_url,
            limit=_STATUS_RESPONSE_BYTES,
            timeout=self._request_timeout_seconds,
            files={"files": (filename, content, media_type)},
            data={
                "include_converted_doc": "false",
                "chunking_use_markdown_tables": "true",
            },
            rejected=ConversionSubmissionError(_SUBMIT_REJECTED_MESSAGE),
            route_absent=ConversionUnavailableError(_ROUTE_ABSENT_MESSAGE),
        )
        payload = _payload(body)
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            logger.warning("Docling accepted %s without returning a task id", filename)
            raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE)
        _reject_terminal_status(payload.get("task_status"))
        return task_id.strip()

    async def await_result(
        self, task_id: str, *, submitted_at: datetime
    ) -> ConvertedDocument:
        """Poll *task_id* to completion and return the converted document."""
        deadline = _monotonic() + self._remaining_seconds(submitted_at)
        # Reset after a successful poll so this measures one continuous outage.
        failing_since: float | None = None
        while True:
            if _monotonic() >= deadline:
                logger.warning("Docling conversion %s exceeded its deadline", task_id)
                raise ConversionDeadlineExceededError(_DEADLINE_MESSAGE)
            try:
                if await self._is_complete(task_id, deadline):
                    # Retry result retrieval: conversion already succeeded.
                    return await self._result(task_id)
            except ConversionUnavailableError:
                now = _monotonic()
                failing_since = now if failing_since is None else failing_since
                if now - failing_since >= _TRANSIENT_FAILURE_GRACE_SECONDS:
                    logger.warning(
                        "Docling has been unreachable for %.0fs while converting %s; giving up",
                        now - failing_since,
                        task_id,
                    )
                    raise
                logger.warning(
                    "Docling is answering transiently for %s; retrying", task_id
                )
            else:
                failing_since = None
            await _sleep(min(self._poll_interval_seconds, _remaining(deadline)))

    def _remaining_seconds(self, submitted_at: datetime) -> float:
        # A naive timestamp would otherwise raise here; treating it as UTC keeps a
        # resume working against a row written by an older revision.
        anchored = (
            submitted_at if submitted_at.tzinfo else submitted_at.replace(tzinfo=UTC)
        )
        elapsed = (datetime.now(UTC) - anchored).total_seconds()
        remaining = self._deadline_seconds - max(elapsed, 0.0)
        if remaining <= 0:
            raise ConversionDeadlineExceededError(_DEADLINE_MESSAGE)
        return remaining

    async def _is_complete(self, task_id: str, deadline: float) -> bool:
        body = await self._request(
            "GET",
            f"{self._status_url}/{task_id}",
            limit=_STATUS_RESPONSE_BYTES,
            timeout=min(self._request_timeout_seconds, _remaining(deadline)),
            rejected=ConversionTaskLostError(_TASK_LOST_MESSAGE),
            route_absent=ConversionTaskLostError(_TASK_LOST_MESSAGE),
        )
        status = _status_of(_payload(body).get("task_status"))
        if status == _SUCCESS_STATUS:
            return True
        if status in _PENDING_STATUSES:
            return False
        logger.warning("Docling reported task status %r for %s", status, task_id)
        raise ConversionFailedError(_FAILED_MESSAGE)

    async def _result(self, task_id: str) -> ConvertedDocument:
        # The conversion deadline does not cut off retrieval after completion.
        body = await self._request(
            "GET",
            f"{self._result_url}/{task_id}",
            limit=self._max_response_bytes * _RESULT_RESPONSE_FACTOR,
            timeout=self._request_timeout_seconds,
            rejected=ConversionResultUnavailableError(_RESULT_MESSAGE),
            route_absent=ConversionTaskLostError(_TASK_LOST_MESSAGE),
        )
        return _extract_segments(_payload(body))

    async def _request(
        self,
        method: str,
        url: str,
        *,
        limit: int,
        timeout: float,
        rejected: Exception,
        route_absent: Exception,
        files: dict[str, tuple[str, bytes, str]] | None = None,
        data: dict[str, str] | None = None,
    ) -> bytes:
        """Perform one bounded request, mapping transport and status to a document error."""
        try:
            async with self._client.stream(
                method,
                url,
                files=files,
                data=data,
                headers={"Accept": "application/json"},
                timeout=timeout,
            ) as response:
                if response.status_code in _ROUTE_ABSENT_STATUSES:
                    await _drain(response, _STATUS_RESPONSE_BYTES)
                    logger.warning(
                        "Docling answered %s with HTTP %d",
                        _url_path(url),
                        response.status_code,
                    )
                    raise route_absent
                if (
                    response.status_code >= 500
                    or response.status_code in _UNAVAILABLE_STATUSES
                ):
                    await _drain(response, _STATUS_RESPONSE_BYTES)
                    logger.warning(
                        "Docling answered %s with HTTP %d",
                        _url_path(url),
                        response.status_code,
                    )
                    raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE)
                body = await _read_bounded(response, limit)
                if response.status_code >= 400:
                    logger.warning(
                        "Docling answered %s with HTTP %d",
                        _url_path(url),
                        response.status_code,
                    )
                    raise rejected
                return body
        except (DocumentTooLargeError, ConversionUnavailableError):
            raise
        except httpx.TimeoutException as exc:
            logger.warning("Docling request to %s timed out", _url_path(url))
            raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE) from exc
        except httpx.HTTPError as exc:
            logger.warning("Docling request failed: %s", type(exc).__name__)
            raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE) from exc


def _url_path(url: str) -> str:
    """The path of *url*, so a log line never carries the converter's address."""
    _, _, rest = url.partition("://")
    _, slash, path = rest.partition("/")
    return f"{slash}{path}"


def _remaining(deadline: float) -> float:
    """Seconds left before *deadline*, floored so httpx never sees zero or negative."""
    return max(deadline - _monotonic(), 0.001)


def _status_of(value: Any) -> str:
    return value.lower() if isinstance(value, str) else ""


def _reject_terminal_status(value: Any) -> None:
    """Refuse a submission that came back already failed, before any polling."""
    status = _status_of(value)
    if status and status not in _PENDING_STATUSES and status != _SUCCESS_STATUS:
        logger.warning("Docling rejected the submission with status %r", status)
        raise ConversionFailedError(_FAILED_MESSAGE)


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


async def _drain(response: httpx.Response, limit: int) -> None:
    """Read and discard an error body, so the connection can be reused."""
    try:
        await _read_bounded(response, limit)
    except (DocumentTooLargeError, httpx.HTTPError):
        pass


def _payload(body: bytes) -> dict[str, Any]:
    try:
        payload: Any = json.loads(body)
    except ValueError as exc:
        raise ConversionFailedError(_FAILED_MESSAGE) from exc
    if not isinstance(payload, dict):
        raise ConversionFailedError(_FAILED_MESSAGE)
    return payload


def _reject_failed_status(status: Any) -> None:
    if isinstance(status, str) and status.lower() != _SUCCESS_STATUS:
        logger.warning("Docling reported conversion status %r", status)
        raise ConversionFailedError(_FAILED_MESSAGE)


def _extract_segments(payload: dict[str, Any]) -> ConvertedDocument:
    """Turn a chunk result into located segments."""
    if payload.get("kind") == "TaskFailureResult":
        logger.warning("Docling returned a task failure result")
        raise ConversionFailedError(_FAILED_MESSAGE)
    for document in payload.get("documents") or ():
        if isinstance(document, dict):
            _reject_failed_status(document.get("status"))

    raw_chunks = payload.get("chunks")
    if not isinstance(raw_chunks, list):
        raise ConversionFailedError(_FAILED_MESSAGE)

    segments: list[DocumentSegment] = []
    for raw in raw_chunks:
        if not isinstance(raw, dict):
            continue
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        page_start, page_end = _page_range(raw.get("page_numbers"))
        heading_path = _heading_path(raw.get("headings"))
        segments.append(
            DocumentSegment(
                text=text,
                page=page_start,
                page_end=page_end,
                section=heading_path[-1] if heading_path else "",
                section_path=heading_path,
                section_ref=_optional_text(raw.get("section_ref")),
                is_table=_is_table(raw.get("doc_items")),
                table_ref=_first_reference(
                    raw.get("doc_items"), (_TABLE_REFERENCE_PREFIX,)
                ),
                caption=_first_caption(raw.get("captions")),
            )
        )
    if not segments:
        raise ConversionFailedError(_NO_TEXT_MESSAGE)
    return build_converted_document(
        segments,
        # Hierarchical chunks expose page ranges for their text, but not an exact
        # source page count: blank or figure-only pages may produce no chunk.
        page_count=None,
        # Hierarchical chunks carry useful heading paths and section references,
        # but neither is an authoritative enumeration of the source outline.
        recognized_section_count=None,
        recognized_table_count=_recognized_reference_count(
            raw_chunks, (_TABLE_REFERENCE_PREFIX,)
        ),
        # A figure without text need not appear in the chunk result, so references
        # found on text chunks cannot establish the whole-source figure count.
        recognized_figure_count=None,
    )


def _page_range(value: Any) -> tuple[int | None, int | None]:
    """The first and last valid pages reported for one converter chunk."""
    if not isinstance(value, list):
        return None, None
    pages = [
        item
        for item in value
        if isinstance(item, int) and not isinstance(item, bool) and item >= 1
    ]
    return (min(pages), max(pages)) if pages else (None, None)


def _is_table(value: Any) -> bool:
    """Whether this chunk came from a table, per the converter's own references.

    ``doc_items`` holds self-references such as ``#/tables/0``, so a table is
    recognized by where its content came from, not by how it was serialized.
    """
    return _first_reference(value, (_TABLE_REFERENCE_PREFIX,)) is not None


def _first_caption(value: Any) -> str:
    """The first caption the converter attached, which names the table."""
    if not isinstance(value, list):
        return ""
    for caption in value:
        if isinstance(caption, str) and caption.strip():
            return caption.strip()
    return ""


def _heading_path(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        heading.strip()
        for heading in value
        if isinstance(heading, str) and heading.strip()
    )


def _first_reference(value: Any, prefixes: tuple[str, ...]) -> str | None:
    if not isinstance(value, list):
        return None
    for reference in value:
        if isinstance(reference, str) and reference.startswith(prefixes):
            return reference
    return None


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _recognized_reference_count(
    raw_chunks: list[Any], prefixes: tuple[str, ...]
) -> int | None:
    if not raw_chunks or any(
        not isinstance(raw, dict) or not isinstance(raw.get("doc_items"), list)
        for raw in raw_chunks
    ):
        return None
    references = {
        reference
        for raw in raw_chunks
        for reference in raw["doc_items"]
        if isinstance(reference, str) and reference.startswith(prefixes)
    }
    return len(references)
