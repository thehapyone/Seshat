"""Docling adapter behaviour, driven by a mock transport rather than a live server.

Conversion goes through docling-serve's asynchronous task API, so what these
tests hold it to is the three-request shape of that contract: submit once, poll
until the task settles, then fetch the result -- plus the failure mapping that
tells a caller whether uploading the same file again is worth trying.
"""

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app.parsing import (
    ConversionDeadlineExceededError,
    ConversionFailedError,
    ConversionResultUnavailableError,
    ConversionSubmissionError,
    ConversionTaskLostError,
    ConversionUnavailableError,
    DoclingClient,
    DocumentTooLargeError,
    ResumableDocumentConverter,
)
from app.parsing.docling import CHUNK_ASYNC_PATH, RESULT_PATH, STATUS_PATH

BASE_URL = "http://docling:5001"
TASK_ID = "3f6d0a1e"


class FakeClock:
    """A monotonic clock that only moves when the client sleeps.

    It is what makes a conversion longer than Docling's 600-second synchronous
    wait testable: the polling loop advances simulated time instead of real time.
    """

    def __init__(self, step: float = 30.0) -> None:
        self.now = 1_000.0
        self.step = step
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += max(seconds, self.step)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr("app.parsing.docling._monotonic", fake.monotonic)
    monkeypatch.setattr("app.parsing.docling._sleep", fake.sleep)
    return fake


def client_with(
    handler,
    *,
    max_response_bytes: int = 1_000_000,
    deadline_seconds: float = 3_600,
    poll_interval_seconds: float = 5,
) -> DoclingClient:
    transport = httpx.MockTransport(handler)
    return DoclingClient(
        BASE_URL,
        httpx.AsyncClient(transport=transport),
        max_response_bytes=max_response_bytes,
        request_timeout_seconds=120,
        deadline_seconds=deadline_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )


async def convert(client: DoclingClient):
    return await client.convert(
        filename="report.pdf", media_type="application/pdf", content=b"%PDF-1.7"
    )


def result_body(chunks: list, *, status: str = "success") -> dict:
    return {
        "chunks": chunks,
        "documents": [
            {"kind": "ExportResult", "status": status, "content": {"filename": "report.pdf"}}
        ],
        "processing_time": 0.1,
    }


CHUNKS = [{"text": "Body text.", "page_numbers": [2], "headings": ["1 Service"]}]


def scripted(
    *,
    statuses: list[str],
    result: object | None = None,
    submit: object | None = None,
    submit_status: int = 200,
    status_code: int = 200,
    result_status: int = 200,
    calls: list[str] | None = None,
):
    """A transport that answers submit, then *statuses* in order, then the result."""
    remaining = list(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(f"{request.method} {request.url.path}")
        if request.url.path == CHUNK_ASYNC_PATH:
            body = {"task_id": TASK_ID, "task_status": "pending"} if submit is None else submit
            return httpx.Response(submit_status, json=body)
        if request.url.path.startswith(STATUS_PATH):
            state = remaining.pop(0) if remaining else "success"
            return httpx.Response(status_code, json={"task_id": TASK_ID, "task_status": state})
        assert request.url.path.startswith(RESULT_PATH), request.url.path
        return httpx.Response(
            result_status, json=result_body(CHUNKS) if result is None else result
        )

    return handler


async def test_the_docling_client_is_a_resumable_converter() -> None:
    assert isinstance(client_with(scripted(statuses=[])), ResumableDocumentConverter)


async def test_submission_posts_the_async_chunk_route_and_returns_a_task_id(
    clock: FakeClock,
) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["body"] = request.content
        return httpx.Response(200, json={"task_id": TASK_ID, "task_status": "pending"})

    task_id = await client_with(handler).submit(
        filename="report.pdf", media_type="application/pdf", content=b"%PDF-1.7"
    )

    assert task_id == TASK_ID
    assert seen["method"] == "POST"
    assert seen["url"] == f"{BASE_URL}{CHUNK_ASYNC_PATH}"
    body = seen["body"]
    assert isinstance(body, bytes)
    assert b'name="files"; filename="report.pdf"' in body
    assert b"%PDF-1.7" in body
    assert b'name="include_converted_doc"' in body
    assert b'name="chunking_use_markdown_tables"' in body
    # The hierarchical route is used precisely because it needs no tokenizer, so
    # no token budget is sent: sizing chunks is this service's own job.
    assert b"chunking_max_tokens" not in body
    assert b"/v1/chunk/hierarchical/file/async" in seen["url"].encode()


async def test_the_converters_table_structure_is_carried_into_segments(
    clock: FakeClock,
) -> None:
    handler = scripted(
        statuses=["success"],
        result=result_body(
            [
                {
                    "text": "Diagnostic codes are listed below.",
                    "page_numbers": [111],
                    "headings": ["4.2 Diagnostics"],
                    "doc_items": ["#/texts/12"],
                },
                {
                    "text": "| Diagnostic | Action |\n| --- | --- |\n| D-142 | Replace sensor S-3 |",
                    "page_numbers": [112],
                    "headings": ["4.2 Diagnostics"],
                    "captions": ["Table 4-2 Diagnostic codes"],
                    "doc_items": ["#/tables/0"],
                },
            ]
        ),
    )

    converted = await convert(client_with(handler))

    assert [(segment.is_table, segment.caption) for segment in converted.segments] == [
        (False, ""),
        (True, "Table 4-2 Diagnostic codes"),
    ]
    assert converted.segments[1].table_ref == "#/tables/0"


@pytest.mark.parametrize(
    "doc_items",
    [None, [], "#/tables/0", ["#/texts/1"], [None], [{"self_ref": "#/tables/0"}]],
)
async def test_only_a_table_reference_marks_a_segment_as_a_table(
    clock: FakeClock, doc_items: object
) -> None:
    handler = scripted(
        statuses=["success"],
        result=result_body([{"text": "Body text.", "doc_items": doc_items}]),
    )

    converted = await convert(client_with(handler))

    assert converted.segments[0].is_table is False


async def test_an_unusable_caption_is_dropped(clock: FakeClock) -> None:
    handler = scripted(
        statuses=["success"],
        result=result_body(
            [{"text": "Body.", "doc_items": ["#/tables/0"], "captions": [None, "  ", "Real"]}]
        ),
    )

    converted = await convert(client_with(handler))

    assert converted.segments[0].caption == "Real"


async def test_a_conversion_is_polled_to_completion_and_keeps_page_provenance(
    clock: FakeClock,
) -> None:
    calls: list[str] = []
    handler = scripted(
        statuses=["pending", "started", "success"],
        result=result_body(
            [
                {
                    "text": "Chapter 1\nInspect the controller module every three years.",
                    "page_numbers": [4, 5],
                    "headings": ["Service manual", "1 Preventive maintenance"],
                },
                {
                    "text": "Controller diagnostic code 1.",
                    "page_numbers": [17],
                    "headings": ["2 Alarms"],
                },
            ]
        ),
        calls=calls,
    )

    converted = await convert(client_with(handler))

    # The page a chunk starts on, and the most specific heading above it.
    assert [(segment.page, segment.section) for segment in converted.segments] == [
        (4, "1 Preventive maintenance"),
        (17, "2 Alarms"),
    ]
    # Chunk page ranges do not prove a whole-source page count: trailing blank or
    # figure-only pages may have no chunk.
    assert converted.page_count is None
    assert converted.has_provenance is True
    assert "Inspect the controller module" in converted.text
    assert calls == [
        f"POST {CHUNK_ASYNC_PATH}",
        f"GET {STATUS_PATH}/{TASK_ID}",
        f"GET {STATUS_PATH}/{TASK_ID}",
        f"GET {STATUS_PATH}/{TASK_ID}",
        f"GET {RESULT_PATH}/{TASK_ID}",
    ]


async def test_a_conversion_longer_than_the_sync_wait_still_completes(
    clock: FakeClock,
) -> None:
    """Docling answers its synchronous routes with 504 after 600 seconds.

    Twenty-five polls thirty simulated seconds apart put completion well past
    that, which the task API has no opinion about.
    """
    calls: list[str] = []
    handler = scripted(statuses=["started"] * 25 + ["success"], calls=calls)

    converted = await convert(client_with(handler))

    assert converted.text == "Body text."
    assert clock.now - 1_000.0 > 600
    assert calls.count(f"POST {CHUNK_ASYNC_PATH}") == 1


async def test_a_deadline_shorter_than_the_conversion_is_reported_as_such(
    clock: FakeClock,
) -> None:
    handler = scripted(statuses=["started"] * 100)

    with pytest.raises(ConversionDeadlineExceededError, match="conversion deadline"):
        await convert(client_with(handler, deadline_seconds=120))


async def test_a_resumed_task_keeps_the_deadline_it_was_submitted_under(
    clock: FakeClock,
) -> None:
    """The deadline runs from submission, so resuming grants no fresh budget."""
    client = client_with(scripted(statuses=["started"] * 100), deadline_seconds=600)
    long_ago = datetime.now(UTC) - timedelta(seconds=900)

    with pytest.raises(ConversionDeadlineExceededError):
        await client.await_result(TASK_ID, submitted_at=long_ago)

    # A task submitted moments ago still has almost all of its budget.
    fresh = client_with(scripted(statuses=["started", "success"]), deadline_seconds=600)
    converted = await fresh.await_result(TASK_ID, submitted_at=datetime.now(UTC))
    assert converted.text == "Body text."


async def test_polls_are_spaced_by_the_configured_interval(clock: FakeClock) -> None:
    await convert(
        client_with(scripted(statuses=["started", "started", "success"]), poll_interval_seconds=7)
    )

    assert clock.slept == [7, 7]


@pytest.mark.parametrize("state", ["failure", "skipped", "exploded"])
async def test_a_failed_or_unknown_task_status_fails_the_job(
    clock: FakeClock, state: str
) -> None:
    with pytest.raises(ConversionFailedError, match="could not read this file"):
        await convert(client_with(scripted(statuses=["started", state])))


async def test_a_submission_that_comes_back_failed_is_not_polled(clock: FakeClock) -> None:
    calls: list[str] = []
    handler = scripted(
        statuses=[], submit={"task_id": TASK_ID, "task_status": "failure"}, calls=calls
    )

    with pytest.raises(ConversionFailedError):
        await convert(client_with(handler))
    assert calls == [f"POST {CHUNK_ASYNC_PATH}"]


@pytest.mark.parametrize("submit", [{"task_status": "pending"}, {"task_id": "  "}, {"task_id": 7}])
async def test_a_submission_without_a_task_id_reports_unavailability(
    clock: FakeClock, submit: dict
) -> None:
    with pytest.raises(ConversionUnavailableError):
        await convert(client_with(scripted(statuses=[], submit=submit)))


@pytest.mark.parametrize("status_code", [400, 415, 422])
async def test_a_rejected_submission_is_reported_as_a_submission_failure(
    clock: FakeClock, status_code: int
) -> None:
    handler = scripted(statuses=[], submit={"detail": "no"}, submit_status=status_code)

    with pytest.raises(ConversionSubmissionError, match="refused this file"):
        await convert(client_with(handler))


@pytest.mark.parametrize("status_code", [404, 405, 501])
async def test_a_missing_async_route_is_reported_rather_than_downgraded(
    clock: FakeClock, status_code: int
) -> None:
    """No Markdown fallback: losing page provenance silently is worse than failing."""
    handler = scripted(statuses=[], submit={"detail": "Not Found"}, submit_status=status_code)

    with pytest.raises(ConversionUnavailableError, match="asynchronous chunking API"):
        await convert(client_with(handler))


async def test_a_submission_outage_reports_unavailability(clock: FakeClock) -> None:
    handler = scripted(statuses=[], submit={"detail": "boom"}, submit_status=503)

    with pytest.raises(ConversionUnavailableError, match="unavailable"):
        await convert(client_with(handler))


async def test_an_unreachable_converter_reports_unavailability(clock: FakeClock) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ConversionUnavailableError, match="unavailable"):
        await convert(client_with(handler))


async def test_a_request_timeout_reports_unavailability(clock: FakeClock) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ConversionUnavailableError):
        await convert(client_with(handler))


async def test_a_forgotten_task_is_reported_as_lost(clock: FakeClock) -> None:
    handler = scripted(statuses=["started"], status_code=404)

    with pytest.raises(ConversionTaskLostError, match="Upload the file again"):
        await convert(client_with(handler))


async def test_a_sustained_polling_outage_reports_unavailability(clock: FakeClock) -> None:
    handler = scripted(statuses=["started"], status_code=503)

    with pytest.raises(ConversionUnavailableError):
        await convert(client_with(handler, deadline_seconds=3_600))
    # Given up on well inside the hour-long deadline rather than polling it out.
    assert clock.now - 1_000.0 < 600


@pytest.mark.parametrize("status_code", [429, 500, 503])
async def test_a_polling_outage_shorter_than_the_grace_window_does_not_lose_the_conversion(
    clock: FakeClock, status_code: int
) -> None:
    """A conversion can run for an hour; a brief converter blip must not discard it.

    Six failing polls thirty simulated seconds apart is a three-minute outage --
    long enough that a small consecutive-failure count would have given up.
    """
    answers = [(status_code, "started")] * 6 + [(200, "success")]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == CHUNK_ASYNC_PATH:
            return httpx.Response(200, json={"task_id": TASK_ID, "task_status": "pending"})
        if request.url.path.startswith(STATUS_PATH):
            code, state = answers.pop(0) if answers else (200, "success")
            return httpx.Response(code, json={"task_id": TASK_ID, "task_status": state})
        return httpx.Response(200, json=result_body(CHUNKS))

    assert (await convert(client_with(handler))).text == "Body text."


async def test_intermittent_blips_spread_out_do_not_accumulate_into_a_failure(
    clock: FakeClock,
) -> None:
    """The grace window measures one continuous outage, not a tally over the hour."""
    answers = [(503, "started"), (200, "started")] * 30 + [(200, "success")]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == CHUNK_ASYNC_PATH:
            return httpx.Response(200, json={"task_id": TASK_ID, "task_status": "pending"})
        if request.url.path.startswith(STATUS_PATH):
            code, state = answers.pop(0) if answers else (200, "success")
            return httpx.Response(code, json={"task_id": TASK_ID, "task_status": state})
        return httpx.Response(200, json=result_body(CHUNKS))

    assert (await convert(client_with(handler))).text == "Body text."


async def test_a_transient_result_failure_is_retried_rather_than_discarded(
    clock: FakeClock,
) -> None:
    """The conversion already succeeded; one bad fetch must not throw it away."""
    result_calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == CHUNK_ASYNC_PATH:
            return httpx.Response(200, json={"task_id": TASK_ID, "task_status": "pending"})
        if request.url.path.startswith(STATUS_PATH):
            return httpx.Response(200, json={"task_id": TASK_ID, "task_status": "success"})
        result_calls.append(1)
        if len(result_calls) < 3:
            return httpx.Response(503, json={"detail": "overloaded"})
        return httpx.Response(200, json=result_body(CHUNKS))

    assert (await convert(client_with(handler))).text == "Body text."
    assert len(result_calls) == 3


async def test_a_result_the_converter_has_discarded_is_not_retried(clock: FakeClock) -> None:
    """404 is a lost task, not a transient one: retrying it would only wait."""
    result_calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == CHUNK_ASYNC_PATH:
            return httpx.Response(200, json={"task_id": TASK_ID, "task_status": "pending"})
        if request.url.path.startswith(STATUS_PATH):
            return httpx.Response(200, json={"task_id": TASK_ID, "task_status": "success"})
        result_calls.append(1)
        return httpx.Response(404, json={"detail": "Task not found."})

    with pytest.raises(ConversionTaskLostError):
        await convert(client_with(handler))
    assert len(result_calls) == 1


@pytest.mark.parametrize("status_code", [401, 403, 429])
async def test_a_throttled_or_rejected_submission_reports_unavailability(
    clock: FakeClock, status_code: int
) -> None:
    """These say "not now" about the converter, not "never" about the document."""
    handler = scripted(statuses=[], submit={"detail": "no"}, submit_status=status_code)

    with pytest.raises(ConversionUnavailableError, match="unavailable"):
        await convert(client_with(handler))


async def test_a_missing_result_is_reported_as_lost(clock: FakeClock) -> None:
    handler = scripted(statuses=["success"], result={"detail": "gone"}, result_status=404)

    with pytest.raises(ConversionTaskLostError):
        await convert(client_with(handler))


async def test_an_unreadable_result_is_reported_as_a_retrieval_failure(
    clock: FakeClock,
) -> None:
    handler = scripted(statuses=["success"], result={"detail": "no"}, result_status=409)

    with pytest.raises(ConversionResultUnavailableError, match="could not be retrieved"):
        await convert(client_with(handler))


async def test_a_task_failure_result_fails_the_job(clock: FakeClock) -> None:
    handler = scripted(
        statuses=["success"],
        result={"kind": "TaskFailureResult", "failure": {"phase": "execution"}},
    )

    with pytest.raises(ConversionFailedError):
        await convert(client_with(handler))


async def test_partial_success_is_rejected(clock: FakeClock) -> None:
    handler = scripted(
        statuses=["partial_success"], result=result_body(CHUNKS, status="partial_success")
    )

    with pytest.raises(ConversionFailedError):
        await convert(client_with(handler))


async def test_structural_paths_ranges_and_counts_are_preserved(clock: FakeClock) -> None:
    handler = scripted(
        statuses=["success"],
        result=result_body(
            [
                {
                    "text": "Maintenance overview.",
                    "page_numbers": [2, 3],
                    "headings": ["Manual", "Maintenance"],
                    "doc_items": ["#/texts/1", "#/pictures/0"],
                    "section_ref": "#/groups/maintenance",
                },
                {
                    "text": "| Code | Action |\n| --- | --- |\n| 310 | Recalibrate |",
                    "page_numbers": [4],
                    "headings": ["Manual", "Maintenance"],
                    "doc_items": ["#/tables/0"],
                },
                {
                    "text": "Alarm overview.",
                    "page_numbers": [5],
                    "headings": ["Manual", "Alarms"],
                    "doc_items": ["#/texts/2"],
                },
            ]
        ),
    )

    converted = await convert(client_with(handler))

    assert converted.page_count is None
    assert converted.recognized_section_count is None
    assert converted.recognized_table_count == 1
    assert converted.recognized_figure_count is None
    assert converted.segments[0].page == 2
    assert converted.segments[0].page_end == 3
    assert converted.segments[0].section_path == ("Manual", "Maintenance")
    assert converted.segments[0].section_ref == "#/groups/maintenance"


async def test_supported_empty_docling_structure_and_unavailable_figures(
    clock: FakeClock,
) -> None:
    handler = scripted(
        statuses=["success"],
        result=result_body(
            [{"text": "Body.", "page_numbers": [1], "headings": [], "doc_items": []}]
        ),
    )

    converted = await convert(client_with(handler))

    assert converted.recognized_section_count is None
    assert converted.recognized_table_count == 0
    assert converted.recognized_figure_count is None


async def test_repeated_heading_text_does_not_claim_an_exact_section_count(
    clock: FakeClock,
) -> None:
    handler = scripted(
        statuses=["success"],
        result=result_body(
            [
                {
                    "text": "Maintenance A.",
                    "headings": ["Maintenance"],
                    "section_ref": "#/groups/a",
                    "doc_items": ["#/texts/0"],
                },
                {
                    "text": "Maintenance B.",
                    "headings": ["Maintenance"],
                    "section_ref": "#/groups/b",
                    "doc_items": ["#/texts/1"],
                },
            ]
        ),
    )

    converted = await convert(client_with(handler))

    assert [segment.section_ref for segment in converted.segments] == [
        "#/groups/a",
        "#/groups/b",
    ]
    assert converted.recognized_section_count is None


async def test_a_failed_document_status_in_the_result_fails_the_job(clock: FakeClock) -> None:
    handler = scripted(statuses=["success"], result=result_body(CHUNKS, status="failure"))

    with pytest.raises(ConversionFailedError):
        await convert(client_with(handler))


async def test_chunks_without_pages_report_no_pagination(clock: FakeClock) -> None:
    handler = scripted(
        statuses=["success"],
        result=result_body([{"text": "Body text.", "page_numbers": None, "headings": None}]),
    )

    converted = await convert(client_with(handler))

    assert converted.page_count is None
    assert converted.segments[0].page is None
    assert converted.segments[0].section == ""
    assert converted.has_provenance is False


@pytest.mark.parametrize("pages", [[], [0], ["4"], [True], "4", {"page": 4}])
async def test_unusable_page_numbers_are_not_turned_into_pages(
    clock: FakeClock, pages: object
) -> None:
    handler = scripted(
        statuses=["success"], result=result_body([{"text": "Body.", "page_numbers": pages}])
    )

    converted = await convert(client_with(handler))

    assert converted.segments[0].page is None
    assert converted.page_count is None


async def test_empty_and_malformed_chunks_are_skipped(clock: FakeClock) -> None:
    handler = scripted(
        statuses=["success"],
        result=result_body(
            [
                "not an object",
                {"text": "   "},
                {"text": None},
                {"text": "Real content.", "page_numbers": [2]},
            ]
        ),
    )

    converted = await convert(client_with(handler))

    assert [segment.text for segment in converted.segments] == ["Real content."]


async def test_a_result_with_no_usable_text_fails_the_job(clock: FakeClock) -> None:
    handler = scripted(statuses=["success"], result=result_body([{"text": "  "}]))

    with pytest.raises(ConversionFailedError, match="no readable text"):
        await convert(client_with(handler))


@pytest.mark.parametrize("payload", [{"chunks": "not a list"}, {}, []])
async def test_unusable_results_fail_the_job(clock: FakeClock, payload: object) -> None:
    with pytest.raises(ConversionFailedError):
        await convert(client_with(scripted(statuses=["success"], result=payload)))


async def test_non_json_responses_fail_the_job(clock: FakeClock) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == CHUNK_ASYNC_PATH:
            return httpx.Response(200, json={"task_id": TASK_ID, "task_status": "pending"})
        if request.url.path.startswith(STATUS_PATH):
            return httpx.Response(200, json={"task_status": "success"})
        return httpx.Response(200, content=b"<html>gateway</html>")

    with pytest.raises(ConversionFailedError):
        await convert(client_with(handler))


async def test_oversized_results_are_cut_off_before_parsing(clock: FakeClock) -> None:
    handler = scripted(statuses=["success"], result=result_body([{"text": "x" * 5_000}]))

    with pytest.raises(DocumentTooLargeError):
        await convert(client_with(handler, max_response_bytes=64))


async def test_an_oversized_status_response_is_cut_off_before_parsing(
    clock: FakeClock,
) -> None:
    """A status poll is a short object; a document-sized one is refused outright."""
    padding = json.dumps({"task_status": "started", "junk": "y" * 200_000})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == CHUNK_ASYNC_PATH:
            return httpx.Response(200, json={"task_id": TASK_ID, "task_status": "pending"})
        return httpx.Response(200, content=padding.encode())

    with pytest.raises(DocumentTooLargeError):
        await convert(client_with(handler))


async def test_no_message_leaks_the_converter_address(clock: FakeClock) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ConversionUnavailableError) as failure:
        await convert(client_with(handler))
    assert "docling:5001" not in str(failure.value)
