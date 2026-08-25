"""Shared representative source setup for knowledge-workflow tests."""

import json
from pathlib import Path
from typing import Any

from httpx import AsyncClient

from app.parsing import DocumentSegment, build_converted_document
from tests.fakes import RecordingConverter

WORKFLOW_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "knowledge_workflow_manual.json"
)
SEARCH_VOCAB = (
    "101",
    "102",
    "190",
    "communication",
    "controller",
    "maintenance",
    "power",
    "temperature",
    "voltage",
)


def load_workflow_fixture() -> dict[str, Any]:
    return json.loads(WORKFLOW_FIXTURE_PATH.read_text())


def configure_workflow_converter(
    converter: RecordingConverter, fixture: dict[str, Any]
) -> None:
    converter.converted_document = build_converted_document(
        [
            DocumentSegment(
                text=segment["text"],
                page=segment.get("page"),
                page_end=segment.get("page_end"),
                section_path=tuple(segment.get("section_path", ())),
                section_ref=segment.get("section_ref"),
                is_table=segment.get("is_table", False),
                table_ref=segment.get("table_ref"),
                caption=segment.get("caption", ""),
            )
            for segment in fixture["segments"]
        ],
        page_count=fixture["page_count"],
        recognized_section_count=fixture["recognized_section_count"],
        recognized_table_count=fixture["recognized_table_count"],
        recognized_figure_count=fixture["recognized_figure_count"],
    )


async def index_workflow_fixture(
    client: AsyncClient,
    converter: RecordingConverter,
    fixture: dict[str, Any],
    *,
    headers: dict[str, str],
) -> dict[str, Any]:
    configure_workflow_converter(converter, fixture)
    accepted = await client.post(
        "/v1/documents/file",
        files={
            "file": (
                "equipment-handbook.pdf",
                b"%PDF-knowledge-workflow",
                "application/pdf",
            )
        },
        data={
            "collection_id": fixture["collection_id"],
            "external_id": fixture["external_id"],
            "title": fixture["title"],
        },
        headers=headers,
    )
    accepted.raise_for_status()
    ingestion = client.app.state.runtime.ingestion  # type: ignore[attr-defined]
    await ingestion.wait_for_pending()
    job = await client.get(
        f"/v1/jobs/{accepted.json()['job_id']}", headers=headers
    )
    job.raise_for_status()
    body = job.json()
    assert body.get("status") == "completed", body
    return body
