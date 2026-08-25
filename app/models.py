"""Versioned request/response contract for Seshat."""

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.config import MAXIMUM_SEARCH_COLLECTIONS

JobStatus = Literal["accepted", "processing", "completed", "failed"]
# What a source looks like to a UI: an ingest job's four states collapsed into
# the three a person can act on.
SourceStatus = Literal["processing", "ready", "failed"]
SEARCH_CONTEXT_METADATA_KEY = "search_context"
COLLECTION_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,62}$"
SECTION_REFERENCE_PATTERN = r"^sec_[A-Za-z0-9_-]{24}$"
MAX_EXTERNAL_ID_CHARACTERS = 256

CollectionId = Annotated[
    str,
    StringConstraints(pattern=COLLECTION_ID_PATTERN),
]
ExternalId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=MAX_EXTERNAL_ID_CHARACTERS,
        strip_whitespace=True,
    ),
]
SectionReference = Annotated[
    str, StringConstraints(pattern=SECTION_REFERENCE_PATTERN)
]

# Metadata keys the service owns. User-supplied metadata may not overwrite them,
# because search filtering and collection isolation read these keys back.
RESERVED_METADATA_KEYS = frozenset(
    {
        "collection_id",
        "document_id",
        "revision_id",
        "projection_state",
        "section_id",
        "section_ref",
        "external_id",
        "title",
        "source_type",
        "source_uri",
        "checksum",
        "version",
        "page",
        "page_end",
        "section",
        "section_path",
        "source_section_ref",
        "block_ordinals",
        "updated_at",
        "updated_at_ts",
        "embedding_model",
        "embedding_dim",
        "filename",
        "media_type",
        "is_table",
        "table_part",
        "table_parts",
        "caption",
        "table_ref",
        "table_header",
        "_node_content",
        SEARCH_CONTEXT_METADATA_KEY,
    }
)


class TextDocumentRequest(BaseModel):
    """Normalized text document submitted by a caller or a future parser worker."""

    model_config = ConfigDict(extra="forbid")

    collection_id: CollectionId
    external_id: ExternalId
    text: str = Field(min_length=1)
    title: str = Field(default="", max_length=512)
    source_type: str = Field(default="text", max_length=64)
    source_uri: str = Field(default="", max_length=2048)
    version: str = Field(default="", max_length=128)
    page: int | None = Field(default=None, ge=0)
    section: str = Field(default="", max_length=512)
    metadata: dict[str, Any] = Field(default_factory=dict)
    force_reindex: bool = False


class IngestAcceptedResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    collection_id: str
    external_id: str


class JobResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    collection_id: str
    external_id: str
    chunk_count: int = 0
    unchanged: bool = False
    detail: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SourceItem(BaseModel):
    """One selectable document source."""

    external_id: str
    title: str
    source_type: str
    status: SourceStatus
    chunk_count: int = Field(ge=0)
    detail: str | None = None
    filename: str | None = None
    media_type: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    # Whether this source has an uploaded original that can be opened.
    viewable: bool = False
    byte_size: int | None = None
    page_count: int | None = None
    preview_available: bool = False


class SourceListResponse(BaseModel):
    collection_id: str
    items: list[SourceItem]
    truncated: bool = False


class SourceContentResponse(BaseModel):
    """What a viewer needs before it decides how to render a source."""

    collection_id: str
    external_id: str
    title: str
    source_type: str
    status: SourceStatus
    filename: str
    media_type: str
    byte_size: int
    checksum: str
    page_count: int | None = None
    preview_available: bool = False
    preview_bytes: int | None = None
    chunk_count: int = Field(default=0, ge=0)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SourceOutlineSection(BaseModel):
    """One converter-recognized section in source order."""

    section_ref: SectionReference
    section_path: list[str]
    page_start: int | None = Field(default=None, ge=0)
    page_end: int | None = Field(default=None, ge=0)


class SourceOutlineResponse(BaseModel):
    """Persisted structural facts for one current source."""

    collection_id: str
    external_id: str
    page_count: int | None = Field(ge=0)
    recognized_section_count: int | None = Field(ge=0)
    recognized_table_count: int | None = Field(ge=0)
    recognized_figure_count: int | None = Field(ge=0)
    sections: list[SourceOutlineSection]
    reason: str | None = None


class ScanRequest(BaseModel):
    """Ordered traversal of one current source or recognized section."""

    model_config = ConfigDict(extra="forbid")

    collection_id: CollectionId
    external_id: ExternalId
    section_ref: SectionReference | None = None
    limit: int | None = Field(default=None, ge=1)
    cursor: str | None = Field(default=None, min_length=1, max_length=4096)


class ScanItem(BaseModel):
    text: str
    section_ref: SectionReference | None = None
    section_path: list[str] | None = None
    page_start: int | None = Field(default=None, ge=0)
    page_end: int | None = Field(default=None, ge=0)


class ScanResponse(BaseModel):
    collection_id: str
    external_id: str
    items: list[ScanItem]
    next_cursor: str | None


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: list[str] | None = None
    external_id: list[str] | None = None
    exclude_external_id: list[str] | None = None
    updated_after: datetime | None = None


class SearchRequest(BaseModel):
    """Search is always scoped to an explicit, non-empty set of collections."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2048)
    collection_ids: list[CollectionId] = Field(
        min_length=1, max_length=MAXIMUM_SEARCH_COLLECTIONS
    )
    top_k: int | None = Field(default=None, ge=1, le=200)
    filters: SearchFilters | None = None


class Citation(BaseModel):
    label: str
    source_uri: str | None = None
    locator: str | None = None
    page: int | None = None
    page_end: int | None = None
    section: str | None = None
    section_ref: SectionReference | None = None
    section_path: list[str] | None = None


class SearchResultItem(BaseModel):
    text: str
    score: float
    retrieval_score: float | None = None
    collection_id: str
    external_id: str | None = None
    title: str | None = None
    source_type: str | None = None
    source_uri: str | None = None
    version: str | None = None
    checksum: str | None = None
    page: int | None = None
    page_end: int | None = None
    section: str | None = None
    section_ref: SectionReference | None = None
    section_path: list[str] | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)


class SearchStats(BaseModel):
    retrieved: int = Field(ge=0)
    returned: int = Field(ge=0)


class SearchResponse(BaseModel):
    items: list[SearchResultItem]
    warnings: list[str] = Field(default_factory=list)
    stats: SearchStats


class HealthResponse(BaseModel):
    status: str
    database: str | None = None
    detail: str | None = None
