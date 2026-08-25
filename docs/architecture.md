# Architecture

This document covers Seshat's deployment model, runtime boundaries, and
configuration. Start with the [README](../README.md) to run and use the service,
or see the [API Reference](api.md) for supported endpoints.

## Deployment

The default [`compose.yaml`](../compose.yaml) starts three containers:

| Container                | Responsibility                                                          |
| ------------------------ | ----------------------------------------------------------------------- |
| Seshat                   | Authenticated ingest, storage, indexing, search, outline, and scan API. |
| PostgreSQL with pgvector | Document metadata, jobs, and vector search.                             |
| Docling                  | PDF, Office, and HTML conversion.                                       |

Seshat keeps original uploads in a Docker volume and exposes only its API on
`127.0.0.1:8080`. PostgreSQL and Docling have no host ports. Docling is a
sizeable CPU image and may need a separate deployment when resources are tight.

Docling runs conversion, including OCR, on the deployment's CPU. Setting
`converter.backend = "azure"` delegates conversion to Azure AI Document
Intelligence instead; the Docling container is then unnecessary.

## Data Flow

1. A client submits text or a supported file with a stable `external_id` to a
   named collection.
2. Seshat writes uploaded bytes under a private job/revision key before
   background processing begins.
3. Text and Markdown are normalized in Seshat. Other supported files are sent
   to the configured converter, which returns text and available page, heading,
   and table context.
4. Seshat builds bounded, ordered, non-overlapping canonical blocks and a section
   hierarchy, then derives overlapping search chunks and embeddings from them.
5. After validating the staged representation, one PostgreSQL transaction swaps
   the current document metadata, source pointer, sections, blocks, and search
   projection. Failed replacements leave the prior revision untouched.
6. A client either searches explicit collection IDs for ranked passages or reads
   one current source through its structural outline and ordered canonical blocks.
   Search results carry the same public section references used by section scans,
   while internal document, section, revision, and search-chunk IDs stay private.

`collection_id` is the data-isolation boundary. Seshat applies the collection
filter during retrieval and checks every returned chunk before responding.

The calling agent owns the workflow around these read paths. Search ranks likely
evidence but does not prove coverage. Outline reports only structure established
by normalization. Scan provides deterministic source coverage, but it does not
extract records, compare them, deduplicate them, or summarize them. A caller that
needs an exhaustive result must retain its own state and continue scanning until
the response carries `next_cursor: null`.

## Asynchronous Conversion

Conversion is asynchronous. `POST /v1/documents/file` returns `202 Accepted`
with a Seshat `job_id`, which the client polls through `GET /v1/jobs/{job_id}`.
Seshat submits work to the configured converter and polls for the result; it
never waits for conversion inside the upload request.

The converter supplies document structure, including page, heading, and table
facts. Seshat uses those facts to build canonical blocks, then applies its own
central chunking policy for embeddings. It never infers structure from text.

## Sources and Connectors

Seshat accepts normalized text and file uploads; it does not fetch remote
systems. Source-specific connectors remain with the calling application, which
owns credentials, permissions, crawling, incremental sync, and deletion policy.

Files use the extension to select a format. Seshat directly handles `.txt`,
`.md`, and `.markdown`; the configured document converter handles `.pdf`,
`.docx`, `.pptx`, `.xlsx`, and `.html`. Docling does not OCR a scanned PDF that
has no extractable text; Azure Document Intelligence does, since OCR is part
of its layout analysis.

## Configuration

Seshat has exactly two configuration sources, and each setting belongs to one of
them:

- **A TOML file** holds every non-secret setting. `SESHAT_CONFIG_FILE` selects
  it, defaulting to `/etc/seshat/seshat.toml`; the Compose stack mounts
  `seshat.toml` there read-only. Start from
  [`seshat.example.toml`](../seshat.example.toml).
- **Environment variables** hold the five secrets, so no credential has to sit in
  a file that could be committed or baked into an image. The Compose stack reads
  them from `.env`.

Nothing else is consulted. Environment variables cannot override file settings,
unknown file settings fail startup, and validation errors never echo values.

Every setting, its default, and its purpose are listed in the
[Configuration Reference](configuration.md), which is generated from the single
declaration in [`app/config.py`](../app/config.py).

Uploads are bounded by `limits.max_upload_bytes` (50 MiB by default); extracted
text is separately bounded by `limits.max_document_bytes`. Provider-specific
limits should remain above the configured upload limit.

## Lifecycle and Limits

Text and file sources are identified by caller-owned `(collection_id,
external_id)`. Seshat computes the content checksum. Matching content updates
current display and citation metadata without re-embedding; changed content is
staged by job and atomically replaces the current canonical and search
representations only after validation. Replaced source objects and vector rows
are recorded in a durable cleanup queue during activation, removed afterward,
and retried at startup if a transient backend failure interrupted cleanup.

Ingestion and conversion run as in-process background work. Re-uploading an
active `(collection_id, external_id)` returns the existing job rather than
starting a second conversion. A persisted Docling task resumes after a Seshat
restart within its original deadline; other interrupted jobs fail and must be
submitted again. If Docling loses an in-flight task, the job fails with a
retryable message rather than hanging.

Seshat has no built-in deletion API, remote object-storage adapter, per-user
authorization, or remote-system connector.

At more than 2,000 embedding dimensions, pgvector cannot build Seshat's HNSW
index, so searches use an exact scan instead.
