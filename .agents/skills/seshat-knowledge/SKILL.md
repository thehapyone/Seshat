---
name: seshat-knowledge
description: Run or integrate Seshat as a document-processing and retrieval service. Use when an agent needs to ingest text or files, obtain extracted content, or retrieve evidence with search, outline, and scan.
---

# Seshat Knowledge

Seshat processes documents and returns grounded evidence. The calling agent
owns source synchronization, authorization, retrieval strategy, working state,
extraction, comparison, counting, and final answers.

Seshat can:

- ingest and retain text, PDF, Office, HTML, and Markdown sources;
- return ranked passages with source and location provenance; and
- expose recognized structure or scan source content deterministically.

It does not crawl remote systems, reason over evidence, or produce answers.

## Connect

Start the repository's Compose stack when needed:

```bash
cp seshat.example.toml seshat.toml
cp .env.example .env
docker compose up -d --build
curl -fsS http://127.0.0.1:8080/health/ready
```

Readiness returns `{"status":"ready","database":"ok"}`. Liveness at
`/health/live` returns `{"status":"ok"}` without checking the database.

Before starting it, configure these baseline values:

- In `.env`: `SESHAT_POSTGRES_PASSWORD`, `SESHAT_API_TOKEN`,
  `SESHAT_CURSOR_SIGNING_KEY`, and `SESHAT_EMBEDDING_API_KEY`.
- In `seshat.toml`: `embedding.base_url`, `embedding.model`, and
  `embedding.dimension` for an OpenAI-compatible embeddings endpoint.

The example configuration uses the Compose-managed Docling converter. For Azure
Document Intelligence instead, set `converter.backend = "azure"`, configure
`converter.azure.endpoint`, and set `AZURE_OCR_API_KEY` in `.env`.

For a running deployment, set an authorized service URL and token:

```bash
export SESHAT_URL="http://127.0.0.1:8080"
export SESHAT_API_TOKEN="<configured API token>"
```

Send the bearer token on every `/v1/*` request. Map the caller's authorized
knowledge scope to explicit `collection_id` or `collection_ids` values; never
let an agent broaden that scope itself.

## Ingest

`collection_id` is the isolation boundary. `external_id` is the caller-owned,
stable source identity within that collection. Reusing the pair updates the
source only after changed content has been processed successfully.

Send normalized text directly:

```bash
curl -sS -X POST "$SESHAT_URL/v1/documents/text" \
  -H "Authorization: Bearer $SESHAT_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "collection_id": "manuals",
    "external_id": "equipment-handbook",
    "title": "Equipment handbook",
    "text": "The source text to index goes here."
  }'
```

Upload `.txt`, `.md`, `.markdown`, `.pdf`, `.docx`, `.pptx`, `.xlsx`, or
`.html` files:

```bash
curl -sS -X POST "$SESHAT_URL/v1/documents/file" \
  -H "Authorization: Bearer $SESHAT_API_TOKEN" \
  -F collection_id=manuals \
  -F external_id=equipment-handbook \
  -F file=@./equipment-handbook.pdf
```

Both requests return `202 Accepted` with a `job_id`. Poll until `status` is
`completed`; do not retrieve before then. Report a failed job's `detail` rather
than treating it as absent evidence.

```json
{
  "job_id": "94c32d46-9f54-4df0-9e35-1683cfe3a4b8",
  "status": "accepted",
  "collection_id": "manuals",
  "external_id": "equipment-handbook"
}
```

```bash
curl -sS "$SESHAT_URL/v1/jobs/<job_id>" \
  -H "Authorization: Bearer $SESHAT_API_TOKEN"
```

```json
{
  "job_id": "94c32d46-9f54-4df0-9e35-1683cfe3a4b8",
  "status": "completed",
  "collection_id": "manuals",
  "external_id": "equipment-handbook",
  "chunk_count": 12,
  "unchanged": false,
  "detail": null
}
```

For an uploaded source, retrieve converter text with
`GET /v1/documents/source/content?variant=preview` when metadata reports
`preview_available: true`. `variant=original` retrieves the retained upload.
First read `GET /v1/documents/source` for the metadata:

```json
{
  "collection_id": "manuals",
  "external_id": "equipment-handbook",
  "title": "Equipment handbook",
  "status": "ready",
  "filename": "equipment-handbook.pdf",
  "media_type": "application/pdf",
  "byte_size": 18342,
  "page_count": 4,
  "preview_available": true,
  "preview_bytes": 9120,
  "chunk_count": 12
}
```

`/documents/source/content` streams bytes, not JSON. It returns `200` (or
`206` for a `Range` request), plus `Content-Type`, `Content-Length`, `ETag`, and
`Accept-Ranges` headers.

## Choose the retrieval capability

| Need                                 | Seshat capability                 | Caller responsibility                              |
| ------------------------------------ | --------------------------------- | -------------------------------------------------- |
| Focused fact or known target         | `search`                          | Judge and retain candidates                        |
| Recognized source structure          | `outline`                         | Treat `null` as unavailable, not zero              |
| Complete source or one section scope | `scan` to a null cursor           | Process every ordered item in that scope           |
| Exhaustive topic, list, or count      | `outline`, then relevant `scan`s  | Establish the topic span, extract, deduplicate, count |

Search is ranked and incomplete. Increasing `top_k` does not establish
coverage. Outline reports converter-recognized facts only. Use scan when the
task requires all content in source order.

## Search focused evidence

Search every collection the caller is authorized to inspect. Filter by
`external_id` when the question names a known source.

```bash
curl -sS -X POST "$SESHAT_URL/v1/search" \
  -H "Authorization: Bearer $SESHAT_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What is diagnostic code 101?",
    "collection_ids": ["manuals"],
    "filters": {"external_id": ["equipment-handbook"]},
    "top_k": 5
  }'
```

Retain source and location fields with each candidate passage. A result's
`section_ref` can scope a later scan; an absent result does not prove absent
source content.

```json
{
  "items": [{
    "text": "Diagnostic code 101 identifies the calibration procedure.",
    "score": 0.89,
    "collection_id": "manuals",
    "external_id": "equipment-handbook",
    "title": "Equipment handbook",
    "page": 2,
    "section_ref": "sec_Fl0pT0PxOYcdY1qRoKDNw5iv",
    "section_path": ["Diagnostics", "Codes"]
  }]
}
```

## Outline and scan for coverage

Request `/v1/documents/source/outline` for one source when recognized structure
can narrow the task. Use a returned `section_ref` unchanged. If structure is
unavailable or too coarse, scan the whole source.

```bash
curl -sS -G "$SESHAT_URL/v1/documents/source/outline" \
  -H "Authorization: Bearer $SESHAT_API_TOKEN" \
  --data-urlencode "collection_id=manuals" \
  --data-urlencode "external_id=equipment-handbook"
```

```json
{
  "collection_id": "manuals",
  "external_id": "equipment-handbook",
  "page_count": 4,
  "recognized_section_count": 2,
  "recognized_table_count": 1,
  "recognized_figure_count": null,
  "sections": [{
    "section_ref": "sec_Fl0pT0PxOYcdY1qRoKDNw5iv",
    "section_path": ["Diagnostics", "Codes"],
    "page_start": 2,
    "page_end": 2
  }]
}
```

Counts are `null` when the converter could not establish them; zero means it
established that none were present. An outline without sections includes a
`reason` field.

For an exhaustive topic, list, or exact count, first establish the topic's full
span from outline order, page ranges, headings, and repeated table headers. A
converter-recognized section can be narrower than the topic a person names. For
example, a table heading can be in one section while its rows continue through
adjacent sections named `Description`, `Additional information`, or
`Recommended action`. Treat such neighboring scopes as candidates until the
next clearly unrelated heading. If the outline cannot establish reliable
boundaries, scan the whole source instead.

```bash
curl -sS -X POST "$SESHAT_URL/v1/scan" \
  -H "Authorization: Bearer $SESHAT_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "collection_id": "manuals",
    "external_id": "equipment-handbook",
    "limit": 100
  }'
```

```json
{
  "collection_id": "manuals",
  "external_id": "equipment-handbook",
  "items": [{
    "text": "Diagnostic code 101 identifies the calibration procedure.",
    "section_ref": "sec_Fl0pT0PxOYcdY1qRoKDNw5iv",
    "section_path": ["Diagnostics", "Codes"],
    "page_start": 2,
    "page_end": 2
  }],
  "next_cursor": "eyJ2IjoxLCJjb2xsZWN0aW9uX2lkIjoibWFudWFscyJ9.signature"
}
```

Each item is a substantive passage assembled from adjacent paragraphs, headings,
and tables in source order. A passage can cross recognized section boundaries;
in that case it carries page bounds without a `section_ref`. Process each
response's ordered `items`; `limit` is the maximum number of items in that
response, though the payload bound can return fewer with a continuation cursor.
Retain only the caller state required for the task and repeat with `next_cursor`
in the same source and optional section scope. Traversal is complete only when
`next_cursor` is `null`. On `source_changed`, discard the partial traversal and
restart.

A null cursor completes only the requested source or `section_ref` scope. It
does not prove that the user's broader topic is complete. For exhaustive work:

1. Scan every outline scope identified as part of the topic, following each
   scope's cursor to `null`.
2. Extract values from the requested table column or record field rather than
   nearby identifiers, footnotes, or nested error numbers.
3. Preserve source order, deduplicate by the requested domain key, and inspect
   opening and closing fragments for rows split across passages or sections.
4. If the result is implausibly short, recheck the outline and adjacent scopes
   before claiming a count or complete list.

Knowledge errors use a machine-readable code:

```json
{
  "detail": {
    "code": "source_changed",
    "message": "The source changed after this scan began."
  }
}
```

The deployed instance's `/docs` is the authoritative API contract.
