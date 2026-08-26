# API Reference

Seshat exposes an HTTP API at `/v1`. Every `/v1/*` request requires:

```http
Authorization: Bearer <SESHAT_API_TOKEN>
```

The running service also exposes an interactive OpenAPI reference at `/docs`.

## Health

| Method | Path            | Purpose                                                   |
| ------ | --------------- | --------------------------------------------------------- |
| `GET`  | `/health/live`  | Process liveness. Does not require a database connection. |
| `GET`  | `/health/ready` | Database readiness. Returns `503` until Seshat is ready.  |

## Ingestion

| Method | Path                 | Purpose                              |
| ------ | -------------------- | ------------------------------------ |
| `POST` | `/v1/documents/text` | Create or update a text source.      |
| `POST` | `/v1/documents/file` | Upload and index one supported file. |
| `GET`  | `/v1/jobs/{job_id}`  | Check an asynchronous ingestion job. |

`POST /v1/documents/text` accepts JSON with `collection_id`, `external_id`, and
`text`. Optional fields include `title`, `source_type`, `source_uri`, `version`,
`page`, `section`, `metadata`, and `force_reindex`.

`POST /v1/documents/file` accepts `multipart/form-data` with `file`,
`collection_id`, caller-owned `external_id`, and an optional `title`. Supported
extensions are `.txt`, `.md`, `.markdown`, `.pdf`, `.docx`, `.pptx`, `.xlsx`,
and `.html`.

Both ingestion endpoints return `202 Accepted` with a `job_id`. Poll the job
endpoint until `status` is `completed` or `failed`. The pair `(collection_id,
external_id)` is the stable source identity.

## Sources

| Method | Path                                                         | Purpose                                                                     |
| ------ | ------------------------------------------------------------ | --------------------------------------------------------------------------- |
| `GET`  | `/v1/documents?collection_id=…`                              | List up to 200 sources in a collection.                                     |
| `GET`  | `/v1/documents/source?collection_id=…&external_id=…`         | Get metadata for a retained upload.                                         |
| `GET`  | `/v1/documents/source/outline?collection_id=…&external_id=…` | Get fixed structural counts and recognized sections for one current source. |
| `GET`  | `/v1/documents/source/content?collection_id=…&external_id=…` | Download or stream a retained upload.                                       |

Source content supports `Range` requests and `ETag` validation. Set
`variant=preview` to retrieve extracted text when it is available; the default
`variant=original` retrieves the uploaded file.

The outline returns structural counts from the current normalized
representation. `null` means the converter could not establish a count; zero
means it established that none were present. Sections are in source order with
an opaque `section_ref`, full `section_path`, and page bounds when known.
References change when source content is replaced.

## Retrieval

| Method | Path         | Purpose                                                  |
| ------ | ------------ | -------------------------------------------------------- |
| `POST` | `/v1/search` | Retrieve ranked passages from explicit collections.      |
| `POST` | `/v1/scan`   | Traverse ordered source passages in one source or section. |

Search requires `query` and `collection_ids`. Optional `top_k` limits results;
`filters` supports `source_type`, `external_id`, `exclude_external_id`, and
`updated_after`.

Each result includes passage text, score, caller-owned collection and source
identity, and available page and section location. Use `section_ref` with the
result's collection and external IDs to scan that section. Empty optional fields
are omitted, and responses never expose internal identifiers or repeat document
management metadata. `page_end` appears only when a passage spans pages.

Use search when ranked relevance is sufficient. Use scan when the caller must
inspect all content in a source or an outline-selected section; scan has no query
or relevance filtering.

Search and scan passages pack adjacent source content around the configured token
target. Paragraphs, headings, tables, and section transitions remain in source
order. A table that fits stays with its surrounding text; a larger table is split
only between rows and repeats its header. Search may overlap split passages, while
scan passages never overlap.

The first scan request supplies `collection_id`, `external_id`, an optional
outline `section_ref`, and an optional `limit`. Continue with `next_cursor` and
the same scope until it is `null`; only then is the selected scope complete.

Scan items contain complete passage text with available section and page
provenance. A passage spanning sections uses page provenance without claiming a
single section. Pages obey source-block and payload limits; content is deferred,
never truncated. Internal identifiers are not returned.

Knowledge errors use a machine-readable `detail.code`:

| Status | Code                | Meaning                                                                    |
| ------ | ------------------- | -------------------------------------------------------------------------- |
| `400`  | `invalid_cursor`    | The cursor is malformed or does not match the requested source or section. |
| `404`  | `source_not_found`  | The requested current source does not exist.                               |
| `404`  | `section_not_found` | The section reference is not current for the requested source.             |
| `409`  | `source_changed`    | The source representation changed after the scan began.                    |
| `422`  | `invalid_limit`     | `limit` exceeds the configured deployment maximum.                         |
