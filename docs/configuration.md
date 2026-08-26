<!-- Generated from app/config.py by scripts/generate_config_reference.py. -->
<!-- Run that script after changing SETTINGS; CI checks this file is current. -->

# Configuration Reference

Seshat reads every non-secret setting from one TOML file

`SESHAT_CONFIG_FILE` selects the file and defaults to `/etc/seshat/seshat.toml`.
Start from [`seshat.example.toml`](../seshat.example.toml).

## File settings

| Setting | Default | Purpose |
| --- | --- | --- |
| `database.schema` | `"seshat"` | PostgreSQL schema holding Seshat's tables. |
| `embedding.base_url` | required | OpenAI-compatible embedding endpoint. |
| `embedding.model` | required | Embedding model, or provider deployment name. |
| `embedding.dimension` | required | Embedding dimension requested from the provider. |
| `embedding.batch_size` | `64` | Texts sent per embedding request. |
| `chunking.size` | `800` | Target search and scan passage size in tokens. |
| `chunking.overlap` | `120` | Tokens shared between splits of oversized content. Must be smaller than `chunking.size`. |
| `retrieval.mode` | `"hybrid"` | `hybrid` combines vector and keyword search; `vector` uses vector search alone. |
| `retrieval.default_top_k` | `8` | Results returned when a search omits `top_k`. |
| `retrieval.max_top_k` | `50` | Largest `top_k` a search may request. Must be at least `retrieval.default_top_k`. |
| `scan.default_limit` | `25` | Passage items returned when a scan omits `limit`. |
| `scan.max_limit` | `100` | Largest passage-item count accepted for one scan page. |
| `scan.max_payload_bytes` | `256000` | Largest aggregate UTF-8 text payload returned by one scan page. |
| `limits.max_document_bytes` | `8000000` | Largest extracted text accepted from one document. |
| `limits.max_canonical_block_bytes` | `64000` | Largest rendered canonical block accepted during ingestion. |
| `limits.max_upload_bytes` | `52428800` | Largest upload accepted; the maximum is 50 MiB. |
| `limits.max_filename_characters` | `255` | Longest accepted upload filename. |
| `storage.source_dir` | `"/var/lib/seshat/sources"` | Absolute path where original uploads are kept. |
| `converter.backend` | `"docling"` | Which service converts PDF and Office uploads. |
| `converter.docling.base_url` | empty | docling-serve instance. Empty rejects uploads that need conversion. |
| `converter.docling.timeout_seconds` | `120` | Timeout for one HTTP request to Docling (submit, poll, or result fetch). |
| `converter.docling.conversion_deadline_seconds` | `3600` | Total time one document's conversion may take, measured from submission and across restarts. Must be at least `converter.docling.timeout_seconds`. |
| `converter.docling.poll_interval_seconds` | `5` | Delay between Docling task status polls. Must not exceed `converter.docling.conversion_deadline_seconds`. |
| `converter.azure.endpoint` | empty | Azure Document Intelligence resource URL. Required when `converter.backend` is `azure`. |
| `converter.azure.model_id` | `"prebuilt-layout"` | Analysis model used for conversion. |
| `converter.azure.timeout_seconds` | `300` | Total time allowed for one document's analysis, including polling. |
| `logging.level` | `"INFO"` | Root log level. |

## Secrets

Provided as environment variables; the Compose stack reads them from `.env`.
See [`.env.example`](../.env.example).

| Variable | Required | Purpose |
| --- | --- | --- |
| `SESHAT_DATABASE_URL` | yes | PostgreSQL connection URL, including the password. |
| `SESHAT_API_TOKEN` | yes | Bearer token every API request must present; at least 16 characters. |
| `SESHAT_CURSOR_SIGNING_KEY` | yes | Server-only key for scan cursor integrity; at least 32 characters. |
| `SESHAT_EMBEDDING_API_KEY` | yes | Credential for the embedding endpoint. |
| `AZURE_OCR_API_KEY` | no | Credential for the Azure Document Intelligence resource. Required when `converter.backend` is `azure`. |
