# Seshat Agent Guide

## Purpose and boundaries

Seshat ingests sources and returns grounded evidence. It does not decide which
capability a caller should use, perform LLM reasoning, compare evidence, extract
domain records, or produce final answers. Keep those responsibilities with the
calling agent or application.

The public knowledge capabilities have distinct contracts:

- `search` returns ranked, incomplete evidence for focused questions.
- `outline` returns converter-recognized structure and fixed structural counts.
  A `null` count means the converter could not establish it; it is not zero.
- `scan` returns deterministic source or section content. Exhaustive coverage is
  established only after following cursors until `next_cursor` is `null`.

## Design principles

- Apply YAGNI. Implement requirements demonstrated by a current caller or public
  contract; do not add speculative modes, compatibility layers, fallbacks, IDs,
  response fields, or extension points.
- Keep one clear path for each behavior. Reuse the central settings declaration,
  representation builders, repository contracts, and API models instead of
  maintaining parallel implementations.
- Do not add defensive branches for impossible states or hide failures behind
  best-effort behavior. Validate real trust boundaries and invariants, then fail
  explicitly when they are violated.
- Preserve converter facts; do not infer missing headings, relationships, pages,
  tables, figures, or semantic records.
- Keep canonical scan blocks ordered and non-overlapping. Retrieval chunks may
  overlap, but search must never imply complete coverage.
- Keep changes narrow. Avoid unrelated renames, formatting churn, abstractions,
  and cleanup in feature patches.
- **Docs are part of the change**: when you change behavior or architecture, update relevant docs in the same patch.
- **Prefer deletion over addition**: remove deprecated paths if replaced and safe to delete.
- **Conventional commits only**: use conventional commit format with a concise summary.
- **No AI fingerprints**: never add AI attribution in commit messages or code comments.

## Code conventions

- Follow the existing typed, asynchronous style.
- Prefer small functions with explicit inputs and return types over stateful
  helpers or generic frameworks.
- Put API validation in Pydantic models, shared configuration in `app/config.py`,
  persistence behavior behind repository interfaces, and converter-specific
  translation under `app/parsing/`.
- Raise specific domain errors at the layer that can explain the failure. Map
  them to HTTP responses at the API boundary.
- Comments should explain an invariant or non-obvious reason, not narrate code.
- Never expose internal document, revision, section, block, or vector identifiers
  through public API models. Public job IDs, opaque section references, and scan
  cursors are part of the supported contracts.
- **Stay DRY**: avoid duplicating logic across subsystems.

## Tests

- Name tests after durable behavior.
- Prefer the smallest test that proves the contract. Use unit tests for local
  behavior, PostgreSQL integration tests for persistence boundaries, and live
  converter tests only for the real external contract.
- Keep fixtures synthetic and domain-neutral.
- When changing settings, update `SETTINGS`, `seshat.example.toml`, and regenerate
  `docs/configuration.md` with `python scripts/generate_config_reference.py`.

## Verification

Start with the narrowest relevant test, then broaden according to risk:

```bash
uv run pytest -q <targeted tests>
uv run pytest -q
docker build -t seshat:local .
```

PostgreSQL tests require `SESHAT_TEST_DATABASE_URL`. Converter tests that require
a live service remain explicitly opt-in. Review `git diff --check`, the complete
diff, and `git status` before handing work back. Do not commit transient build
artifacts, credentials, local configuration, source documents, or unrelated user
changes.

For agent integrations, follow
[`.agents/skills/seshat-knowledge/SKILL.md`](.agents/skills/seshat-knowledge/SKILL.md).
