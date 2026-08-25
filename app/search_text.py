"""Render and refresh mutable provenance around immutable search text."""

from typing import Any

_HEADING_LOOKAHEAD_CHARACTERS = 200


def context_header(metadata: dict[str, Any], text: str = "") -> str:
    """Return the display provenance line for one search chunk."""
    title = str(metadata.get("title") or "").strip()
    section = str(metadata.get("section") or "").strip()
    if section and section in text[: len(section) + _HEADING_LOOKAHEAD_CHARACTERS]:
        section = ""
    located = " > ".join(part for part in (title, section) if part)
    page = metadata.get("page")
    if isinstance(page, int) and not isinstance(page, bool):
        located = f"{located} - page {page}" if located else f"page {page}"
    return f"[{located}]" if located else ""


def contextual_search_text(metadata: dict[str, Any], body: str) -> str:
    """Render mutable display provenance around an immutable search body."""
    header = context_header(metadata, body)
    if not header or body.startswith(header):
        return body
    return f"{header}\n{body}"


def search_body(
    text: str,
    context: str,
    *,
    legacy_metadata: dict[str, Any] | None = None,
) -> str:
    """Remove a context line that Seshat previously added to search text."""
    prefix = f"{context}\n"
    if context and text.startswith(prefix):
        return text[len(prefix) :]
    if legacy_metadata is None:
        return text

    first_line, separator, body = text.partition("\n")
    if not separator:
        return text
    sectionless = {**legacy_metadata, "section": ""}
    legacy_headers = {
        context_header(legacy_metadata),
        context_header(sectionless),
    }
    return body if first_line and first_line in legacy_headers else text
