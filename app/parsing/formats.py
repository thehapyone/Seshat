"""Supported upload formats and filename handling."""

import re
import unicodedata
from dataclasses import dataclass

from app.parsing.errors import UnsupportedDocumentError

# A stored filename is display metadata only; it is never used as a path.
_UNSAFE_FILENAME_CHARACTERS = re.compile(r"[\x00-\x1f\x7f/\\]")
_GENERIC_MEDIA_TYPES = frozenset(
    {
        "",
        "application/octet-stream",
        "binary/octet-stream",
        "application/x-www-form-urlencoded",
    }
)


@dataclass(frozen=True, slots=True)
class UploadFormat:
    """One accepted upload extension and how it becomes text."""

    extension: str
    source_type: str
    label: str
    canonical_media_type: str
    accepted_media_types: frozenset[str]
    needs_conversion: bool


def _format(
    extension: str,
    source_type: str,
    label: str,
    media_types: tuple[str, ...],
    *,
    needs_conversion: bool,
) -> UploadFormat:
    return UploadFormat(
        extension=extension,
        source_type=source_type,
        label=label,
        canonical_media_type=media_types[0],
        accepted_media_types=frozenset(media_types),
        needs_conversion=needs_conversion,
    )


SUPPORTED_FORMATS: dict[str, UploadFormat] = {
    format_.extension: format_
    for format_ in (
        _format(".txt", "text", "Plain text", ("text/plain",), needs_conversion=False),
        _format(
            ".md",
            "markdown",
            "Markdown",
            ("text/markdown", "text/x-markdown", "text/plain"),
            needs_conversion=False,
        ),
        _format(
            ".markdown",
            "markdown",
            "Markdown",
            ("text/markdown", "text/x-markdown", "text/plain"),
            needs_conversion=False,
        ),
        _format(".pdf", "pdf", "PDF", ("application/pdf",), needs_conversion=True),
        _format(
            ".docx",
            "docx",
            "Word document",
            (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            needs_conversion=True,
        ),
        _format(
            ".pptx",
            "pptx",
            "PowerPoint presentation",
            (
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ),
            needs_conversion=True,
        ),
        _format(
            ".xlsx",
            "xlsx",
            "Excel workbook",
            ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",),
            needs_conversion=True,
        ),
        _format(
            ".html", "html", "HTML document", ("text/html",), needs_conversion=True
        ),
    )
}

SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(sorted(SUPPORTED_FORMATS))


def supported_extensions_text() -> str:
    return ", ".join(SUPPORTED_EXTENSIONS)


def sanitize_filename(raw: str, *, maximum_characters: int) -> str:
    """Reduce a browser-supplied name to a bounded, path-free display label."""
    candidate = unicodedata.normalize("NFC", raw).strip()
    # Defend against "dir/name.pdf" and Windows "C:\dir\name.pdf" alike.
    candidate = candidate.replace("\\", "/").rsplit("/", 1)[-1]
    candidate = _UNSAFE_FILENAME_CHARACTERS.sub("", candidate).strip().strip(".")
    if not candidate:
        raise UnsupportedDocumentError(
            "The uploaded file needs a name with a supported extension."
        )
    if len(candidate) > maximum_characters:
        raise UnsupportedDocumentError(
            f"The file name must be at most {maximum_characters} characters."
        )
    return candidate


def resolve_format(filename: str, media_type: str | None) -> UploadFormat:
    """Return the format for *filename*, rejecting anything unsupported."""
    _, separator, extension = filename.rpartition(".")
    extension = f".{extension.lower()}" if separator else ""
    format_ = SUPPORTED_FORMATS.get(extension)
    if format_ is None:
        raise UnsupportedDocumentError(
            f"{extension or 'That file type'} is not supported. "
            f"Upload one of: {supported_extensions_text()}."
        )

    declared = (media_type or "").split(";", 1)[0].strip().lower()
    if (
        declared not in _GENERIC_MEDIA_TYPES
        and declared not in format_.accepted_media_types
    ):
        raise UnsupportedDocumentError(
            f"The content type '{declared}' does not match a {format_.label} file."
        )
    return format_
