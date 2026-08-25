"""Turn an uploaded file into the normalized, located text the ingest path expects."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from app.parsing.errors import (
    ConversionUnavailableError,
    DocumentDecodeError,
    DocumentTooLargeError,
)
from app.parsing.formats import UploadFormat
from app.parsing.segments import (
    ConvertedDocument,
    DocumentSegment,
    build_converted_document,
    clean_text,
)


class DocumentConverter(Protocol):
    """Converts a binary document to located text."""

    @property
    def name(self) -> str: ...

    async def convert(
        self,
        *,
        filename: str,
        media_type: str,
        content: bytes,
    ) -> ConvertedDocument: ...


@runtime_checkable
class ResumableDocumentConverter(DocumentConverter, Protocol):
    """A converter whose in-flight work is addressable by a durable task id."""

    async def submit(
        self,
        *,
        filename: str,
        media_type: str,
        content: bytes,
    ) -> str: ...

    async def await_result(
        self, task_id: str, *, submitted_at: datetime
    ) -> ConvertedDocument: ...


@dataclass(frozen=True, slots=True)
class UploadedFile:
    """One accepted upload, already validated against a supported format."""

    filename: str
    media_type: str
    content: bytes
    format: UploadFormat


class DocumentNormalizer:
    """Produces indexable, located text for an uploaded file."""

    def __init__(
        self,
        *,
        converter: DocumentConverter | None,
        max_text_bytes: int,
    ) -> None:
        self._converter = converter
        self._max_text_bytes = max_text_bytes

    @property
    def conversion_available(self) -> bool:
        return self._converter is not None

    @property
    def converter_name(self) -> str | None:
        return self._converter.name if self._converter is not None else None

    @property
    def resumable_converter(self) -> ResumableDocumentConverter | None:
        """The converter, when its work can be resumed by task id."""
        if isinstance(self._converter, ResumableDocumentConverter):
            return self._converter
        return None

    def require_converter(self, upload_format: UploadFormat) -> DocumentConverter:
        if self._converter is None:
            raise ConversionUnavailableError(
                f"{upload_format.label} uploads need a document converter, which is not "
                "configured on this deployment. Plain text and Markdown uploads still work."
            )
        return self._converter

    async def to_document(self, upload: UploadedFile) -> ConvertedDocument:
        """Convert *upload* in one call, waiting for the converter throughout."""
        if upload.format.needs_conversion:
            converter = self.require_converter(upload.format)
            converted = await converter.convert(
                filename=upload.filename,
                media_type=upload.media_type,
                content=upload.content,
            )
        else:
            # A decoded text or Markdown file is its own normalized form, and it
            # has no pagination to report.
            converted = build_converted_document(
                [DocumentSegment(text=clean_text(_decode_text(upload.content)))]
            )
        return self.bounded(converted)

    async def submit_conversion(self, upload: UploadedFile) -> str:
        """Hand *upload* to a resumable converter and return its task id."""
        converter = self.resumable_converter
        if converter is None:  # pragma: no cover - callers check first
            raise ConversionUnavailableError(
                "This deployment's document converter cannot run conversions asynchronously."
            )
        return await converter.submit(
            filename=upload.filename,
            media_type=upload.media_type,
            content=upload.content,
        )

    async def await_conversion(
        self, task_id: str, *, submitted_at: datetime
    ) -> ConvertedDocument:
        """Follow an already-submitted conversion to its end."""
        converter = self.resumable_converter
        if converter is None:
            raise ConversionUnavailableError(
                "This deployment's document converter can no longer finish the conversion "
                "this document was submitted for. Upload the file again."
            )
        return self.bounded(
            await converter.await_result(task_id, submitted_at=submitted_at)
        )

    def bounded(self, converted: ConvertedDocument) -> ConvertedDocument:
        """Reject converted text that is empty or larger than this service indexes."""
        if not converted.text:
            raise DocumentDecodeError(
                "The document contains no readable text to index."
            )
        if len(converted.text.encode("utf-8")) > self._max_text_bytes:
            raise DocumentTooLargeError(
                "The document's text exceeds the size this service indexes."
            )
        return converted


def _decode_text(content: bytes) -> str:
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentDecodeError(
            "The file is not valid UTF-8 text. Save it as UTF-8 and upload it again."
        ) from exc
    return decoded
