"""Upload validation and normalization behind a converter-agnostic adapter."""

from app.parsing.azure_di import AzureDocumentIntelligenceClient
from app.parsing.docling import DoclingClient
from app.parsing.errors import (
    ConversionDeadlineExceededError,
    ConversionFailedError,
    ConversionResultUnavailableError,
    ConversionSubmissionError,
    ConversionTaskLostError,
    ConversionUnavailableError,
    DocumentDecodeError,
    DocumentError,
    DocumentTooLargeError,
    UnsupportedDocumentError,
)
from app.parsing.formats import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FORMATS,
    UploadFormat,
    resolve_format,
    sanitize_filename,
    supported_extensions_text,
)
from app.parsing.normalize import (
    DocumentConverter,
    DocumentNormalizer,
    ResumableDocumentConverter,
    UploadedFile,
)
from app.parsing.segments import (
    ConvertedDocument,
    DocumentSegment,
    build_converted_document,
    clean_text,
)

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_FORMATS",
    "AzureDocumentIntelligenceClient",
    "ConversionDeadlineExceededError",
    "ConversionFailedError",
    "ConversionResultUnavailableError",
    "ConversionSubmissionError",
    "ConversionTaskLostError",
    "ConversionUnavailableError",
    "ConvertedDocument",
    "DoclingClient",
    "DocumentConverter",
    "DocumentDecodeError",
    "DocumentError",
    "DocumentNormalizer",
    "DocumentSegment",
    "DocumentTooLargeError",
    "ResumableDocumentConverter",
    "UnsupportedDocumentError",
    "UploadFormat",
    "UploadedFile",
    "build_converted_document",
    "clean_text",
    "resolve_format",
    "sanitize_filename",
    "supported_extensions_text",
]
