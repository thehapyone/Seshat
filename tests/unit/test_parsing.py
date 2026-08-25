"""Format resolution, filename safety, and text normalization."""

import pytest

from app.parsing import (
    ConversionUnavailableError,
    ConvertedDocument,
    DocumentDecodeError,
    DocumentNormalizer,
    DocumentSegment,
    DocumentTooLargeError,
    SUPPORTED_FORMATS,
    UnsupportedDocumentError,
    UploadedFile,
    build_converted_document,
    resolve_format,
    sanitize_filename,
)


def upload(filename: str, content: bytes, media_type: str | None = None) -> UploadedFile:
    document_format = resolve_format(filename, media_type)
    return UploadedFile(
        filename=filename,
        media_type=document_format.canonical_media_type,
        content=content,
        format=document_format,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("manual.txt", "manual.txt"),
        ("  spaced.md  ", "spaced.md"),
        ("../../etc/passwd.txt", "passwd.txt"),
        ("/absolute/path/notes.md", "notes.md"),
        (r"C:\Users\tester\report.pdf", "report.pdf"),
        ("with\x00null.txt", "withnull.txt"),
        ("line\nbreak.txt", "linebreak.txt"),
    ],
)
def test_filenames_are_reduced_to_a_path_free_label(raw: str, expected: str) -> None:
    assert sanitize_filename(raw, maximum_characters=255) == expected


@pytest.mark.parametrize("raw", ["", "   ", "...", "/", "\\"])
def test_unusable_filenames_are_rejected(raw: str) -> None:
    with pytest.raises(UnsupportedDocumentError):
        sanitize_filename(raw, maximum_characters=255)


def test_filename_length_is_bounded() -> None:
    with pytest.raises(UnsupportedDocumentError, match="at most 32 characters"):
        sanitize_filename(f"{'a' * 40}.txt", maximum_characters=32)


def test_supported_formats_resolve_by_extension() -> None:
    assert resolve_format("notes.TXT", "text/plain").source_type == "text"
    assert resolve_format("notes.md", None).source_type == "markdown"
    assert resolve_format("report.pdf", "application/pdf").needs_conversion is True
    assert resolve_format("report.docx", None).needs_conversion is True
    # Browsers commonly send a generic type; the extension still decides.
    assert resolve_format("report.pdf", "application/octet-stream").source_type == "pdf"


@pytest.mark.parametrize("filename", ["archive.zip", "script.exe", "noextension", "photo.png"])
def test_unsupported_extensions_are_rejected(filename: str) -> None:
    with pytest.raises(UnsupportedDocumentError):
        resolve_format(filename, None)


def test_a_declared_type_that_contradicts_the_extension_is_rejected() -> None:
    with pytest.raises(UnsupportedDocumentError, match="does not match"):
        resolve_format("report.pdf", "text/html")


def test_only_text_formats_skip_the_converter() -> None:
    local = {name for name, fmt in SUPPORTED_FORMATS.items() if not fmt.needs_conversion}
    assert local == {".txt", ".md", ".markdown"}


async def test_text_uploads_normalize_without_a_converter() -> None:
    normalizer = DocumentNormalizer(converter=None, max_text_bytes=1_000)

    converted = await normalizer.to_document(
        upload("manual.txt", b"\xef\xbb\xbfLine one\r\nLine two\r\n")
    )

    assert converted.text == "Line one\nLine two"
    # A text file has no pagination, so none is claimed for it.
    assert converted.page_count is None
    assert converted.recognized_section_count is None
    assert converted.recognized_table_count is None
    assert converted.recognized_figure_count is None
    assert [segment.page for segment in converted.segments] == [None]
    assert converted.has_provenance is False


async def test_non_utf8_text_is_rejected_with_a_readable_message() -> None:
    normalizer = DocumentNormalizer(converter=None, max_text_bytes=1_000)

    with pytest.raises(DocumentDecodeError, match="UTF-8"):
        await normalizer.to_document(upload("manual.txt", b"\xff\xfe\x00bad"))


async def test_whitespace_only_text_is_rejected() -> None:
    normalizer = DocumentNormalizer(converter=None, max_text_bytes=1_000)

    with pytest.raises(DocumentDecodeError, match="no readable text"):
        await normalizer.to_document(upload("manual.md", b"   \n\n\t "))


async def test_converted_text_is_bounded() -> None:
    normalizer = DocumentNormalizer(converter=None, max_text_bytes=16)

    with pytest.raises(DocumentTooLargeError):
        await normalizer.to_document(upload("manual.txt", b"x" * 64))


async def test_pdf_without_a_converter_reports_the_limitation() -> None:
    normalizer = DocumentNormalizer(converter=None, max_text_bytes=1_000)

    with pytest.raises(ConversionUnavailableError, match="not configured"):
        await normalizer.to_document(upload("report.pdf", b"%PDF-1.7"))


async def test_converted_documents_go_through_the_adapter() -> None:
    seen: dict[str, object] = {}

    class StubConverter:
        name = "stub"

        async def convert(
            self, *, filename: str, media_type: str, content: bytes
        ) -> ConvertedDocument:
            seen.update(filename=filename, media_type=media_type, content=content)
            return build_converted_document(
                [DocumentSegment(text="# Heading\r\n\r\nBody text.\x07")]
            )

    normalizer = DocumentNormalizer(converter=StubConverter(), max_text_bytes=1_000)

    converted = await normalizer.to_document(upload("report.docx", b"PK\x03\x04"))

    assert converted.text == "# Heading\n\nBody text."
    assert seen["filename"] == "report.docx"
    assert seen["media_type"] == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert seen["content"] == b"PK\x03\x04"


async def test_located_segments_survive_normalization() -> None:
    class LocatedConverter:
        name = "located"

        async def convert(
            self, *, filename: str, media_type: str, content: bytes
        ) -> ConvertedDocument:
            return build_converted_document(
                [
                    DocumentSegment(text="Battery replacement", page=4, section="2 Maintenance"),
                    DocumentSegment(text="Empty section", page=9, section=""),
                    DocumentSegment(text="   ", page=11, section="Dropped"),
                ]
            )

    normalizer = DocumentNormalizer(converter=LocatedConverter(), max_text_bytes=1_000)

    converted = await normalizer.to_document(upload("report.pdf", b"%PDF-1.7"))

    assert [(segment.page, segment.section) for segment in converted.segments] == [
        (4, "2 Maintenance"),
        (9, ""),
    ]
    assert [segment.page_end for segment in converted.segments] == [4, 9]
    assert [segment.section_path for segment in converted.segments] == [
        ("2 Maintenance",),
        (),
    ]
    # The blank segment is dropped, so it cannot contribute a page either.
    assert converted.page_count == 9
    assert converted.text == "Battery replacement\n\nEmpty section"
    assert converted.has_provenance is True
