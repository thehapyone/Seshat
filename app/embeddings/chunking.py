"""Chunking that keeps a document's structure retrievable."""

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    TextNode,
    TransformComponent,
)
from llama_index.core.utils import get_tokenizer

from app.models import SEARCH_CONTEXT_METADATA_KEY
from app.parsing.errors import DocumentTooLargeError
from app.parsing.segments import DocumentSegment
from app.passages import passage_groups
from app.search_text import context_header

TABLE_METADATA_KEYS = ("table_part", "table_parts", "table_ref", "table_header")

_SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|\n+)|[^.!?\n]+$")


@dataclass(frozen=True, slots=True)
class CanonicalBlock:
    """One bounded, ordered, non-overlapping unit of normalized source content."""

    ordinal: int
    kind: Literal["text", "table_part"]
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: tuple[str, ...] = ()
    source_section_ref: str | None = None
    table_ref: str | None = None
    table_caption: str = ""
    table_header: str = ""
    part: int | None = None
    parts: int | None = None

    @property
    def rendered_text(self) -> str:
        """Return the self-contained text exposed by a future scan operation."""
        if self.kind == "text":
            return self.text
        return "\n".join(
            value
            for value in (self.table_caption, self.table_header, self.text)
            if value
        )


def build_canonical_blocks(
    segments: Sequence[DocumentSegment], *, max_rendered_bytes: int
) -> tuple[CanonicalBlock, ...]:
    """Build scan-safe blocks before any overlapping retrieval chunking.

    Prose is split into byte-bounded adjacent slices without overlap. Tables are
    split only between rows; their caption and header are stored separately and
    included when enforcing the rendered-item limit.
    """
    if max_rendered_bytes < 1:
        raise ValueError("max_rendered_bytes must be positive")

    blocks: list[CanonicalBlock] = []
    for segment in segments:
        if segment.is_table:
            blocks.extend(_canonical_table_blocks(segment, max_rendered_bytes))
        else:
            blocks.extend(
                CanonicalBlock(
                    ordinal=0,
                    kind="text",
                    text=part,
                    page_start=segment.page,
                    page_end=segment.page_end,
                    section_path=segment.section_path,
                    source_section_ref=segment.section_ref,
                )
                for part in _split_utf8(segment.text, max_rendered_bytes)
            )

    return tuple(
        replace(block, ordinal=ordinal) for ordinal, block in enumerate(blocks)
    )


def _canonical_table_blocks(
    segment: DocumentSegment, max_rendered_bytes: int
) -> list[CanonicalBlock]:
    caption, header, rows = _table_parts(segment.text, segment.caption)
    if not rows:
        block = _table_block(segment, text=caption, caption="", header="")
        _require_fits(block, max_rendered_bytes)
        return [block]

    blocks: list[CanonicalBlock] = []
    current: list[str] = []
    for row in rows:
        candidate = _table_block(
            segment,
            text="\n".join((*current, row)),
            caption=caption,
            header=header,
        )
        if current and _byte_size(candidate.rendered_text) > max_rendered_bytes:
            blocks.append(
                _table_block(
                    segment,
                    text="\n".join(current),
                    caption=caption,
                    header=header,
                )
            )
            current = [row]
            _require_fits(
                _table_block(segment, text=row, caption=caption, header=header),
                max_rendered_bytes,
            )
        else:
            _require_fits(candidate, max_rendered_bytes)
            current.append(row)

    blocks.append(
        _table_block(
            segment,
            text="\n".join(current),
            caption=caption,
            header=header,
        )
    )
    total = len(blocks)
    return [
        replace(block, part=index, parts=total)
        for index, block in enumerate(blocks, start=1)
    ]


def _table_parts(text: str, caption: str) -> tuple[str, str, list[str]]:
    lines = text.splitlines()
    cleaned_caption = caption.strip()
    if cleaned_caption and lines and lines[0].strip() == cleaned_caption:
        lines = lines[1:]
    if len(lines) < 3:
        return cleaned_caption, "", ["\n".join(lines)] if lines else []
    return cleaned_caption, "\n".join(lines[:2]), lines[2:]


def _table_block(
    segment: DocumentSegment, *, text: str, caption: str, header: str
) -> CanonicalBlock:
    return CanonicalBlock(
        ordinal=0,
        kind="table_part",
        text=text,
        page_start=segment.page,
        page_end=segment.page_end,
        section_path=segment.section_path,
        source_section_ref=segment.section_ref,
        table_ref=segment.table_ref,
        table_caption=caption,
        table_header=header,
    )


def _require_fits(block: CanonicalBlock, maximum: int) -> None:
    if _byte_size(block.rendered_text) > maximum:
        raise DocumentTooLargeError(
            "A canonical block cannot fit within limits.max_canonical_block_bytes "
            "without splitting source content."
        )


def _split_utf8(text: str, maximum: int) -> list[str]:
    """Split *text* into adjacent UTF-8 byte-bounded slices, preferring whitespace."""
    parts: list[str] = []
    remaining = text
    while _byte_size(remaining) > maximum:
        encoded = remaining.encode("utf-8")
        boundary = min(maximum, len(encoded))
        while boundary and boundary < len(encoded) and encoded[boundary] & 0xC0 == 0x80:
            boundary -= 1
        if boundary == 0:
            raise DocumentTooLargeError(
                "A source character cannot fit within limits.max_canonical_block_bytes."
            )
        prefix = encoded[:boundary].decode("utf-8")
        whitespace = tuple(re.finditer(r"\s+", prefix))
        if whitespace:
            boundary_text = whitespace[-1].end()
            prefix = prefix[:boundary_text]
        parts.append(prefix)
        remaining = remaining[len(prefix) :]
    if remaining:
        parts.append(remaining)
    return parts


def _byte_size(text: str) -> int:
    return len(text.encode("utf-8"))


def split_sentences(text: str) -> list[str]:
    """Split sentences and manual lines without NLTK or downloaded corpora."""
    return _SENTENCE_PATTERN.findall(text)


def split_table(table: str, *, budget: int, tokenizer: Any) -> list[str]:
    """Split one Markdown table into parts of at most *budget* tokens.

    A row is never split and the header is repeated in every part, so a part may
    exceed *budget* when a single row does. Rows that lose their header also lose
    what each column means.
    """
    rows = table.split("\n")
    header, body = rows[:2], rows[2:]
    if not body:
        return [table]

    header_tokens = len(tokenizer("\n".join(header)))
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for row in body:
        row_tokens = len(tokenizer(row))
        if current and header_tokens + current_tokens + row_tokens > budget:
            parts.append("\n".join(header + current))
            current = []
            current_tokens = 0
        current.append(row)
        current_tokens += row_tokens
    parts.append("\n".join(header + current))
    return parts


class StructuralChunker(TransformComponent):
    """Assemble located source blocks into substantive search passages.

    A segment the converter identified as a table is kept whole when it fits and
    split on row boundaries when it does not. Oversized prose is split on sentence
    boundaries. The resulting pieces are packed in source order across paragraph,
    table, and section boundaries so converter fragments are not indexed alone.
    """

    chunk_size: int
    chunk_overlap: int
    splitter: SentenceSplitter

    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            splitter=SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunking_tokenizer_fn=split_sentences,
            ),
        )

    @classmethod
    def class_name(cls) -> str:
        return "StructuralChunker"

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[BaseNode]:
        pieces = [piece for node in nodes for piece in self._split_node(node)]
        groups = passage_groups(
            pieces,
            text_of=lambda piece: piece.get_content(metadata_mode=MetadataMode.NONE),
            target_tokens=self.chunk_size,
        )
        return [_passage(group) for group in groups]

    def _split_node(self, node: BaseNode) -> list[BaseNode]:
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        if not text.strip():
            return []

        header = context_header(node.metadata, text)
        if node.metadata.get("is_table"):
            chunks = self._table_chunks(node, text, header=header)
        else:
            chunks = list(self.splitter([node]))

        return chunks

    def _table_chunks(
        self, node: BaseNode, text: str, *, header: str
    ) -> list[BaseNode]:
        caption = str(node.metadata.get("caption") or "").strip()
        table_header = str(node.metadata.get("table_header") or "").strip()
        table = f"{table_header}\n{text}" if table_header else text
        prefix = "" if not caption or caption in table else f"{caption}\n"

        tokenizer = get_tokenizer()
        # The header row, caption and provenance line ride along in every part, so
        # the row budget has to leave room for them.
        reserved = len(tokenizer(f"{header}\n{prefix}")) if (header or prefix) else 0
        parts = split_table(
            table,
            budget=max(self.chunk_size - reserved, 1),
            tokenizer=tokenizer,
        )

        chunks = [_derived(node, f"{prefix}{part}") for part in parts]
        if len(chunks) > 1:
            for position, chunk in enumerate(chunks, start=1):
                chunk.metadata["table_part"] = position
                chunk.metadata["table_parts"] = len(chunks)
                _exclude_metadata(chunk, TABLE_METADATA_KEYS)
        return chunks


def _derived(node: BaseNode, text: str) -> TextNode:
    """One chunk of *node*, carrying its metadata, exclusions and provenance.

    The source relationship is copied because deletion is scoped by ``ref_doc_id``:
    a chunk that named an intermediate node would survive its document's re-ingest.
    """
    chunk = TextNode(text=text, metadata=dict(node.metadata))
    chunk.relationships = dict(node.relationships)
    chunk.excluded_embed_metadata_keys = list(node.excluded_embed_metadata_keys)
    chunk.excluded_llm_metadata_keys = list(node.excluded_llm_metadata_keys)
    return chunk


def _section_identity(node: BaseNode) -> tuple[tuple[str, ...], str | None, str]:
    raw_path = node.metadata.get("section_path")
    path = (
        tuple(part for part in raw_path if isinstance(part, str))
        if isinstance(raw_path, (list, tuple))
        else ()
    )
    raw_ref = node.metadata.get("source_section_ref")
    section_ref = raw_ref if isinstance(raw_ref, str) else None
    section = str(node.metadata.get("section") or "")
    return path, section_ref, section


def _passage(parts: tuple[BaseNode, ...]) -> TextNode:
    text = "\n\n".join(
        part.get_content(metadata_mode=MetadataMode.NONE) for part in parts
    )
    passage = _derived(parts[0], text)
    ordinals = list(
        dict.fromkeys(
            ordinal
            for part in parts
            for ordinal in part.metadata.get("block_ordinals", ())
        )
    )
    passage.metadata["block_ordinals"] = ordinals

    pages = [
        page
        for part in parts
        for page in (
            part.metadata.get("page"),
            part.metadata.get("page_end"),
        )
        if isinstance(page, int) and not isinstance(page, bool)
    ]
    if pages:
        passage.metadata["page"] = min(pages)
        passage.metadata["page_end"] = max(pages)

    if len({_section_identity(part) for part in parts}) > 1:
        for key in ("section_id", "section", "section_path", "source_section_ref"):
            passage.metadata.pop(key, None)
    if len(parts) > 1:
        for key in (*TABLE_METADATA_KEYS, "is_table", "caption"):
            passage.metadata.pop(key, None)

    _exclude_metadata(
        passage,
        (
            "block_ordinals",
            "page",
            "page_end",
            "section_id",
            "section",
            "section_path",
            "source_section_ref",
            SEARCH_CONTEXT_METADATA_KEY,
        ),
    )
    _apply_context(passage)
    return passage


def _apply_context(chunk: BaseNode) -> None:
    body = chunk.get_content(metadata_mode=MetadataMode.NONE)
    header = context_header(chunk.metadata, body)
    applied = header if header and not body.startswith(header) else ""
    chunk.metadata[SEARCH_CONTEXT_METADATA_KEY] = applied
    _exclude_metadata(chunk, (SEARCH_CONTEXT_METADATA_KEY,))
    chunk.set_content(f"{applied}\n{body}" if applied else body)


def _exclude_metadata(chunk: BaseNode, keys: Sequence[str]) -> None:
    for key in keys:
        if key in chunk.metadata:
            if key not in chunk.excluded_embed_metadata_keys:
                chunk.excluded_embed_metadata_keys.append(key)
            if key not in chunk.excluded_llm_metadata_keys:
                chunk.excluded_llm_metadata_keys.append(key)
