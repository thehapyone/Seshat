"""Assemble ordered source items into useful reading passages."""

from collections.abc import Callable, Sequence
from typing import TypeVar

from llama_index.core.utils import get_tokenizer


T = TypeVar("T")


def passage_groups(
    items: Sequence[T], *, text_of: Callable[[T], str], target_tokens: int
) -> list[tuple[T, ...]]:
    """Pack adjacent items without leaving a short lead-in on its own."""
    tokenizer = get_tokenizer()
    groups: list[tuple[T, ...]] = []
    current: list[T] = []

    for item in items:
        candidate = (*current, item)
        text = "\n\n".join(text_of(part) for part in candidate)
        current_text = "\n\n".join(text_of(part) for part in current)
        if (
            current
            and len(tokenizer(current_text)) >= target_tokens // 2
            and len(tokenizer(text)) > target_tokens
        ):
            groups.append(tuple(current))
            current = [item]
        else:
            current.append(item)

    if current:
        groups.append(tuple(current))
    return groups
