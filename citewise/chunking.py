"""Parent/child chunking that never crosses a page or section boundary.

Parents (~1200 chars) are what Claude reads and cites; children (~300 chars) are what we
embed, so a precise match on a small window pulls in its larger, readable parent."""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from citewise.parsing import Section

_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class ParentChunk:
    index: int
    text: str
    page_number: int | None
    heading: str | None
    children: list[str] = field(default_factory=list)


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")  # non-breaking space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, size: int, overlap: int = 0) -> list[str]:
    """Recursive splitter: prefer paragraph, then line, sentence, word boundaries."""
    if len(text) <= size:
        return [text] if text.strip() else []
    pieces = _split_recursive(text, size, 0)
    if overlap <= 0:
        return pieces
    out = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:], strict=False):
        tail = prev[-overlap:]
        cut = tail.find(" ")
        tail = tail[cut + 1 :] if cut != -1 else tail
        out.append(f"{tail} {cur}".strip() if tail else cur)
    return out


def _split_recursive(text: str, size: int, level: int) -> list[str]:
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    if level >= len(_SEPARATORS):
        return [text[i : i + size].strip() for i in range(0, len(text), size) if text[i : i + size].strip()]
    sep = _SEPARATORS[level]
    parts = text.split(sep)
    if len(parts) == 1:
        return _split_recursive(text, size, level + 1)
    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= size:
            current = candidate
            continue
        if current.strip():
            chunks.append(current.strip())
        if len(part) > size:
            chunks.extend(_split_recursive(part, size, level + 1))
            current = ""
        else:
            current = part
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_sections(
    sections: list[Section], parent_size: int, parent_overlap: int, child_size: int
) -> Iterator[ParentChunk]:
    index = 0
    for section in sections:
        text = _normalize(section.text)
        if not text:
            continue
        for parent_text in split_text(text, parent_size, parent_overlap):
            children = split_text(parent_text, child_size) or [parent_text]
            yield ParentChunk(
                index=index,
                text=parent_text,
                page_number=section.page_number,
                heading=section.heading,
                children=children,
            )
            index += 1
