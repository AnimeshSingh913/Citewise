"""Turn uploaded bytes into page/section-aware text. Every Section keeps the page it came
from, so citations can point at a real page instead of a placeholder."""

import io
import re
import zipfile
from dataclasses import dataclass

PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MARKDOWN = "text/markdown"
TEXT = "text/plain"

SUPPORTED_EXTENSIONS = {".pdf": PDF, ".docx": DOCX, ".md": MARKDOWN, ".markdown": MARKDOWN, ".txt": TEXT}


class UnsupportedFile(ValueError):
    pass


@dataclass
class Section:
    text: str
    page_number: int | None = None
    heading: str | None = None


@dataclass
class ParsedDocument:
    sections: list[Section]
    page_count: int | None


def detect_media_type(filename: str, head: bytes) -> str:
    """Decide the type from content, using the extension only to choose between formats
    that share a container (zip) or have no magic bytes (text)."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFile("Supported formats: PDF, DOCX, Markdown, plain text")
    expected = SUPPORTED_EXTENSIONS[ext]
    if expected == PDF:
        if not head.startswith(b"%PDF-"):
            raise UnsupportedFile("File has a .pdf extension but is not a PDF")
    elif expected == DOCX:
        if not head.startswith(b"PK\x03\x04"):
            raise UnsupportedFile("File has a .docx extension but is not a Word document")
    else:
        if b"\x00" in head:
            raise UnsupportedFile("Text files must not contain binary data")
    return expected


def parse(data: bytes, media_type: str) -> ParsedDocument:
    if media_type == PDF:
        return _parse_pdf(data)
    if media_type == DOCX:
        return _parse_docx(data)
    if media_type in (MARKDOWN, TEXT):
        text = _decode_text(data)
        sections = _split_markdown(text) if media_type == MARKDOWN else [Section(text=text)]
        return ParsedDocument(sections=[s for s in sections if s.text.strip()], page_count=None)
    raise UnsupportedFile(f"Unsupported media type: {media_type}")


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnsupportedFile("Text files must be UTF-8 or UTF-16 encoded")


def _parse_pdf(data: bytes) -> ParsedDocument:
    import pymupdf

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise UnsupportedFile("The PDF is corrupted or encrypted") from e
    with doc:
        if doc.needs_pass:
            raise UnsupportedFile("Password-protected PDFs are not supported")
        sections = []
        for index, page in enumerate(doc):
            text = page.get_text("text", sort=True)
            if text.strip():
                sections.append(Section(text=text, page_number=index + 1))
        if not sections:
            raise UnsupportedFile("No extractable text found (scanned PDFs need OCR first)")
        return ParsedDocument(sections=sections, page_count=doc.page_count)


def _parse_docx(data: bytes) -> ParsedDocument:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except (zipfile.BadZipFile, KeyError, ValueError) as e:
        raise UnsupportedFile("The Word document is corrupted") from e

    sections: list[Section] = []
    heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            sections.append(Section(text="\n".join(buffer), heading=heading))
            buffer.clear()

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name if paragraph.style is not None else "") or ""
        if style.lower().startswith(("heading", "title")):
            flush()
            heading = text[:500]
            buffer.append(text)
        else:
            buffer.append(text)
    flush()

    for table in document.tables:
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        if any(r.strip(" |") for r in rows):
            sections.append(Section(text="\n".join(rows), heading="Table"))

    if not sections:
        raise UnsupportedFile("The Word document contains no text")
    return ParsedDocument(sections=sections, page_count=None)


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _split_markdown(text: str) -> list[Section]:
    sections: list[Section] = []
    heading: str | None = None
    buffer: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        match = None if in_fence else _MD_HEADING.match(line)
        if match:
            if buffer:
                sections.append(Section(text="\n".join(buffer), heading=heading))
                buffer = []
            heading = match.group(2)[:500]
        buffer.append(line)
    if buffer:
        sections.append(Section(text="\n".join(buffer), heading=heading))
    return sections
