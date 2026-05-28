"""
Text extractor for regulatory documents (PDF, DOCX, TXT, Markdown).

Returns a list of text blocks preserving document order.
Each block is a dict: {text: str, page: int | None}
"""

from pathlib import Path


def process_document(path: str | Path) -> list[dict]:
    """
    Extract text blocks from a supported document.

    Args:
        path: Path to a .pdf, .docx, .txt, or .md file.

    Returns:
        List of dicts with keys 'text' (str) and 'page' (int or None).
        Each dict represents one paragraph or text block.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix == ".docx":
        return _extract_docx(path)
    elif suffix == ".doc":
        raise ValueError("Legacy .doc files are not supported directly. Convert to .docx first.")
    elif suffix == ".txt":
        return _extract_txt(path)
    elif suffix == ".md":
        return _extract_markdown(path)
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Expected .pdf, .docx, .txt, or .md")


def _extract_pdf(path: Path) -> list[dict]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required for PDF extraction: pip install pypdf")

    reader = PdfReader(str(path))
    blocks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for para in _split_into_paragraphs(text):
            if para.strip():
                blocks.append({"text": para.strip(), "page": page_num})
    return blocks


def _extract_docx(path: Path) -> list[dict]:
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for DOCX extraction: pip install python-docx")

    doc = Document(str(path))
    blocks = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            blocks.append({"text": text, "page": None})
    return blocks


def _extract_txt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [{"text": p.strip(), "page": None} for p in _split_into_paragraphs(text) if p.strip()]


def _split_into_paragraphs(text: str) -> list[str]:
    """Split raw PDF page text into paragraphs on blank lines."""
    import re
    paragraphs = re.split(r"\n{2,}", text)
    # Normalize single newlines within a paragraph to spaces
    return [re.sub(r"\n(?!\n)", " ", p).strip() for p in paragraphs]


def _extract_markdown(path: Path) -> list[dict]:
    """
    Extract text blocks from a Markdown regulation file.

    Splits on Markdown headers to produce article-level blocks.
    Headers are emitted as standalone blocks so structure_detector
    can classify them as METADATA.  Inline bold/italic and link
    syntax is stripped to plain text.
    """
    import re
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks: list[dict] = []
    current_lines: list[str] = []

    def _flush(buf: list[str]) -> None:
        chunk = "\n".join(buf).strip()
        if chunk:
            blocks.append({"text": chunk, "page": None})

    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^#{1,6}\s", stripped):
            _flush(current_lines)
            current_lines = []
            header = re.sub(r"^#{1,6}\s+", "", stripped).strip()
            if header:
                blocks.append({"text": header, "page": None})
        else:
            if re.match(r"^[-–—]{3,}$", stripped):
                _flush(current_lines)
                current_lines = []
                continue
            clean = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", stripped)
            clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)
            if clean:
                current_lines.append(clean)

    _flush(current_lines)
    return blocks
