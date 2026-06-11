"""
Document structure classifier for regulatory texts.

Classifies text blocks as:
  RELEVANT  – body text that contains (or may contain) institutional statements
  SKIP      – preambles, signatures, boilerplate to ignore
  METADATA  – article/chapter/section headings (used as reference, not coded)
"""

import re
from typing import Literal

BlockType = Literal["RELEVANT", "SKIP", "METADATA"]

# ── SKIP patterns ────────────────────────────────────────────────────────────
# Preambles, recitals, closing formulas, signatures

_SKIP_PATTERNS = [
    # Preamble openings (Spanish)
    re.compile(r"(?i)^(CONSIDERANDO|VISTO|POR\s+CUANTO|HABIENDO\s+VISTO)"),
    re.compile(r"(?i)^(EXPOSICI[OÓ]N\s+DE\s+MOTIVOS|PRE[AÁ]MBULO)"),
    re.compile(r"(?i)^(EN\s+SU\s+VIRTUD|POR\s+LO\s+TANTO|POR\s+TANTO)"),
    # Closing / signature formulas (Spanish)
    re.compile(r"(?i)^(Dado\s+(en|a)\s+|En\s+fe\s+de\s+lo\s+cual|En\s+testimonio)"),
    re.compile(r"(?i)^(Firmado|Rubricado)"),
    # Signature lines like "El Presidente" / "La Secretaria" only when they
    # stand (nearly) alone — sentences starting with "El Presidente convocará…"
    # are normative body text (bug found 2026-06-10 via context_filter_conflict)
    re.compile(r"(?i)^(El\s+Presidente|La\s+Secretar[ií]a?|El\s+Secretario|El\s+Tesorero)\s*[,.:]?\s*$"),
    re.compile(r"(?i)^(Aprobado\s+en|Publicado\s+en\s+el|B\.O\.|D\.O\.|BOE)"),
    re.compile(r"(?i)^(En\s+\w+,\s+a\s+\d+\s+de)"),        # "En Madrid, a 5 de..."
    re.compile(r"(?i)^Declaro\s+estar\s+de\s+acuerdo"),      # signature consent clause
    re.compile(r"(?i)^Firma\s+(del|de\s+la|y\s+sello)"),     # "Firma del Regante"
    re.compile(r"(?i)^(R[uú]brica|Nombre\s+y\s+apellido|Lugar\s+y\s+fecha)\s*:?$"),
    # Preamble openings (English)
    re.compile(r"(?i)^(WHEREAS|RECITALS|IN\s+WITNESS\s+WHEREOF|NOW\s+THEREFORE)"),
    re.compile(r"(?i)^(Signed\s+by|Dated|Approved\s+by)"),
    # Founding / promulgation statements (English)
    re.compile(r"(?i)^We\s+.{0,80}\bcome\s+together\b"),     # "We ... come together and form"
    re.compile(r"(?i)^The\s+above\s+\w+\s+at\s+a\s+general\s+meeting\s+held"),
    re.compile(r"(?i)^This\s+(constitution|document|agreement|regulation)\s+when\s+drafted"),
]

# ── METADATA patterns ────────────────────────────────────────────────────────
# Article, chapter, section headings — used as reference anchors, not coded

# "º" = U+00BA (MASCULINE ORDINAL INDICATOR) — used in Spanish scanned PDFs
_ART_PREFIX = r"(?:Art[ií]culo|Art\xba|Art°|Art\.|Article)"
_ROMAN_OR_DIGIT = r"(?:[IVXLCDMivxlcdm]+|\d+(?:\.\d+)?)"

_METADATA_PATTERNS = [
    re.compile(r"(?i)^" + _ART_PREFIX + r"\s*" + _ROMAN_OR_DIGIT),
    re.compile(r"(?i)^Cap[ií]tulo\s+[IVXivx\d]+"),
    re.compile(r"(?i)^T[ií]tulo\s+[IVXivx\d]+"),
    re.compile(r"(?i)^Secci[oó]n\s+[IVXivx\d]+"),
    re.compile(r"(?i)^Disposici[oó]n\s+(adicional|transitoria|final|derogatoria)"),
    re.compile(r"(?i)^Anexo\s+[IVXivx\d]*"),
    # English
    re.compile(r"(?i)^ARTICLE\s+" + _ROMAN_OR_DIGIT),
    re.compile(r"(?i)^Chapter\s+[IVXivx\d]+"),
    re.compile(r"(?i)^(Section|Sec\.)\s+" + _ROMAN_OR_DIGIT),
    re.compile(r"(?i)^Annex\s+[IVXivx\d]*"),
    # English keyword headings: only when the line has no lowercase tail
    # ("MEMBERSHIP" yes; "Membership is open to households..." is body text —
    # bug found 2026-06-10 via context_filter_conflict)
    re.compile(r"(?i)^(BY-LAWS|CONSTITUTION\s+AND\s+BY-LAWS|BOARD\s+OF\s+TRUSTEES|DUTIES\s+AND\s+RESPONSIBILITIES|MEMBERSHIP|MEETINGS|QUORUM|ELECTION\s+OF\s+OFFICERS)\b(?=[^a-z]*$)"),
]

# ── Article reference extractor ──────────────────────────────────────────────

_TITLE_RE = re.compile(r"(?i)T[ií]tulo\s+([IVXivx\d]+(?:\.[ \t]+[^\n.]{1,60})?)")
# Limit chapter capture to avoid greedily consuming the rest of the document
_CHAPTER_RE = re.compile(
    r"(?i)Cap[ií]tulo\s+([IVXivx\d]+(?:[\.\-]\s*[^\n]{1,80})?)"
    r"|Chapter\s+([IVXivx\d]+(?:[\.\:]\s*[^\n]{1,80})?)"
)
_ARTICLE_RE = re.compile(r"(?i)(?:" + _ART_PREFIX + r")\s*(" + _ROMAN_OR_DIGIT + r")")
_SECTION_RE = re.compile(r"(?i)(?:Section|Sec\.)\s*(" + _ROMAN_OR_DIGIT + r")")

# Patterns that mark the START of an article number (to split inline content)
_INLINE_ARTICLE_RE = re.compile(
    r"(?i)(" + _ART_PREFIX + r")\s*(" + _ROMAN_OR_DIGIT + r"[\xba°]?)\s*[\.\-–\xba°:]?",
)


def split_inline_article(text: str) -> tuple[str, str]:
    """
    If a block starts with an article marker followed by body text,
    return (article_ref, body_content).  Otherwise return (text, "").

    Example:
        "Art. 5. Para el control..." → ("Art. 5", "Para el control...")
        "Capítulo 3: Conservación"  → ("Capítulo 3: Conservación", "")
    """
    stripped = text.strip()
    m = _INLINE_ARTICLE_RE.match(stripped)
    if not m:
        return stripped, ""

    # Everything after the matched article prefix
    rest = stripped[m.end():].strip()
    # Remove a leading period that may remain after "Art. 5."
    if rest.startswith("."):
        rest = rest[1:].strip()
    # Only split if there is meaningful body text (> 10 chars)
    if len(rest) > 10:
        return stripped[: m.end()].strip().rstrip(".–-"), rest
    return stripped, ""


# Pattern to detect article/chapter starts anywhere mid-text (for large page blocks)
_INLINE_SPLIT_RE = re.compile(
    r"(?i)(?<!\w)"
    + r"(?:"
    + r"(?:" + _ART_PREFIX + r")\s*" + _ROMAN_OR_DIGIT + r"[\xba°]?\s*[\.\-–\xba°:]?"
    + r"|Cap[ií]tulo\s+[IVXivx\d]+"
    + r"|Chapter\s+[IVXivx\d]+"
    + r"|(?:Section|Sec\.)\s+" + _ROMAN_OR_DIGIT
    + r")"
)


def split_page_block(text: str) -> list[str]:
    """
    Split a large page-sized block at article/chapter boundaries.

    Used when a PDF returns an entire page as one text block.
    Returns a list of sub-blocks (each still may contain the article header
    merged with its body text, which split_inline_article handles).
    """
    text = text.strip()
    positions = [m.start() for m in _INLINE_SPLIT_RE.finditer(text)]
    if not positions:
        return [text]

    segments: list[str] = []
    prev = 0
    for pos in positions:
        chunk = text[prev:pos].strip()
        if chunk:
            segments.append(chunk)
        prev = pos
    tail = text[prev:].strip()
    if tail:
        segments.append(tail)
    return segments or [text]


def classify_block(text: str) -> BlockType:
    """Classify a text block as RELEVANT, SKIP, or METADATA."""
    stripped = text.strip()
    if not stripped:
        return "SKIP"

    for pat in _SKIP_PATTERNS:
        if pat.search(stripped):
            return "SKIP"

    for pat in _METADATA_PATTERNS:
        if pat.match(stripped):
            return "METADATA"

    return "RELEVANT"


def extract_article_ref(text: str) -> dict:
    """
    Extract structural reference from a METADATA block.

    Returns dict with keys 'title', 'chapter', 'article' (any may be None).
    """
    ref: dict[str, str | None] = {"title": None, "chapter": None, "article": None}

    m = _TITLE_RE.search(text)
    if m:
        ref["title"] = m.group(1).strip().rstrip(".")

    m = _CHAPTER_RE.search(text)
    if m:
        ref["chapter"] = (m.group(1) or m.group(2) or "").strip().rstrip(".")

    m = _ARTICLE_RE.search(text)
    if m:
        ref["article"] = m.group(1).strip()
    else:
        m = _SECTION_RE.search(text)
        if m:
            ref["article"] = m.group(1).strip()

    return ref
