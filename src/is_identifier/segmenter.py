"""
Paso 1 segmenter: document -> structured segments (list-aware, non-destructive).

Replaces the legacy behaviour where list items were merged into their parent
sentence (destroying list structure) or short bullets were dropped as junk.
Every block of the document is emitted as a segment with a neutral structural
description; nothing taxonomic is decided here (that is Paso 2).

segment_type values:
    body_sentence  - substantive sentence in article body
    list_header    - sentence/phrase introducing a list (usually ends with ':')
    list_item      - item of a list, linked to its header via parent_segment_id
    heading        - article/chapter/section heading
    preamble       - recitals, exposicion de motivos, considerandos
    signature      - closing/signature/approval formulas
    annex          - annex headings/content markers
    metadata       - other non-coding reference text

is_substantive is True for body_sentence / list_header / list_item, False for
the documentary-context types. Non-substantive segments ARE emitted (the
useful/non-useful filter must be visible and auditable downstream).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .document_processor import process_document
from .sentence_splitter import split_sentences
from .structure_detector import (
    classify_block,
    extract_article_ref,
    split_inline_article,
    split_page_block,
)

SUBSTANTIVE_TYPES = {"body_sentence", "list_header", "list_item"}

# Line-level list item markers: -, •, *, a), (a), 1., 1), i., (iv) ...
_LINE_ITEM = re.compile(
    r"^\s*(?:[-–—•*]\s+|\(?[a-záéíóúñ]\)\s+|\(?[A-ZÁÉÍÓÚÑ]\)\s+"
    r"|\(?\d{1,2}[\.\)]\s+|\(?[ivxlcdm]{1,6}[\.\)]\s+)",
    re.IGNORECASE,
)
# Inline item markers (lists collapsed into one line, e.g. PDFs):
_INLINE_ITEM = re.compile(r"(?:(?<=^)|(?<=[:;.]))\s*(\(?[a-z]\)|\d{1,2}[\.\)])\s+")

_SIGNATURE_PAT = re.compile(
    r"(?i)^(Dado\s+(en|a)\s+|En\s+fe\s+de|En\s+testimonio|Firmado|Rubricado|Firma\s+"
    r"|Declaro\s+estar|R[uú]brica|Nombre\s+y\s+apellido|Lugar\s+y\s+fecha"
    r"|Aprobado\s+en|Publicado\s+en|B\.O\.|D\.O\.|BOE|En\s+\w+,\s+a\s+\d+\s+de"
    r"|Signed\s+by|Dated|Approved\s+by|IN\s+WITNESS\s+WHEREOF"
    # standalone office lines ("El Presidente") — same rule as
    # structure_detector: only when (nearly) alone on the line
    r"|(?:El\s+Presidente|La\s+Secretar[ií]a?|El\s+Secretario|El\s+Tesorero)\s*[,.:]?\s*$)"
)
_ANNEX_PAT = re.compile(r"(?i)^(Anexo|Annex)\b")
# Editorial/audit annotations embedded in the markdown sources
# (e.g. "<!-- AUDIT:MOVE_TO=preamble;... -->") are not document text.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


@dataclass
class Segment:
    segment_id: str
    parent_segment_id: str | None
    segment_order: int
    segment_type: str
    is_list_item: bool
    is_substantive: bool
    segment_text: str
    title: str | None = None
    chapter: str | None = None
    article: str | None = None
    notes: list[str] = field(default_factory=list)


def _skip_subtype(text: str) -> str:
    if _SIGNATURE_PAT.match(text.strip()):
        return "signature"
    if _ANNEX_PAT.match(text.strip()):
        return "annex"
    return "preamble"


def _metadata_subtype(text: str) -> str:
    # "Anexo I" is classified METADATA by structure_detector but the contract
    # wants it surfaced as annex (non-substantive), not as a heading
    return "annex" if _ANNEX_PAT.match(text.strip()) else "heading"


def _split_heading_body(text: str) -> tuple[str, str]:
    """Split a METADATA block into (heading, trailing body).

    split_page_block cuts large page blocks at any "Section N"/"Capítulo N"
    reference — including references in the middle of a sentence — so a
    METADATA-classified chunk can carry whole normative paragraphs after the
    reference line (found 2026-06-10: a 12-AIM paragraph emitted as heading).
    First try the article-marker split; for long multi-line chunks fall back
    to first-line-is-the-heading.
    """
    _, body = split_inline_article(text)
    if body:
        return text[: len(text) - len(body)].strip(), body
    lines = text.splitlines()
    if len(lines) > 1 and len(text) > 150:
        return lines[0].strip(), "\n".join(lines[1:]).strip()
    return text, ""


# Items at or above this length with no list header in scope are treated as
# numbered paragraphs (legal sub-clauses "1) ...", multi-sentence), not items.
_PARAGRAPH_ITEM_CHARS = 120

_FAMILY_PATTERNS = [
    ("letter", re.compile(r"^\s*(?:[-–—•*]\s+)?\(?[a-záéíóúñ]\)", re.IGNORECASE)),
    ("roman", re.compile(r"^\s*(?:[-–—•*]\s+)?\(?[ivxlcdm]{1,6}[\.\)]", re.IGNORECASE)),
    ("number", re.compile(r"^\s*(?:[-–—•*]\s+)?\(?\d{1,2}[\.\)]")),
    ("bullet", re.compile(r"^\s*[-–—•*]\s+")),
]


def _marker_family(text: str) -> str:
    """Marker family of a list item ('letter', 'number', 'roman', 'bullet').

    A '- a)' compound marker is classified by its inner marker (letter): the
    family change number->letter / letter->number is what signals that a new
    enumeration started (e.g. legal sub-clause '3)' after items 'a)..d)')."""
    for family, pat in _FAMILY_PATTERNS:
        if pat.match(text):
            return family
    return "bullet"


def _split_inline_list(text: str) -> tuple[str, list[str]] | None:
    """Detect 'header: a) item b) item' collapsed into one line.

    Returns (header, items) when the text before the first marker ends with
    ':' and there are at least two markers; otherwise None (conservative).
    """
    matches = list(_INLINE_ITEM.finditer(text))
    if len(matches) < 2:
        return None
    header = text[: matches[0].start()].strip()
    if not header.endswith(":"):
        return None
    items = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        item = text[m.start():end].strip()
        if item:
            items.append(item)
    return (header, items) if len(items) >= 2 else None


def _parse_block_lines(text: str) -> list[tuple[str, str]]:
    """Split a block into (kind, text) pieces, kind in {'text','item'}.

    Operates on physical lines (markdown blocks preserve newlines). Continuation
    lines are appended to the previous piece.
    """
    pieces: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _LINE_ITEM.match(line):
            pieces.append(("item", line))
        elif pieces and line[:1].islower():
            # lowercase start: continuation of the previous line/item
            kind, prev = pieces[-1]
            pieces[-1] = (kind, prev + " " + line)
        else:
            # each remaining physical line is its own piece so that intra-block
            # context (signatures, preambles) can be classified individually
            pieces.append(("text", line))
    return pieces


class _SegmentBuilder:
    def __init__(self) -> None:
        self.segments: list[Segment] = []
        self.meta: dict[str, str | None] = {"title": None, "chapter": None, "article": None}
        # List state persists ACROSS markdown blocks: a header ending in ':'
        # must adopt items even when a blank line / page break separates them.
        self.last_header: Segment | None = None
        self.last_body: Segment | None = None
        self.list_marker_family: str | None = None

    def reset_list_state(self) -> None:
        self.last_header = None
        self.last_body = None
        self.list_marker_family = None

    def add(self, text: str, segment_type: str, parent: str | None = None,
            notes: list[str] | None = None) -> Segment:
        order = len(self.segments) + 1
        seg = Segment(
            segment_id=f"seg_{order:04d}",
            parent_segment_id=parent,
            segment_order=order,
            segment_type=segment_type,
            is_list_item=segment_type == "list_item",
            is_substantive=segment_type in SUBSTANTIVE_TYPES,
            segment_text=text,
            title=self.meta["title"],
            chapter=self.meta["chapter"],
            article=self.meta["article"],
            notes=notes or [],
        )
        self.segments.append(seg)
        return seg


def segment_document(path: str, language: str = "es") -> list[Segment]:
    """Segment a regulatory document into structure-aware segments."""
    builder = _SegmentBuilder()

    for block in process_document(path):
        raw = _HTML_COMMENT.sub(" ", block["text"]).strip()
        if not raw:
            continue
        sub_blocks = split_page_block(raw) if len(raw) > 400 else [raw]

        for sub in sub_blocks:
            sub = sub.strip()
            if not sub:
                continue
            block_type = classify_block(sub)

            if block_type == "SKIP":
                builder.add(sub, _skip_subtype(sub))
                builder.reset_list_state()
                continue

            if block_type == "METADATA":
                ref = extract_article_ref(sub)
                builder.meta.update({k: v for k, v in ref.items() if v is not None})
                heading_txt, body = _split_heading_body(sub)
                builder.add(heading_txt or sub, _metadata_subtype(sub))
                builder.reset_list_state()
                if not body:
                    continue
                sub = body
            else:
                _, body = split_inline_article(sub)
                if body:
                    ref = extract_article_ref(sub)
                    builder.meta.update({k: v for k, v in ref.items() if v is not None})
                    builder.add(sub[: len(sub) - len(body)].strip(), "heading")
                    sub = body

            _emit_relevant(builder, sub, language)

    return builder.segments


def _emit_relevant(builder: _SegmentBuilder, text: str, language: str) -> None:
    """Emit body text preserving list structure.

    List state lives on the builder so a header ending in ':' adopts items
    even across markdown block boundaries (blank lines, page breaks).
    """
    pieces = _parse_block_lines(text)

    for kind, piece in pieces:
        if kind == "item":
            _emit_item(builder, piece, language)
            continue

        # 'text' piece: documentary context can hide inside a RELEVANT block
        # (markdown blocks span several paragraphs between headers)
        piece_type = classify_block(piece)
        if piece_type == "SKIP":
            builder.add(piece, _skip_subtype(piece))
            builder.reset_list_state()
            continue
        if piece_type == "METADATA":
            ref = extract_article_ref(piece)
            builder.meta.update({k: v for k, v in ref.items() if v is not None})
            heading_txt, body = _split_heading_body(piece)
            builder.add(heading_txt or piece, _metadata_subtype(piece))
            builder.reset_list_state()
            if not body:
                continue
            piece = body

        # may itself contain an inline collapsed list
        inline = _split_inline_list(piece.replace("\n", " "))
        if inline:
            header_txt, items = inline
            header_seg = builder.add(header_txt, "list_header")
            builder.last_header = header_seg
            builder.list_marker_family = None
            for it in items:
                builder.add(it, "list_item", parent=header_seg.segment_id)
                if builder.list_marker_family is None:
                    builder.list_marker_family = _marker_family(it)
            continue

        for sentence in split_sentences(piece.replace("\n", " "), language=language):
            sentence = sentence.strip()
            if not sentence:
                continue
            if sentence.endswith(":"):
                builder.last_header = builder.add(sentence, "list_header")
                builder.last_body = None
                builder.list_marker_family = None
            else:
                builder.last_body = builder.add(sentence, "body_sentence")
                builder.last_header = None


def _emit_item(builder: _SegmentBuilder, piece: str, language: str) -> None:
    """Emit one line-level list item, deciding what it structurally is."""
    family = _marker_family(piece)

    # A marker-family change closes the active enumeration: legal sub-clause
    # "3) ..." after items "a)..d)" belongs to the article, not to the list.
    if (builder.last_header is not None
            and builder.list_marker_family is not None
            and family != builder.list_marker_family):
        builder.reset_list_state()

    if piece.rstrip().endswith(":"):
        # sub-header introducing its own items ("c) Su vestimenta:")
        parent = builder.last_header
        seg = builder.add(piece, "list_header",
                          parent=parent.segment_id if parent else None)
        builder.last_header = seg
        builder.last_body = None
        builder.list_marker_family = None
        return

    if builder.last_header is not None:
        builder.add(piece, "list_item", parent=builder.last_header.segment_id)
        if builder.list_marker_family is None:
            builder.list_marker_family = family
        return

    if len(piece) >= _PARAGRAPH_ITEM_CHARS:
        # numbered paragraph (legal sub-clause style "1) Multi-sentence ...");
        # autonomous body text, split into sentences, marker kept verbatim
        for sentence in split_sentences(piece, language=language):
            sentence = sentence.strip()
            if sentence:
                builder.last_body = builder.add(sentence, "body_sentence")
        return

    parent = builder.last_body
    builder.add(piece, "list_item",
                parent=parent.segment_id if parent else None)
    if builder.list_marker_family is None:
        builder.list_marker_family = family
