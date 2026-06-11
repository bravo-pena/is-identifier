"""
Golden structure tests for the Paso 1 segmenter (paso1_polish_20260610).

Complements tests/test_segmenter.py: these tests pin the classification of
every documentary structure the contract cares about, including the two
regression bugs found 2026-06-10 via the context_filter_conflict flag
(signature "El Presidente" vs normative sentence; English keyword heading
"MEMBERSHIP" vs normative "Membership is open to...").
"""

from __future__ import annotations

import pytest

from is_identifier.paso1_output import _review_flags, pipeline_paso1
from is_identifier.segmenter import Segment, segment_document
from is_identifier.structure_detector import classify_block


def _segment_text_doc(tmp_path, text: str, language: str = "es"):
    doc = tmp_path / "golden.md"
    doc.write_text(text, encoding="utf-8")
    return segment_document(str(doc), language=language)


def _by_text(segments, fragment: str):
    matches = [s for s in segments if fragment in s.segment_text]
    assert matches, f"no segment contains {fragment!r}: " + \
        "; ".join(f"[{s.segment_type}] {s.segment_text[:50]}" for s in segments)
    return matches[0]


# ── block classification (structure_detector) ────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    # preambles / signatures / boilerplate
    ("CONSIDERANDO que es necesario regular el uso del agua.", "SKIP"),
    ("Dado en Madrid, a 5 de marzo de 1984.", "SKIP"),
    ("Firmado por los miembros fundadores.", "SKIP"),
    ("IN WITNESS WHEREOF the parties have signed.", "SKIP"),
    # regression bug 1: signature line vs normative sentence
    ("El Presidente", "SKIP"),
    ("El Presidente,", "SKIP"),
    ("El Presidente convocará las reuniones ordinarias y extraordinarias.", "RELEVANT"),
    ("La Secretaria", "SKIP"),
    ("El Secretario llevará el libro de actas de la Comunidad.", "RELEVANT"),
    # regression bug 2: keyword heading vs normative sentence
    ("MEMBERSHIP", "METADATA"),
    ("MEETINGS", "METADATA"),
    ("Membership is open to all households in the community.", "RELEVANT"),
    ("Meetings shall be held quarterly at the community hall.", "RELEVANT"),
    # headings
    ("Artículo 5. De las obligaciones de los socios.", "METADATA"),
    ("Capítulo II", "METADATA"),
    ("ARTICLE IV", "METADATA"),
    ("Anexo I", "METADATA"),
    # plain normative sentence
    ("Los socios deberán pagar la cuota mensual.", "RELEVANT"),
])
def test_classify_block_golden(text, expected):
    assert classify_block(text) == expected


# ── segment_type assignment (segmenter) ──────────────────────────────────────

_GOLDEN_DOC = """\
CONSIDERANDO que es necesario regular el aprovechamiento de las aguas.

Capítulo I. Disposiciones generales

Artículo 1. La Comunidad de Regantes administrará las aguas del canal.

Artículo 2. Son obligaciones de los socios:

- Pagar la cuota anual establecida por la Asamblea.
- Asistir a las faenas comunitarias.
- Dignidad.

El Presidente convocará las reuniones ordinarias.

Anexo I. Tarifas vigentes.

Dado en Rodeo, a 12 de mayo de 1995.

El Presidente
"""


def test_golden_document_structure(tmp_path):
    segs = _segment_text_doc(tmp_path, _GOLDEN_DOC)

    preamble = _by_text(segs, "CONSIDERANDO")
    assert preamble.segment_type == "preamble"
    assert preamble.is_substantive is False

    chapter = _by_text(segs, "Capítulo I")
    assert chapter.segment_type == "heading"

    art1_body = _by_text(segs, "administrará las aguas")
    assert art1_body.segment_type == "body_sentence"
    assert art1_body.is_substantive is True
    # article reference propagated from the heading
    assert art1_body.article == "1"

    header = _by_text(segs, "obligaciones de los socios")
    assert header.segment_type in ("list_header", "body_sentence")

    item1 = _by_text(segs, "Pagar la cuota anual")
    assert item1.segment_type == "list_item"
    assert item1.is_list_item is True
    assert item1.parent_segment_id is not None

    # short bullet is preserved, not dropped as junk
    short = _by_text(segs, "Dignidad")
    assert short.segment_type == "list_item"

    # regression bug 1 both ways inside a real document
    normative = _by_text(segs, "convocará las reuniones")
    assert normative.segment_type == "body_sentence"
    assert normative.is_substantive is True
    signature_alone = [s for s in segs
                       if s.segment_text.strip() == "El Presidente"]
    assert signature_alone and signature_alone[0].segment_type == "signature"

    annex = _by_text(segs, "Anexo I")
    assert annex.segment_type == "annex"
    assert annex.is_substantive is False

    dado = _by_text(segs, "Dado en Rodeo")
    assert dado.segment_type == "signature"
    assert dado.is_substantive is False


_GOLDEN_DOC_EN = """\
ARTICLE III

MEMBERSHIP

Membership is open to all households in the barangay.

Members shall pay their dues before the annual assembly.
"""


def test_golden_membership_heading_vs_sentence(tmp_path):
    segs = _segment_text_doc(tmp_path, _GOLDEN_DOC_EN, language="en")

    heading = [s for s in segs if s.segment_text.strip() == "MEMBERSHIP"]
    assert heading and heading[0].segment_type == "heading"
    assert heading[0].is_substantive is False

    sentence = _by_text(segs, "Membership is open")
    assert sentence.segment_type == "body_sentence"
    assert sentence.is_substantive is True


# ── review flags (paso1_output) ──────────────────────────────────────────────

def _seg(text: str, segment_type: str = "body_sentence",
         is_list_item: bool = False, parent: str | None = None) -> Segment:
    return Segment(
        segment_id="seg_0001", parent_segment_id=parent, segment_order=1,
        segment_type=segment_type, is_list_item=is_list_item,
        is_substantive=segment_type in ("body_sentence", "list_header", "list_item"),
        segment_text=text,
    )


def _main(count=None, conf=None, has=False):
    return {"aim_n_suggested": count, "aim_candidate_confidence": conf,
            "has_own_aim_candidate": has}


def test_review_flag_short_segment():
    needs, reason = _review_flags(_seg("Dignidad."), _main())
    assert needs and "short_segment" in reason


def test_review_flag_high_aim_count():
    text = "Una oración suficientemente larga para no ser corta de verdad."
    needs, reason = _review_flags(_seg(text), _main(count=3, has=True))
    assert needs and "high_aim_count" in reason


def test_review_flag_possible_bad_split():
    text = "de la Constitución Política que menciona los derechos comunitarios."
    needs, reason = _review_flags(_seg(text), _main())
    assert needs and "possible_bad_split" in reason


def test_review_flag_low_confidence_only_below_threshold():
    from is_identifier.paso1_output import _LOW_CONFIDENCE

    text = "Los socios deberán pagar la cuota anual establecida en asamblea."
    hi, lo = _LOW_CONFIDENCE + 0.05, _LOW_CONFIDENCE - 0.05
    _, reason_hi = _review_flags(_seg(text), _main(count=1, conf=hi, has=True))
    assert "low_confidence" not in reason_hi
    needs, reason_lo = _review_flags(_seg(text), _main(count=1, conf=lo, has=True))
    assert needs and "low_confidence" in reason_lo


def test_review_flag_list_item_no_parent():
    text = "Asistir a todas las faenas comunitarias del canal principal."
    needs, reason = _review_flags(
        _seg(text, segment_type="list_item", is_list_item=True, parent=None),
        _main())
    assert needs and "list_item_no_parent" in reason


# ── list-structure rules (2026-06-10 polish fixes) ──────────────────────────

def test_list_header_adopts_items_across_blocks(tmp_path):
    # blank line between header and items (separate markdown blocks) must not
    # orphan the items — list state persists across blocks
    doc = (
        "Son funciones del pasante las siguientes:\n\n"
        "- a) Acceder a ser autoridad previo sorteo de la comunidad.\n"
        "- b) Permanecer en toda su gestión dentro de la comunidad.\n"
    )
    segs = _segment_text_doc(tmp_path, doc)
    header = _by_text(segs, "funciones del pasante")
    item_a = _by_text(segs, "Acceder a ser autoridad")
    item_b = _by_text(segs, "Permanecer en toda su")
    assert header.segment_type == "list_header"
    assert item_a.parent_segment_id == header.segment_id
    assert item_b.parent_segment_id == header.segment_id


def test_item_ending_in_colon_becomes_subheader(tmp_path):
    doc = (
        "Son obligaciones de la autoridad las siguientes:\n\n"
        "- a) Respetar el territorio de la comunidad en toda su gestión.\n"
        "- b) Su vestimenta será la siguiente:\n"
        "- Poncho, chalina y sombrero según la tradición local.\n"
    )
    segs = _segment_text_doc(tmp_path, doc)
    sub = _by_text(segs, "vestimenta")
    assert sub.segment_type == "list_header"
    # the sub-header hangs from the outer header
    outer = _by_text(segs, "obligaciones de la autoridad")
    assert sub.parent_segment_id == outer.segment_id
    # and its own item hangs from it
    inner = _by_text(segs, "Poncho, chalina")
    assert inner.segment_type == "list_item"
    assert inner.parent_segment_id == sub.segment_id


def test_long_numbered_paragraphs_are_body_not_orphan_items(tmp_path):
    # legal sub-clause style (case 13): "1) <multi-sentence paragraph>" with
    # no list header in scope is body text, split into sentences
    doc = (
        "## § 3 RIGHTS AND OBLIGATIONS\n\n"
        "1)  Each Member shall be entitled to exercise the use to the extent "
        "of its shareholding. The member shall participate in the "
        "administration as provided in these Articles.\n\n"
        "2)  Members are obliged to:\n"
        "    a) comply with the rules on the exercise of uses;\n"
        "    b) comply with the orders of the chairman;\n"
    )
    segs = _segment_text_doc(tmp_path, doc, language="en")
    clause1 = _by_text(segs, "entitled to exercise")
    assert clause1.segment_type == "body_sentence"
    # the multi-sentence paragraph was split
    second = _by_text(segs, "participate in the administration")
    assert second.segment_type == "body_sentence"
    # "2) ... obliged to:" introduces a real list
    header = _by_text(segs, "obliged to")
    assert header.segment_type == "list_header"
    item = _by_text(segs, "comply with the rules")
    assert item.segment_type == "list_item"
    assert item.parent_segment_id == header.segment_id


def test_marker_family_change_closes_list(tmp_path):
    # after items a)..b), a long clause "3)" belongs to the article, not the list
    doc = (
        "2) Members are obliged to:\n"
        "    a) comply with the rules on the exercise of the uses granted;\n"
        "    b) comply with the orders of the chairman at all assemblies;\n"
        "3)  Each eligible adult member is obliged to accept the election to "
        "the position of chairman of the agricultural community.\n"
    )
    segs = _segment_text_doc(tmp_path, doc, language="en")
    clause3 = _by_text(segs, "accept the election")
    assert clause3.segment_type == "body_sentence"


def test_metadata_chunk_with_trailing_body_is_not_swallowed(tmp_path):
    # split_page_block cuts at mid-sentence references ("Section 15(6))...");
    # the normative text after the reference line must come back as body,
    # not be silently absorbed into a non-substantive heading
    chunk = (
        "Section 15(6)).\n"
        "The result of the audit shall be recorded in writing and submitted "
        "to the committee of the agricultural community. "
        "The chairman shall notify all members of the outcome within thirty days.\n"
    )
    doc = "## § 19 AUDIT\n\n" + chunk
    segs = _segment_text_doc(tmp_path, doc, language="en")
    body = _by_text(segs, "recorded in writing")
    assert body.segment_type == "body_sentence"
    assert body.is_substantive is True
    notify = _by_text(segs, "notify all members")
    assert notify.segment_type == "body_sentence"


def test_html_audit_comments_are_not_segments(tmp_path):
    # markdown sources carry embedded audit annotations; they are not text
    doc = (
        "Artículo 1. Los socios deberán pagar la cuota anual.\n\n"
        '<!-- AUDIT:MOVE_TO=preamble;reason="fix: mover a preamble" -->\n\n'
        "Los usuarios cuidarán el canal principal de la comunidad.\n"
    )
    segs = _segment_text_doc(tmp_path, doc)
    assert not any("AUDIT" in s.segment_text for s in segs)
    assert _by_text(segs, "cuidarán el canal").segment_type == "body_sentence"


def test_pipeline_flags_orphan_list_item(tmp_path):
    # a list item with no preceding header/body gets parent=None + the flag
    doc = tmp_path / "orphan.md"
    doc.write_text("- Asistir a las faenas comunitarias del canal.\n",
                   encoding="utf-8")
    df, _ = pipeline_paso1(str(doc), aim_model=None, language="es", case_id=99)
    items = df[df["is_list_item"]]
    assert len(items) == 1
    row = items.iloc[0]
    assert row["parent_segment_id"] is None
    assert row["needs_review"]
    assert "list_item_no_parent" in row["review_reason"]
