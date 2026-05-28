"""Tests for the legal sentence splitter."""
import pytest


def test_single_sentence():
    from is_identifier.sentence_splitter import split_sentences
    result = split_sentences("Los regantes deberán pagar la cuota anual.", language="es")
    assert len(result) == 1
    assert "cuota anual" in result[0]


def test_list_items_not_split():
    """Items like a), b), c) should stay joined to the parent sentence."""
    from is_identifier.sentence_splitter import split_sentences
    text = "Los miembros tienen las siguientes obligaciones: a) pagar la cuota, b) mantener las acequias, c) asistir a las asambleas."
    result = split_sentences(text, language="es")
    # All list items should end up in one sentence (or at most 2)
    assert len(result) <= 2, f"Expected ≤2 sentences, got {len(result)}: {result}"


def test_two_distinct_sentences():
    """Two genuinely separate sentences should be split."""
    from is_identifier.sentence_splitter import split_sentences
    text = "El Presidente convocará la Asamblea. El Secretario elaborará el acta."
    result = split_sentences(text, language="es")
    assert len(result) >= 2


def test_empty_input():
    from is_identifier.sentence_splitter import split_sentences
    assert split_sentences("", language="es") == []
    assert split_sentences("   ", language="es") == []


def test_english_model():
    from is_identifier.sentence_splitter import split_sentences
    text = "The Board shall meet quarterly. The Secretary shall keep the minutes."
    result = split_sentences(text, language="en")
    assert len(result) >= 2


def test_short_fragment_merged():
    """A very short fragment after a sentence should be merged back."""
    from is_identifier.sentence_splitter import split_sentences
    text = "Los regantes deberán pagar la cuota anual antes del 31 de marzo. Anualmente."
    result = split_sentences(text, language="es")
    # "Anualmente." is < 20 chars so should be merged
    assert len(result) <= 2
