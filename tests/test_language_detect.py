from pathlib import Path

from is_identifier.language_detect import detect_language
from is_identifier.paso1_output import pipeline_paso1

_ES_TEXT = (
    "Artículo 1. Los miembros de la comunidad deberán pagar la cuota anual "
    "establecida por la Junta. El Secretario llevará un registro de las "
    "actas y será responsable de su custodia."
)
_EN_TEXT = (
    "Article 1. Members of the association shall pay the annual fee "
    "established by the Board. The Secretary shall keep a record of all "
    "proceedings and must report any damage to the committee."
)


def test_detects_spanish():
    assert detect_language(_ES_TEXT) == "es"


def test_detects_english():
    assert detect_language(_EN_TEXT) == "en"


def test_empty_defaults_to_spanish():
    assert detect_language("") == "es"
    assert detect_language("12345 --- ###") == "es"


def test_pipeline_auto_resolves_english(tmp_path):
    doc = tmp_path / "25. test_bylaws.md"
    doc.write_text(
        "# CONSTITUTION\n\n## Article 1\n\n" + _EN_TEXT + "\n",
        encoding="utf-8",
    )
    df, _ = pipeline_paso1(str(doc), aim_model=None, language="auto")
    # English deontics only resolve with the English spaCy model: the
    # heuristic extractor must find 'shall'/'must' triggers, not blanks.
    triggers = ";".join(df["aim_trigger_candidate"].fillna("").tolist()).lower()
    assert "shall" in triggers or "must" in triggers


def test_pipeline_auto_resolves_spanish(tmp_path):
    doc = tmp_path / "7. test_reglamento.md"
    doc.write_text(
        "# REGLAMENTO\n\n## Artículo 1\n\n" + _ES_TEXT + "\n",
        encoding="utf-8",
    )
    df, _ = pipeline_paso1(str(doc), aim_model=None, language="auto")
    triggers = ";".join(df["aim_trigger_candidate"].fillna("").tolist()).lower()
    assert "deber" in triggers or "pagar" in triggers or "llevará" in triggers
