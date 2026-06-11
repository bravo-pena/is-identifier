"""
Paso 1 pipeline contract tests (mock model — fast, no weights).
"""

from types import SimpleNamespace

from is_identifier import PASO1_COLUMNS, pipeline_paso1

FORBIDDEN_COLUMNS = {
    "TYPE", "TAXON", "LINK", "LINK.typ", "specificAIM",
    "suggested_link_role", "link_grammar_candidate", "link_aim_candidate",
    "verb_canonical", "verb_tense", "verb_mood",
    "Ini.Coder", "Rev.Coder", "Date.Ini.Code", "Date.Rev",
    "Version.Ini.Code", "Current.Version", "Reliability", "Notes",
    "Excerpt.Original.language", "Excerpt.Tranlsated",
}


class DummyModel:
    """Mimics ISIdentifierModel.predict: .spans / .count / .confidence."""

    def predict(self, sentence):
        return SimpleNamespace(
            count=1,
            spans=[(0, min(20, len(sentence)), sentence[:20])],
            confidence=0.91,
        )


def _write_doc(tmp_path):
    doc = tmp_path / "regulation.md"
    doc.write_text(
        "Artículo 1. Los socios deberán pagar la cuota anual.\n\n"
        "Son obligaciones de los socios:\n\n"
        "- Asistir a las asambleas ordinarias.\n"
        "- Cuidar el canal principal.\n",
        encoding="utf-8",
    )
    return str(doc)


def test_pipeline_paso1_contract(tmp_path):
    df, technical = pipeline_paso1(_write_doc(tmp_path), aim_model=DummyModel(),
                                   language="es")
    assert list(df.columns) == PASO1_COLUMNS
    bad = [c for c in df.columns
           if c in FORBIDDEN_COLUMNS or c.upper().startswith(("TYPE.", "TAXON.", "LINK."))]
    assert bad == []
    assert df["segment_text"].str.len().gt(0).all()
    assert (df["model_version"] == "is-identifier-1.0").all()
    # list structure preserved
    items = df[df["is_list_item"]]
    assert len(items) == 2
    assert items["parent_segment_id"].notna().all()
    # technical sidecar: one entry per segment
    assert set(technical.keys()) == set(df["segment_id"])


def test_pipeline_paso1_heuristic_path(tmp_path):
    df, _ = pipeline_paso1(_write_doc(tmp_path), aim_model=None, language="es")
    assert list(df.columns) == PASO1_COLUMNS
    assert (df["model_version"] == "heuristic_extractor").all()
