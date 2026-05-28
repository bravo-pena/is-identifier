from types import SimpleNamespace


class DummyModel:
    def __init__(self):
        self.sentences = None

    def predict_batch(self, sentences):
        self.sentences = sentences
        return [
            SimpleNamespace(
                count=1,
                spans=[(0, 12, "Members shall")],
                ordinal_count=1,
                aim_n_bucket=1,
                confidence=0.91,
            )
            for _ in sentences
        ]


def _patch_minimal_pipeline(monkeypatch):
    import is_identifier

    monkeypatch.setattr(
        is_identifier,
        "process_document",
        lambda path: [{"text": "Members shall pay the annual fee.", "page": 3}],
    )
    monkeypatch.setattr(is_identifier, "classify_block", lambda text: "RELEVANT")
    monkeypatch.setattr(is_identifier, "split_page_block", lambda text: [text])
    monkeypatch.setattr(is_identifier, "split_inline_article", lambda text: (text, ""))
    monkeypatch.setattr(is_identifier, "split_sentences", lambda text, language="en": [text])
    monkeypatch.setattr(is_identifier, "extract_aims", lambda sentence, language="en": [])
    return is_identifier


def test_pipeline_uses_model_1_0(monkeypatch):
    is_identifier = _patch_minimal_pipeline(monkeypatch)
    model = DummyModel()

    df = is_identifier.pipeline("dummy.md", language="en", aim_model=model)

    assert model.sentences == ["Members shall pay the annual fee."]
    assert df.loc[0, "page"] == 3
    assert df.loc[0, "aim_n"] == 1
    assert bool(df.loc[0, "is_institutional_statement"]) is True
    assert df.loc[0, "aim_source"] == "model_1_0"
    assert df.loc[0, "aim_spans_text"] == "Members shall"


def test_pipeline_requires_trained_model(monkeypatch):
    is_identifier = _patch_minimal_pipeline(monkeypatch)

    try:
        is_identifier.pipeline("dummy.md", language="en", aim_model=None)
    except ValueError as exc:
        assert "trained IS Identifier model" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
