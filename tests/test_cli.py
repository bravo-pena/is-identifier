from types import SimpleNamespace

import pandas as pd

from is_identifier import PASO1_COLUMNS
from is_identifier.cli import run_file


class DummyModel:
    def predict(self, sentence):
        return SimpleNamespace(
            count=1,
            spans=[(0, min(17, len(sentence)), sentence[:17])],
            confidence=0.88,
        )


def test_run_file_writes_paso1_excel(tmp_path):
    input_path = tmp_path / "regulation.md"
    output_path = tmp_path / "result.xlsx"
    input_path.write_text("The members shall pay the annual fee.", encoding="utf-8")

    written = run_file(input_path, output_path, DummyModel(), language="en")

    assert written == output_path
    segments = pd.read_excel(output_path, sheet_name="segments")
    assert list(segments.columns) == PASO1_COLUMNS
    schema = pd.read_excel(output_path, sheet_name="schema")
    assert set(schema["column"]) == set(PASO1_COLUMNS)
    summary = pd.read_excel(output_path, sheet_name="summary")
    assert list(summary.columns) == ["section", "key", "value"]
    # technical sidecar written by default
    assert output_path.with_name(output_path.stem + "_technical.json").exists()
