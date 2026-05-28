from types import SimpleNamespace

import pandas as pd

from is_identifier.cli import run_file


class DummyModel:
    def predict_batch(self, sentences):
        return [
            SimpleNamespace(
                count=1,
                spans=[(0, 18, "The members shall")],
                ordinal_count=1,
                aim_n_bucket=1,
                confidence=0.88,
            )
            for _ in sentences
        ]


def test_run_file_writes_excel(tmp_path):
    input_path = tmp_path / "regulation.md"
    output_path = tmp_path / "result.xlsx"
    input_path.write_text("The members shall pay the annual fee.", encoding="utf-8")

    written = run_file(input_path, output_path, DummyModel(), language="en")

    assert written == output_path
    df = pd.read_excel(output_path)
    assert "sentence" in df.columns
    assert "aim_n" in df.columns
    assert int(df.loc[0, "aim_n"]) == 1
