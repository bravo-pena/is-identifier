from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from . import pipeline
from .model_v1 import ISIdentifierModel

DEFAULT_MODEL = "bravo-pena/is-identifier-1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="is-identifier",
        description="Run IS Identifier 1.0 on a PDF, DOCX, Markdown, or TXT file.",
    )
    parser.add_argument("input", help="Path to .pdf, .docx, .md, or .txt input file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output .xlsx file. Defaults to <input_stem>_is_identifier.xlsx.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=os.environ.get("IS_IDENTIFIER_MODEL", DEFAULT_MODEL),
        help="Hugging Face model id or local model directory.",
    )
    parser.add_argument(
        "-l",
        "--language",
        choices=["es", "en"],
        default="es",
        help="Sentence-splitting language.",
    )
    parser.add_argument(
        "--sheet-name",
        default="institutional_statements",
        help="Excel sheet name.",
    )
    return parser


def run_file(
    input_path: str | Path,
    output_path: str | Path | None,
    model: ISIdentifierModel,
    language: str = "es",
    sheet_name: str = "institutional_statements",
) -> Path:
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_is_identifier.xlsx")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pipeline(input_path, language=language, aim_model=model, predict_mode="model_1_0")
    _write_excel(df, output_path, sheet_name=sheet_name)
    return output_path


def _write_excel(df: pd.DataFrame, output_path: Path, sheet_name: str) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        worksheet.freeze_panes = "A2"
        for column_cells in worksheet.columns:
            header = str(column_cells[0].value or "")
            width = min(max(len(header) + 2, 12), 80)
            worksheet.column_dimensions[column_cells[0].column_letter].width = width


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = ISIdentifierModel.from_pretrained(args.model)
    output = run_file(
        input_path=args.input,
        output_path=args.output,
        model=model,
        language=args.language,
        sheet_name=args.sheet_name,
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
