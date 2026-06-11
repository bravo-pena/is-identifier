from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import pipeline_paso1, write_paso1_excel
from .model_v1 import ISIdentifierModel

DEFAULT_MODEL = "bravo-pena/is-identifier-1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="is-identifier",
        description=(
            "Run IS Identifier (Paso 1) on a PDF, DOCX, Markdown, or TXT file. "
            "Produces a reviewable Excel of structure-aware segments with "
            "institutional-statement candidates."
        ),
    )
    parser.add_argument("input", help="Path to .pdf, .docx, .md, or .txt input file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output .xlsx file. Defaults to <input_stem>_paso1.xlsx.",
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
        help="Document language (translate other languages BEFORE running).",
    )
    parser.add_argument(
        "--no-technical",
        action="store_true",
        help="Skip the *_technical.json sidecar (span offsets for debugging).",
    )
    return parser


def run_file(
    input_path: str | Path,
    output_path: str | Path | None,
    model: ISIdentifierModel,
    language: str = "es",
    write_technical: bool = True,
) -> Path:
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_paso1.xlsx")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df, technical = pipeline_paso1(input_path, aim_model=model, language=language)
    write_paso1_excel(df, output_path, technical if write_technical else None)
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = ISIdentifierModel.from_pretrained(args.model)
    output = run_file(
        input_path=args.input,
        output_path=args.output,
        model=model,
        language=args.language,
        write_technical=not args.no_technical,
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
