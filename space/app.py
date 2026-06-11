from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr

from is_identifier.cli import run_file
from is_identifier.model_v1 import ISIdentifierModel

MODEL_ID = os.environ.get("IS_IDENTIFIER_MODEL", "bravo-pena/is-identifier-1.0")

_MODEL: ISIdentifierModel | None = None


def get_model() -> ISIdentifierModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = ISIdentifierModel.from_pretrained(MODEL_ID)
    return _MODEL


def identify(file_obj):
    if file_obj is None:
        raise gr.Error("Upload a PDF, DOCX, Markdown, or TXT file first.")

    input_path = Path(file_obj.name)
    suffix = input_path.suffix.lower()
    if suffix not in {".pdf", ".docx", ".md", ".txt"}:
        raise gr.Error("Supported files: .pdf, .docx, .md, .txt")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / f"{input_path.stem}_paso1.xlsx"
        written = run_file(
            input_path=input_path,
            output_path=output_path,
            model=get_model(),
            language="auto",
            write_technical=False,
        )
        final_path = Path(tempfile.gettempdir()) / written.name
        final_path.write_bytes(written.read_bytes())
        return str(final_path)


with gr.Blocks(title="IS Identifier — Paso 1") as demo:
    gr.Markdown("# IS Identifier — Paso 1 (beta)")
    gr.Markdown(
        "Upload a regulatory document (Spanish or English — detected "
        "automatically) and download a reviewable Excel: structure-aware "
        "segments, substantive-context filter, institutional-statement "
        "**candidates** and review flags.\n\n"
        "Sheets: `segments` (one row per segment), `schema` (column "
        "descriptions), `summary` (counts). Rows highlighted in amber carry "
        "`needs_review` flags. Paso 1 suggests candidates — it does **not** "
        "produce the final taxonomic coding (TYPE/TAXON/LINK belong to Paso 2)."
    )
    file_input = gr.File(
        label="Document (.pdf, .docx, .md, .txt)",
        file_types=[".pdf", ".docx", ".md", ".txt"],
    )
    run_button = gr.Button("Run Paso 1", variant="primary")
    output_file = gr.File(label="Excel result (Paso 1)")
    run_button.click(identify, inputs=[file_input], outputs=output_file)


if __name__ == "__main__":
    demo.launch()
