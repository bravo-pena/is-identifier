"""
IS Identifier — Paso 1 public beta.

Converts a regulatory document (PDF, DOCX, TXT, Markdown) into a reviewable
Excel of structure-aware segments with institutional-statement (AIM)
candidates:

    from is_identifier import pipeline_paso1, write_paso1_excel
    from is_identifier.model_v1 import ISIdentifierModel

    model = ISIdentifierModel.from_pretrained("bravo-pena/is-identifier-1.0")
    df, technical = pipeline_paso1("regulation.pdf", aim_model=model, language="es")
    write_paso1_excel(df, "regulation_paso1.xlsx", technical)

Paso 1 produces CANDIDATES for human review (segments, document structure,
substantive-context filter, suggested AIM count, review flags). It does NOT
produce the taxonomic classification (TYPE / TAXON / LINK) — that is Paso 2,
a separate future tool.
"""

from .paso1_output import PASO1_COLUMNS, pipeline_paso1, write_paso1_excel

__all__ = ["PASO1_COLUMNS", "pipeline_paso1", "write_paso1_excel"]
