from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .aim_extractor import extract_aims
from .document_processor import process_document
from .sentence_splitter import split_sentences
from .structure_detector import (
    classify_block,
    extract_article_ref,
    split_inline_article,
    split_page_block,
)


def pipeline(
    path: str | Path,
    language: str = "es",
    aim_model=None,
    predict_mode: str = "model_1_0",
) -> pd.DataFrame:
    """
    Process a regulatory document into sentence-level institutional statements.

    A trained model is required for public inference. Supported input files are
    PDF, DOCX, TXT, and Markdown.
    """
    rows = []
    n = 1

    blocks = process_document(path)
    current_meta = {"title": None, "chapter": None, "article": None}
    pending: list[dict] = []

    for block in blocks:
        raw_text = block["text"].strip()
        page = block.get("page")
        if not raw_text:
            continue

        sub_blocks = split_page_block(raw_text) if len(raw_text) > 400 else [raw_text]

        for text in sub_blocks:
            text = text.strip()
            if not text:
                continue

            block_type = classify_block(text)
            if block_type == "SKIP":
                continue

            if block_type == "METADATA":
                ref = extract_article_ref(text)
                current_meta.update({k: v for k, v in ref.items() if v is not None})
                _, body = split_inline_article(text)
                if not body:
                    continue
                text = body
            else:
                _, body = split_inline_article(text)
                if body:
                    ref = extract_article_ref(text)
                    current_meta.update({k: v for k, v in ref.items() if v is not None})
                    text = body

            for sentence in split_sentences(text, language=language):
                sentence = sentence.strip()
                if not sentence:
                    continue
                pending.append(
                    {
                        "n": n,
                        "sentence": sentence,
                        "page": page,
                        "title": current_meta["title"],
                        "chapter": current_meta["chapter"],
                        "article": current_meta["article"],
                    }
                )
                n += 1

    if aim_model is None:
        raise ValueError("A trained IS Identifier model is required for inference.")

    if predict_mode not in {"model_1_0", "v1", "v1_0"}:
        raise ValueError("predict_mode must be one of: model_1_0, v1, v1_0")

    sentences_text = [r["sentence"] for r in pending]
    model_preds = aim_model.predict_batch(sentences_text) if sentences_text else []
    for row, pred in zip(pending, model_preds):
        row["aim_n"] = pred.count
        row["is_institutional_statement"] = pred.count > 0
        row["aim_source"] = "model_1_0"
        row["aim_spans"] = pred.spans
        row["aim_spans_text"] = " | ".join(span[2] for span in pred.spans)
        row["aim_spans_json"] = json.dumps(pred.spans, ensure_ascii=False)
        row["aim_ordinal_count"] = pred.ordinal_count
        row["aim_n_bucket"] = pred.aim_n_bucket
        row["aim_confidence"] = pred.confidence

    for row in pending:
        aims = extract_aims(row["sentence"], language=language) if row["aim_n"] > 0 else []
        row["verbs"] = ", ".join(a["lemma"] for a in aims) if aims else ""

    rows.extend(pending)
    columns = [
        "n",
        "page",
        "title",
        "chapter",
        "article",
        "sentence",
        "is_institutional_statement",
        "aim_n",
        "aim_n_bucket",
        "aim_source",
        "aim_confidence",
        "aim_ordinal_count",
        "aim_spans_text",
        "aim_spans_json",
        "verbs",
    ]
    return pd.DataFrame(rows, columns=columns)
