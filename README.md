# IS Identifier 1.1 — Paso 1 (beta)

IS Identifier converts a regulatory document (PDF, Word `.docx`, Markdown, or
TXT) into a **reviewable Excel of structure-aware segments** with
institutional-statement (AIM) **candidates**: document structure (headings,
preambles, signatures, annexes), list structure (headers + items), a
substantive-context filter, suggested AIM counts, and human-review flags.

This public repository is **inference only**. It does not include private
training data, annotation spreadsheets, source corpora, or training scripts.

> **Paso 1 vs Paso 2.** This tool is *Paso 1*: it proposes **candidates** for
> human coders, not final coding. The taxonomic classification — institutional
> `TYPE`, `TAXON`, `LINK` / `LINK.typ` — belongs to *Paso 2*, a separate
> future tool, and never appears in this output.

## What It Produces

One Excel workbook with three sheets:

- **`segments`** — one row per segment, exactly 21 columns:
  - structure: `segment_id`, `parent_segment_id` (list items point to their
    header), `segment_order`, `title`, `chapter`, `article`, `segment_type`
    (`body_sentence`, `list_header`, `list_item`, `heading`, `preamble`,
    `signature`, `annex`, `metadata`), `is_list_item`
  - context filter: `is_substantive` (`False` = documentary context such as
    preambles/signatures — emitted, never silently dropped)
  - candidates: `aim_n_suggested` (suggested count — **not** a final count),
    `aim_text_candidate`, `aim_trigger_candidate`, `aim_candidate_confidence`,
    `has_own_aim_candidate`
  - review: `needs_review`, `review_reason`
  - traceability: `case_id`, `system`, `segment_text`, `model_version`,
    `source_base`
- **`schema`** — a description of every column (self-documenting).
- **`summary`** — counts by `segment_type` / `is_substantive` /
  `needs_review` (+ reasons) and the `aim_n_suggested` distribution.

Rows with `needs_review = True` are highlighted in amber. `review_reason`
explains why: `low_confidence` (bottom ~fifth of model confidence, threshold
0.85), `short_segment`, `possible_bad_split`, `list_item_no_parent`,
`high_aim_count`, `context_filter_conflict`. A `*_technical.json` sidecar
carries span offsets for debugging (never in the Excel).

> **Note for early testers:** Excel files generated before 2026-06-10 (with
> tooling that predates the tokenizer fix) must not be used to evaluate AIM
> candidates — regenerate them with this version.

## Install

```bash
pip install -e ".[dev]"

python -m spacy download es_core_news_lg
python -m spacy download en_core_web_lg
```

## Run From The Command Line

```bash
is-identifier regulation.pdf \
  --model bravo-pena/is-identifier-1.0 \
  --language es \
  --output regulation_paso1.xlsx
```

Using a local model export:

```bash
is-identifier regulation.docx --model models/is_identifier_1_0 --language en
```

You can also set the model once:

```bash
set IS_IDENTIFIER_MODEL=bravo-pena/is-identifier-1.0
is-identifier regulation.md
```

## Use From Python

```python
from is_identifier import pipeline_paso1, write_paso1_excel
from is_identifier.model_v1 import ISIdentifierModel

model = ISIdentifierModel.from_pretrained("bravo-pena/is-identifier-1.0")
df, technical = pipeline_paso1("regulation.pdf", aim_model=model, language="es")
write_paso1_excel(df, "regulation_paso1.xlsx", technical)
```

## Supported Inputs

- PDF: `.pdf` — Word: `.docx` — Markdown: `.md` — Plain text: `.txt`
- Spanish and English. **Translate other languages BEFORE running Paso 1.**
- Legacy `.doc` files should be converted to `.docx` first.

## Model

Architecture: multilingual DistilBERT with a BIO span head (CRF-decoded) and
an auxiliary ordinal count head. `aim_n_suggested` is derived from decoded
spans. Model weights: `is-identifier-1.0` (unchanged in this release; 1.1
updates the pipeline around it).

Validation summary for model 1.0:

| Metric | Value | Target | Status |
| --- | ---: | ---: | --- |
| `count_macro_f1` | 0.5345 | 0.520 | PASS |
| `span_f1_partial` | 0.6723 | 0.650 | PASS |
| `recall_aim0` | 0.5739 | 0.550 | PASS |
| `recall_aim_ge1` | 0.9225 | 0.850 | PASS |

The validation dataset is private and is not included in this repository.

## Repository Contents

```text
src/is_identifier/
    cli.py                # Command-line interface: document -> Paso 1 Excel
    model_v1.py           # Public model loader and inference wrapper
    modeling.py           # Neural architecture used by the published weights
    segmenter.py          # Structure-aware, list-preserving segmenter
    paso1_output.py       # Paso 1 schema (21 columns) + Excel writer
    document_processor.py # PDF/DOCX/TXT/Markdown extraction
    sentence_splitter.py  # Legal/regulatory sentence splitting
    structure_detector.py # Article/chapter metadata detection
    aim_extractor.py      # Verb extraction for candidate triggers
    verb_lexicon.py       # Institutional verb lexicon

huggingface/README.md     # Model card for the Hugging Face model repository
space/app.py              # Hugging Face Space upload interface
scripts/smoke_test.py     # Production smoke test (real weights)
```

## Tests

```bash
pytest tests/ -q                  # fast suite (no model weights)
python scripts/smoke_test.py      # production smoke test (loads real model)
```

## Confidentiality

Training data, review samples, annotation workbooks, source regulations, and
intermediate audit outputs are intentionally excluded. Publish model weights in
the Hugging Face model repository, and publish this code repository separately.
