# IS Identifier 1.0

IS Identifier 1.0 identifies institutional statements in regulatory documents.
Given a PDF, Word `.docx`, Markdown, or TXT file, it returns an Excel workbook
with one row per sentence and model predictions such as `aim_n`, AIM spans, and
supporting metadata.

This public repository is **inference only**. It does not include private
training data, annotation spreadsheets, source corpora, or training scripts.

## What It Produces

The Excel output includes:

- `sentence`: sentence-level text extracted from the uploaded document
- `page`, `title`, `chapter`, `article`: structural metadata when detected
- `is_institutional_statement`: `True` when `aim_n > 0`
- `aim_n`: number of AIMs detected in the sentence
- `aim_spans_text` and `aim_spans_json`: model-detected AIM spans
- `aim_confidence`: token-level confidence summary
- `verbs`: institutional verbs found for QA/audit support

The model does **not** assign institutional `TYPE` or `TAXON`.

## Install

```bash
pip install -e ".[dev]"

python -m spacy download es_core_news_lg
python -m spacy download en_core_web_lg
```

## Run From The Command Line

Using a Hugging Face model repository:

```bash
is-identifier regulation.pdf \
  --model bravo-pena/is-identifier-1.0 \
  --language es \
  --output result.xlsx
```

Using a local model export:

```bash
is-identifier regulation.docx \
  --model models/is_identifier_1_0 \
  --language en \
  --output result.xlsx
```

You can also set the model once:

```bash
set IS_IDENTIFIER_MODEL=bravo-pena/is-identifier-1.0
is-identifier regulation.md --output result.xlsx
```

## Browser Upload Workflow

For a public web page where a user uploads a PDF/DOCX/MD/TXT and downloads the
Excel result, publish a Hugging Face Space that calls this package and the
published model. The model repository stores the weights; the Space provides the
file-upload interface.

## Use From Python

```python
from is_identifier import pipeline
from is_identifier.model_v1 import ISIdentifierModel

model = ISIdentifierModel.from_pretrained("bravo-pena/is-identifier-1.0")
df = pipeline("regulation.pdf", language="es", aim_model=model)
df.to_excel("result.xlsx", index=False)
```

## Supported Inputs

- PDF: `.pdf`
- Word: `.docx`
- Markdown: `.md`
- Plain text: `.txt`

Legacy `.doc` files should be converted to `.docx` first.

## Model

Architecture: multilingual DistilBERT with a BIO span head and auxiliary ordinal
count head. Production `aim_n` is derived from decoded AIM spans.

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
    cli.py                # Command-line interface: document -> Excel
    model_v1.py           # Public model loader and inference wrapper
    modeling.py           # Neural architecture used by the published weights
    document_processor.py # PDF/DOCX/TXT/Markdown extraction
    sentence_splitter.py  # Legal/regulatory sentence splitting
    structure_detector.py # Article/chapter metadata detection
    aim_extractor.py      # Verb extraction for QA/audit columns
    verb_lexicon.py       # Institutional verb lexicon

huggingface/
    README.md             # Model card for the Hugging Face model repository

space/
    app.py                # Optional Hugging Face Space upload interface
```

## Tests

```bash
pytest tests/ -q
```

## Confidentiality

Training data, review samples, annotation workbooks, source regulations, and
intermediate audit outputs are intentionally excluded. Publish model weights in
the Hugging Face model repository, and publish this code repository separately.
