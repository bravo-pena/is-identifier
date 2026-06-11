---
language:
  - es
  - en
license: mit
library_name: transformers
base_model: distilbert-base-multilingual-cased
pipeline_tag: token-classification
tags:
  - institutional-statements
  - regulatory-text
  - aim-extraction
  - span-extraction
  - multilingual
model-index:
  - name: IS Identifier 1.0
    results:
      - task:
          type: token-classification
          name: AIM span extraction and AIM.n counting
        dataset:
          name: Private regulatory annotation dataset
          type: private
        metrics:
          - name: count_macro_f1
            type: f1
            value: 0.5345
          - name: span_f1_partial
            type: f1
            value: 0.6723
          - name: recall_aim0
            type: recall
            value: 0.5739
          - name: recall_aim_ge1
            type: recall
            value: 0.9225
---

# IS Identifier 1.0

IS Identifier 1.0 identifies institutional statements in regulatory sentences.
It predicts AIM spans using a BIO token-classification head and derives the
suggested AIM count from the decoded spans.

The model is intended to be used with the companion Python package
(**version 1.1, Paso 1**). Given a PDF, Word `.docx`, Markdown, or TXT file,
the package exports a reviewable Excel of structure-aware segments with AIM
**candidates**, a substantive-context filter and human-review flags
(`needs_review` / `review_reason`). Paso 1 proposes candidates for human
coders — the taxonomic classification (`TYPE` / `TAXON` / `LINK`) belongs to a
separate future tool (Paso 2) and is never part of this output.

## Usage

```python
from is_identifier import pipeline_paso1, write_paso1_excel
from is_identifier.model_v1 import ISIdentifierModel

model = ISIdentifierModel.from_pretrained("bravo-pena/is-identifier-1.0")
df, technical = pipeline_paso1("regulation.pdf", aim_model=model, language="es")
write_paso1_excel(df, "regulation_paso1.xlsx", technical)
```

Command line:

```bash
is-identifier regulation.pdf \
  --model bravo-pena/is-identifier-1.0 \
  --language es \
  --output regulation_paso1.xlsx
```

> Outputs generated with package versions prior to the 2026-06-10 tokenizer
> fix must not be used to evaluate AIM candidates — regenerate them.

## Expected Files In This Model Repository

- `model.safetensors`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `training_config.json`, if available

## Validation

| Metric | Value | Target | Status |
| --- | ---: | ---: | --- |
| `count_macro_f1` | 0.5345 | 0.520 | PASS |
| `span_f1_partial` | 0.6723 | 0.650 | PASS |
| `recall_aim0` | 0.5739 | 0.550 | PASS |
| `recall_aim_ge1` | 0.9225 | 0.850 | PASS |

The validation data is private and is not included in this model repository.

## Limitations

- Validated for Spanish and English regulatory-style text.
- Legacy `.doc` files should be converted to `.docx` before processing.
- The model does not assign institutional `TYPE` or `TAXON`.
- This model supports coding and audit workflows; it is not a legal advisor.

## License

MIT.
