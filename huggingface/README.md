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
  - name: IS Identifier 1.2
    results:
      - task:
          type: token-classification
          name: AIM span extraction and AIM.n counting
        dataset:
          name: Private regulatory annotation dataset (June 2026 correction round)
          type: private
        metrics:
          - name: count_macro_f1
            type: f1
            value: 0.556
          - name: span_f1_partial
            type: f1
            value: 0.6925
          - name: recall_aim0
            type: recall
            value: 0.6448
          - name: recall_aim_ge1
            type: recall
            value: 0.9316
---

# IS Identifier 1.2

IS Identifier 1.2 identifies institutional statements in regulatory sentences.
It predicts AIM spans using a BIO token-classification head and derives the
suggested AIM count from the decoded spans.

> This repository keeps its original id (`is-identifier-1.0`) for continuity;
> it hosts the **current model version, 1.2** (see `training_config.json` and
> the validation table below). Earlier revisions remain available in the repo
> history.

The model is intended to be used with the companion Python package
(**version 1.2, Paso 1**). Given a PDF, Word `.docx`, Markdown, or TXT file,
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

Version 1.2 is an interim retrain of the 1.0 recipe on the annotation base
after the June 2026 correction round (verified label fixes from the coding
team; same architecture and hyper-parameters). Figures are provisional: a
further label-review round (double coding + list-article convention) is in
progress and the metrics will be re-frozen with the final base.

Leave-one-regulation-out cross validation, 14 folds:

| Metric | 1.2 (provisional) | 1.0 baseline |
| --- | ---: | ---: |
| `count_macro_f1` | 0.556 | 0.5345 |
| `span_f1_partial` | 0.6925 | 0.6723 |
| `recall_aim0` | 0.6448 | 0.5739 |
| `recall_aim_ge1` | 0.9316 | 0.9225 |

The validation data is private and is not included in this model repository.

## Limitations

- Validated for Spanish and English regulatory-style text.
- Legacy `.doc` files should be converted to `.docx` before processing.
- The model does not assign institutional `TYPE` or `TAXON`.
- This model supports coding and audit workflows; it is not a legal advisor.

## License

MIT.
