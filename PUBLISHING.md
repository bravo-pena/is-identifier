# Publishing Checklist — release 1.1 (Paso 1)

This repository is for public inference code only.

## Release 1.1 summary (2026-06-10)

**What changed vs 1.0** (model weights `is_identifier_1_0` unchanged):

- NEW `src/is_identifier/segmenter.py` — structure-aware, list-preserving segmenter.
- NEW `src/is_identifier/paso1_output.py` — Paso 1 schema (21 columns) +
  Excel writer (sheets `segments`/`schema`/`summary`, review highlighting).
- REPLACED the sentence-level pipeline: `pipeline()` (15-column Excel) was
  removed; `pipeline_paso1()` + `write_paso1_excel()` are the only output path.
- UPDATED `cli.py` and `space/app.py` to produce the Paso 1 Excel.
- UPDATED `structure_detector.py` — 2 regression fixes (standalone signature
  lines like "El Presidente"; English keyword headings like "MEMBERSHIP" no
  longer swallow normative sentences).
- UPDATED `pyproject.toml` — version 1.1.0; explicit `click` dependency
  (spacy 3.8 imports it but typer>=0.16 stopped pulling it); regenerated
  `uv.lock`.
- TESTS: 55 fast tests (segmenter goldens + Paso 1 contract + CLI) and
  `scripts/smoke_test.py` (loads the real weights, checks the tokenizer is
  not degenerate, runs a synthetic document end-to-end, verifies the contract).

**Verification commands** (all green on 2026-06-10):

```bash
pytest tests/ -q                  # 55 passed
python scripts/smoke_test.py      # SMOKE TEST OK (vocab 119547, 0 [UNK])
```

**Minimal example:**

```bash
is-identifier regulation.pdf --model models/is_identifier_1_0 --language es
# -> regulation_paso1.xlsx (+ regulation_paso1_technical.json)
```

**Risks / known limitations:**

- Output schema changed (breaking vs 1.0): one row per SEGMENT with 21
  contract columns; consumers of the old 15-column format must migrate.
- `needs_review` rate is ~15–25% by design (bottom confidence quintile +
  structural flags); documented in README.
- Deeply nested lists without ':' headers can leave orphan items — they are
  flagged (`list_item_no_parent`), not silently fixed.
- Excel files produced by pre-1.1 tooling (before the tokenizer fix of
  2026-06-10) must not be used to evaluate AIM candidates.

**Rollback:** `git checkout 7d6ca72` (Initial public release 1.0). The 1.1
changes live in a single local commit; nothing is pushed without explicit
confirmation.

**Pending before remote publication (requires explicit confirmation):**

1. `git push` to `github.com/bravo-pena/is-identifier`.
2. Upload `huggingface/README.md` (model card mentions 1.1 usage) to the HF
   model repo `bravo-pena/is-identifier-1.0` (weights unchanged — only the
   card text changes).
3. Update the HF Space with `space/app.py` + `space/requirements.txt`.
4. Optional: tag `v1.1.0`.

## GitHub Repository

Publish:

- `src/`
- `tests/`
- `scripts/smoke_test.py`
- `docs/`
- `huggingface/README.md`
- `space/`
- `README.md`
- `pyproject.toml`, `uv.lock`
- `LICENSE`

Do not publish:

- Training data
- Annotation spreadsheets
- Source regulation corpora
- Row-level audit outputs
- Training scripts that expose private file names or data assumptions
- Local experiment folders

## Hugging Face Model Repository

Publish the trained model files plus the model card:

- `model.safetensors`
- `config.json`
- `tokenizer.json`            # REQUIRED — without it predictions degenerate
- `tokenizer_config.json`
- `training_config.json`, if available
- `README.md` copied from `huggingface/README.md`

The code package expects the model id to be passed as:

```bash
is-identifier input.pdf --model bravo-pena/is-identifier-1.0
```

Hugging Face model repository: `bravo-pena/is-identifier-1.0`.

## Hugging Face Space

Create a separate Space if users need a browser upload flow:

1. Upload a PDF/DOCX/MD/TXT file.
2. Load `bravo-pena/is-identifier-1.0`.
3. Call `is_identifier.pipeline_paso1(...)` + `write_paso1_excel(...)`
   (`space/app.py` already does this via `cli.run_file`).
4. Return the Paso 1 Excel as a download.

The Space should not contain training data or audit exports. It only needs the
published package, the model id, and runtime dependencies.
