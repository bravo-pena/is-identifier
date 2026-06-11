"""
Production smoke test — run before any release.

Loads the REAL model (local dir or HF id), checks the tokenizer is not
degenerate, runs Paso 1 end-to-end on a small synthetic regulation, writes
the Excel and verifies the output contract. Not part of pytest (downloads /
loads ~500 MB of weights).

Usage:
    python scripts/smoke_test.py [model_dir_or_hf_id]   # default: models/is_identifier_1_0
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from is_identifier import PASO1_COLUMNS, pipeline_paso1, write_paso1_excel
from is_identifier.model_v1 import ISIdentifierModel

FORBIDDEN = {"TYPE", "TAXON", "LINK", "LINK.typ", "specificAIM",
             "suggested_link_role", "verb_canonical", "Ini.Coder", "Rev.Coder",
             "Reliability", "Notes", "Excerpt.Original.language"}

# Synthetic regulation fragment (no private data)
_DOC = """\
CONSIDERANDO que es necesario regular el aprovechamiento de las aguas.

Artículo 1. La Comunidad de Regantes administrará las aguas del canal principal.

Artículo 2. Son obligaciones de los socios:

- Pagar la cuota anual establecida por la Asamblea.
- Asistir a las faenas comunitarias de limpieza.

El Presidente convocará las reuniones ordinarias una vez al año.

Dado en la ciudad, a 1 de enero de 2026.
"""


def main() -> int:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "models/is_identifier_1_0"
    failures: list[str] = []

    print(f"[1/5] Loading model + tokenizer: {model_id}")
    model = ISIdentifierModel.from_pretrained(model_id)

    print("[2/5] Tokenizer sanity")
    vocab = model._tokenizer.vocab_size
    if vocab <= 1000:
        failures.append(f"DEGENERATE TOKENIZER: vocab_size={vocab}")
    ids = model._tokenizer("Los regantes deberán pagar la cuota.")["input_ids"]
    n_unk = sum(1 for i in ids if i == model._tokenizer.unk_token_id)
    if n_unk > 0:
        failures.append(f"tokenizer produced {n_unk} [UNK] on plain Spanish")
    print(f"      vocab_size={vocab}, unk_in_probe={n_unk}")

    print("[3/5] Paso 1 end-to-end on synthetic regulation")
    with tempfile.TemporaryDirectory() as tmp:
        doc = Path(tmp) / "smoke_regulation.md"
        doc.write_text(_DOC, encoding="utf-8")
        df, technical = pipeline_paso1(str(doc), aim_model=model, language="es")
        out = Path(tmp) / "smoke_paso1.xlsx"
        write_paso1_excel(df, out, technical)

        print("[4/5] Output contract")
        if list(df.columns) != PASO1_COLUMNS:
            failures.append(f"columns != PASO1_COLUMNS: {list(df.columns)}")
        bad = [c for c in df.columns if c in FORBIDDEN
               or c.upper().startswith(("TYPE.", "TAXON.", "LINK."))]
        if bad:
            failures.append(f"forbidden columns present: {bad}")
        for sheet in ("segments", "schema", "summary"):
            try:
                pd.read_excel(out, sheet_name=sheet)
            except Exception as exc:
                failures.append(f"sheet '{sheet}' unreadable: {exc}")

        print("[5/5] Prediction sanity")
        sub = df[df["is_substantive"]]
        n_cand = int(sub["has_own_aim_candidate"].sum())
        if n_cand == 0:
            failures.append("model produced ZERO candidates on substantive "
                            "segments (degenerate predictions?)")
        if not (df["segment_type"] == "list_item").any():
            failures.append("list structure lost (no list_item segments)")
        preamble_ok = (df.loc[df["segment_text"].str.contains("CONSIDERANDO"),
                              "is_substantive"] == False).all()  # noqa: E712
        if not preamble_ok:
            failures.append("preamble not filtered as non-substantive")
        print(f"      {len(df)} segments | {len(sub)} substantive | "
              f"{n_cand} with candidate | needs_review "
              f"{df['needs_review'].mean():.0%}")

    if failures:
        print("\nSMOKE TEST FAILED:")
        for f in failures:
            print("  - " + f)
        return 1
    print("\nSMOKE TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
