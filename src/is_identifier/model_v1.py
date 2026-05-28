"""
Production wrapper for the IS Identifier 1.0 AIM model.

Primary inference path: count derived from BIO span head (CRF-decoded).
Ordinal head count is included as a sanity check only; it is NOT used for
production output.

Usage::

    model = ISIdentifierModel.from_pretrained("models/is_identifier_1_0")
    pred = model.predict("Los regantes deberán pagar la cuota anual.")
    print(pred.count, pred.spans)

The production model directory must contain:
- pytorch_model.bin (or model.safetensors)
- config.json          (saved by AimMultiTaskModel.save_pretrained)
- tokenizer.json + tokenizer_config.json  (saved by tokenizer.save_pretrained)
- training_config.json (records training hyper-parameters, when available)

If the tokenizer files are absent from the model directory, the wrapper falls
back to loading from the backbone (distilbert-base-multilingual-cased).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import DistilBertTokenizerFast

from .modeling import AimMultiTaskModel

_MAX_LEN = 256
_BACKBONE = "distilbert-base-multilingual-cased"
_DEFAULT_MODEL_REF = "bravo-pena/is-identifier-1.0"


@dataclass
class ISPrediction:
    """Prediction output from ISIdentifierModel."""

    spans: List[Tuple[int, int, str]]
    """Character-level AIM spans: (start_char, end_char, text_fragment)."""

    count: int
    """Number of AIMs detected by the span head (primary metric)."""

    ordinal_count: int
    """AIM count from the ordinal head (sanity check; not used for output)."""

    aim_n_bucket: int
    """min(count, 3), capped for the published categorical output."""

    confidence: Optional[float] = None
    """Mean softmax probability of the predicted BIO label over non-special
    tokens.  Higher = model is more certain about its token-level predictions.
    Returns None if no non-special tokens were present."""


class ISIdentifierModel:
    """Production inference pipeline for the IS Identifier 1.0 AIM model."""

    def __init__(
        self,
        model: AimMultiTaskModel,
        tokenizer: DistilBertTokenizerFast,
        device: torch.device,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._model.eval()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls, model_dir: str = _DEFAULT_MODEL_REF
    ) -> "ISIdentifierModel":
        """Load the production model from a local directory or HF Hub id.

        Hugging Face model ids are passed through to Transformers. Local
        directories are supported for offline deployments.
        """
        model_ref, local_path = _resolve_model_ref(model_dir)

        use_crf = True
        ordinal_weight = 0.3
        aim0_penalty_weight = 0.5
        config_path = local_path / "training_config.json" if local_path else None
        if config_path is not None and config_path.exists():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            use_crf = cfg.get("use_crf", use_crf)
            ordinal_weight = cfg.get("ordinal_weight", ordinal_weight)
            aim0_penalty_weight = cfg.get("aim0_penalty_weight", aim0_penalty_weight)

        # Tokenizer: prefer saved copy, fall back to HF hub
        try:
            tokenizer = DistilBertTokenizerFast.from_pretrained(model_ref)
        except OSError:
            warnings.warn(
                f"Tokenizer files not found in '{model_ref}'. "
                f"Loading from '{_BACKBONE}' instead.  "
                "Save the tokenizer alongside the model to avoid this.",
                RuntimeWarning,
                stacklevel=2,
            )
            tokenizer = DistilBertTokenizerFast.from_pretrained(_BACKBONE)

        model = AimMultiTaskModel.from_pretrained(
            model_ref,
            use_crf=use_crf,
            ordinal_weight=ordinal_weight,
            aim0_penalty_weight=aim0_penalty_weight,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return cls(model, tokenizer, device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, sentence: str) -> ISPrediction:
        """Predict AIM spans and count for a single sentence."""
        return self.predict_batch([sentence])[0]

    def predict_batch(self, sentences: List[str]) -> List[ISPrediction]:
        """Predict AIM spans and counts for a batch of sentences."""
        encoding = self._tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=_MAX_LEN,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        # offset_mapping must be popped because it is not a model input.
        offset_mapping = encoding.pop("offset_mapping")  # [batch, seq_len, 2]

        input_ids = encoding["input_ids"].to(self._device)
        attention_mask = encoding["attention_mask"].to(self._device)

        with torch.no_grad():
            out = self._model(input_ids=input_ids, attention_mask=attention_mask)
            decoded_bio = self._model.decode_spans(out["span_logits"], attention_mask)
            softmax_probs = torch.softmax(out["span_logits"], dim=-1).cpu()  # [B, seq, 3]

        predictions: List[ISPrediction] = []
        for i, (sentence_text, bio_seq) in enumerate(zip(sentences, decoded_bio)):
            offsets: list[list[int]] = offset_mapping[i].tolist()
            mask_row: list[int] = attention_mask[i].cpu().tolist()

            # Map BIO tokens to character-level spans
            spans: List[Tuple[int, int, str]] = []
            span_start_char: Optional[int] = None
            span_end_char: int = 0
            confidence_probs: List[float] = []

            for tok_idx, tag in enumerate(bio_seq):
                if not mask_row[tok_idx]:
                    break
                start_char, end_char = offsets[tok_idx]
                is_special = (start_char == 0 and end_char == 0)

                if is_special:
                    if span_start_char is not None:
                        text_frag = sentence_text[span_start_char:span_end_char].strip()
                        if text_frag:
                            spans.append((span_start_char, span_end_char, text_frag))
                        span_start_char = None
                    continue

                confidence_probs.append(softmax_probs[i, tok_idx, tag].item())

                if tag == 1:        # B-AIM
                    if span_start_char is not None:
                        text_frag = sentence_text[span_start_char:span_end_char].strip()
                        if text_frag:
                            spans.append((span_start_char, span_end_char, text_frag))
                    span_start_char = start_char
                    span_end_char = end_char
                elif tag == 2:      # I-AIM
                    if span_start_char is None:
                        span_start_char = start_char
                    span_end_char = end_char
                else:               # O
                    if span_start_char is not None:
                        text_frag = sentence_text[span_start_char:span_end_char].strip()
                        if text_frag:
                            spans.append((span_start_char, span_end_char, text_frag))
                        span_start_char = None

            # Close any span still open at end of sequence
            if span_start_char is not None:
                text_frag = sentence_text[span_start_char:span_end_char].strip()
                if text_frag:
                    spans.append((span_start_char, span_end_char, text_frag))

            count = len(spans)
            ordinal_count = AimMultiTaskModel.predict_count_from_ordinal(
                out["ordinal_logits"][i].cpu()
            )
            confidence = (
                float(sum(confidence_probs) / len(confidence_probs))
                if confidence_probs else None
            )

            predictions.append(ISPrediction(
                spans=spans,
                count=count,
                ordinal_count=ordinal_count,
                aim_n_bucket=min(count, 3),
                confidence=confidence,
            ))

        return predictions


def _resolve_model_ref(model_dir: str) -> tuple[str, Path | None]:
    """Resolve local model directories while allowing HF Hub ids."""
    path = Path(model_dir)
    if path.exists():
        candidate = path
        full_child = candidate / "full"
        if not _has_model_weights(candidate) and _has_model_weights(full_child):
            candidate = full_child
        if not _has_model_weights(candidate):
            raise FileNotFoundError(
                f"Model directory '{candidate}' does not contain model weights. "
                "Expected model.safetensors or pytorch_model.bin."
            )
        return str(candidate), candidate

    if path.is_absolute() or model_dir.startswith((".", "models/", "models\\")):
        raise FileNotFoundError(
            f"Model directory '{path}' not found. "
            "Pass a local model export directory or a Hugging Face model id."
        )

    return model_dir, None


def _has_model_weights(path: Path) -> bool:
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
