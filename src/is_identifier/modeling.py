"""
IS Identifier 1.0 model architecture: BIO span tagging + ordinal AIM count.

Heads:
  - span_head:    per-token logits over {O, B-AIM, I-AIM}
  - ordinal_head: K-1 binary logits {AIM>=1, AIM>=2, AIM>=3}
  - (optional) CRF on top of span_head

Inference: final count = number of B-AIM tags detected by span_head.
Ordinal head regularizes training and serves as sanity-check.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertPreTrainedModel, DistilBertConfig

try:
    from TorchCRF import CRF  # package is TorchCRF (capital letters)
    HAS_CRF = True
except ImportError:
    try:
        from torchcrf import CRF  # fallback for older install naming
        HAS_CRF = True
    except ImportError:
        HAS_CRF = False


class AimMultiTaskModel(DistilBertPreTrainedModel):
    NUM_BIO_LABELS = 3      # O=0, B-AIM=1, I-AIM=2
    NUM_ORDINAL_LEVELS = 3  # AIM>=1, AIM>=2, AIM>=3

    def __init__(
        self,
        config: DistilBertConfig,
        use_crf: bool = True,
        ordinal_weight: float = 0.3,
        aim0_penalty_weight: float = 0.5,
    ):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.span_head = nn.Linear(config.hidden_size, self.NUM_BIO_LABELS)
        self.ordinal_head = nn.Linear(config.hidden_size, self.NUM_ORDINAL_LEVELS)

        self.use_crf = use_crf and HAS_CRF
        if self.use_crf:
            # TorchCRF API: num_labels (not num_tags), no batch_first arg
            self.crf = CRF(num_labels=self.NUM_BIO_LABELS, use_gpu=False)

        self.ordinal_weight = ordinal_weight
        self.aim0_penalty_weight = aim0_penalty_weight
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bio_labels: torch.Tensor | None = None,
        ordinal_labels: torch.Tensor | None = None,
        aim_n: torch.Tensor | None = None,
    ) -> dict:
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        token_repr = self.dropout(outputs.last_hidden_state)
        cls_repr = token_repr[:, 0, :]

        span_logits = self.span_head(token_repr)
        ordinal_logits = self.ordinal_head(cls_repr)

        loss = None
        if bio_labels is not None and ordinal_labels is not None:
            mask = attention_mask.bool()

            # CRF does not accept -100; replace with 0 and mask those positions out
            crf_mask = mask & (bio_labels >= 0)
            bio_for_loss = bio_labels.clone()
            bio_for_loss[bio_for_loss < 0] = 0

            if self.use_crf:
                # TorchCRF forward(h, labels, mask) returns per-example log-likelihood [B]
                span_loss = -self.crf(span_logits, bio_for_loss, crf_mask).mean()
            else:
                logits_flat = span_logits.view(-1, self.NUM_BIO_LABELS)
                labels_flat = bio_labels.view(-1)
                span_loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)

            # Balanced BCE: weight positive class by n_neg/n_pos per threshold.
            # Without this, threshold-1 collapses to predict all-1 because AIM>=1
            # is ~85-88% of training data (7.66x imbalance ratio).
            n_pos = ordinal_labels.sum(dim=0).clamp(min=1)          # [3]
            n_neg = (ordinal_labels.shape[0] - n_pos).clamp(min=1)  # [3]
            ordinal_pw = (n_neg / n_pos).clamp(max=50.0)            # [3], <1 when pos majority
            ordinal_loss = F.binary_cross_entropy_with_logits(
                ordinal_logits, ordinal_labels, pos_weight=ordinal_pw
            )
            loss = span_loss + self.ordinal_weight * ordinal_loss

            # AIM=0 penalty: push B-AIM/I-AIM logits negative for sentences with no AIMs.
            # Addresses over-prediction of spans in AIM=0 sentences.
            if aim_n is not None and self.aim0_penalty_weight > 0:
                is_aim0 = (aim_n == 0)  # [batch_size] bool
                if is_aim0.any():
                    # span_logits: [batch, seq_len, 3] — cols 1,2 are B-AIM, I-AIM
                    positive_logits = span_logits[is_aim0][:, :, 1:]  # [n_aim0, seq, 2]
                    valid_mask = attention_mask[is_aim0].bool().unsqueeze(-1)  # [n_aim0, seq, 1]
                    n_valid = valid_mask.sum().clamp(min=1)
                    penalty = (positive_logits * valid_mask).sum() / n_valid
                    loss = loss + self.aim0_penalty_weight * penalty

        return {"loss": loss, "span_logits": span_logits, "ordinal_logits": ordinal_logits}

    def decode_spans(
        self, span_logits: torch.Tensor, attention_mask: torch.Tensor
    ) -> list[list[int]]:
        if self.use_crf:
            # TorchCRF uses viterbi_decode(h, mask) returning list[list[int]]
            return self.crf.viterbi_decode(span_logits, attention_mask.bool())
        return span_logits.argmax(dim=-1).cpu().tolist()

    @staticmethod
    def count_spans(bio_sequence: list[int]) -> int:
        return sum(1 for tag in bio_sequence if tag == 1)

    @staticmethod
    def predict_count_from_ordinal(ordinal_logits: torch.Tensor) -> int:
        probs = torch.sigmoid(ordinal_logits)
        return int((probs > 0.5).sum().item())
