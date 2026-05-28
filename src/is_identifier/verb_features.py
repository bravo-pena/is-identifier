from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

from .verb_lexicon import VERB_LEXICON_EN, VERB_LEXICON_ES, lookup_verb_type


DEONTIC_RE_ES = re.compile(
    r"\b(debe(?:n)?|debera(?:n)?|deberan|podra(?:n)?|podran|puede(?:n)?|"
    r"tendra(?:n)?\s+que|tendran\s+que|habra(?:n)?\s+de|habran\s+de|"
    r"corresponde(?:ra)?|correspondera|queda(?:n)?\s+(?:obligado|prohibido|facultado))\b",
    flags=re.IGNORECASE,
)
DEONTIC_RE_EN = re.compile(
    r"\b(shall|must|should|may|will|can|have\s+to|required\s+to|obligated\s+to)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class VerbFeatures:
    n_institutional_verbs: int
    has_institutional_verb: bool
    verb_lemmas: list[str]
    verb_categories: list[str]
    has_deontic_aux: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "n_institutional_verbs": self.n_institutional_verbs,
            "has_institutional_verb": self.has_institutional_verb,
            "verb_lemmas": ";".join(self.verb_lemmas),
            "verb_categories": ";".join(self.verb_categories),
            "has_deontic_aux": self.has_deontic_aux,
        }


def extract_verb_features(sentence: str, language: str = "es") -> VerbFeatures:
    """
    Extract non-predictive verb features for QA/error analysis.

    Prefer the existing spaCy-based extractor. If the required spaCy model is not
    installed, fall back to lexicon substring matching so audit scripts can still run.
    """
    sentence = sentence or ""
    deontic_re = DEONTIC_RE_EN if language == "en" else DEONTIC_RE_ES
    has_deontic_aux = bool(deontic_re.search(sentence))

    try:
        from .aim_extractor import extract_aims

        aims = extract_aims(sentence, language=language)
        lemmas = []
        categories = []
        for aim in aims:
            lemma = str(aim.get("lemma") or "").lower()
            if not lemma or lemma in lemmas:
                continue
            lemmas.append(lemma)
            category = aim.get("suggested_type") or lookup_verb_type(lemma, language)
            if category and category not in categories:
                categories.append(str(category))
    except Exception as exc:
        warnings.warn(
            f"verb_features: spaCy extraction failed ({exc!r}); using lexicon fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        lemmas, categories = _lexicon_fallback(sentence, language)

    return VerbFeatures(
        n_institutional_verbs=len(lemmas),
        has_institutional_verb=bool(lemmas),
        verb_lemmas=lemmas,
        verb_categories=categories,
        has_deontic_aux=has_deontic_aux,
    )


def _lexicon_fallback(sentence: str, language: str) -> tuple[list[str], list[str]]:
    lexicon = VERB_LEXICON_EN if language == "en" else VERB_LEXICON_ES
    lowered = sentence.lower()
    lemmas: list[str] = []
    categories: list[str] = []

    for category, verbs in lexicon.items():
        for lemma in sorted(verbs, key=len, reverse=True):
            if re.search(rf"\b{re.escape(lemma.lower())}\w*\b", lowered):
                if lemma not in lemmas:
                    lemmas.append(lemma)
                if category not in categories:
                    categories.append(category)
    return lemmas, categories
