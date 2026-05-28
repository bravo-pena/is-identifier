"""
Legal sentence splitter using spaCy with post-processing for regulatory texts.

Handles:
- Numbered list items that should stay with their parent sentence (a), b), 1., 2.)
- Short fragments (< 20 chars) that are continuations, not new sentences
- Both Spanish (es_core_news_lg) and English (en_core_web_lg)
"""

import re
from functools import lru_cache

import spacy

_LIST_ITEM = re.compile(r"^(?:\(?[ivxlcdm]+\)|[a-záéíóúüñ]\)|[A-ZÁÉÍÓÚÜÑ]\)|\d+[\.\)])\s", re.IGNORECASE)
_PURE_LIST_LABEL = re.compile(r"^(?:\(?[ivxlcdm]+\)|[a-z]\)|[A-Z]\)|\d+[\.\)])\s*$", re.IGNORECASE)
_SHORT_COLON_HEADING = re.compile(r"^.{1,50}:\s*$")  # e.g. "2. DUES:", "EXECUTIVE BODY:"
# Fragment starting with digit + space + lowercase: continuation after "Sect. 1 by paying..."
_DIGIT_CONTINUATION = re.compile(r"^\d+\s+[a-záéíóú]")
# Sentence starting with a lowercase letter = continuation of previous (not a list item)
_LOWERCASE_START = re.compile(r"^[a-záéíóúüñ]")
_SHORT_FRAGMENT_THRESHOLD = 20


@lru_cache(maxsize=2)
def _load_model(language: str) -> spacy.language.Language:
    model = "es_core_news_lg" if language == "es" else "en_core_web_lg"
    try:
        return spacy.load(model)
    except OSError:
        raise OSError(
            f"spaCy model '{model}' not found. "
            f"Install it with: python -m spacy download {model}"
        )


def split_sentences(text: str, language: str = "es") -> list[str]:
    """
    Split a block of regulatory text into individual sentences.

    Args:
        text: Plain text block (one or several sentences).
        language: 'es' or 'en'.

    Returns:
        List of sentence strings.
    """
    if not text or not text.strip():
        return []

    nlp = _load_model(language)
    doc = nlp(text)
    raw = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    return _merge_legal_fragments(raw)


def _merge_legal_fragments(sentences: list[str]) -> list[str]:
    """
    Post-process spaCy sentence boundaries for legal texts.

    Rules:
    - List items starting with a)/b)/1./2.) are merged into the previous sentence.
    - Fragments shorter than the threshold (likely continuations) are merged back.
    """
    if not sentences:
        return []

    merged: list[str] = []
    buffer = sentences[0]

    for sent in sentences[1:]:
        if _PURE_LIST_LABEL.match(buffer) or _SHORT_COLON_HEADING.match(buffer):
            # bare label "1." / "a)" or short heading "2. DUES:" — prepend to next
            buffer = buffer.rstrip() + " " + sent
        elif (_LIST_ITEM.match(sent) or _DIGIT_CONTINUATION.match(sent)
              or _LOWERCASE_START.match(sent) or len(sent) < _SHORT_FRAGMENT_THRESHOLD):
            buffer = buffer.rstrip() + " " + sent
        else:
            merged.append(buffer)
            buffer = sent

    merged.append(buffer)
    return [s.strip() for s in merged if s.strip()]
