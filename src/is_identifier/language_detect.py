"""
Lightweight es/en language detection for regulatory documents.

Counts high-frequency function words from disjoint Spanish/English sets.
The corpus is Spanish/English only, so a stopword ratio is reliable and
avoids an extra dependency. Ties and empty text default to Spanish.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-záéíóúüñ]+")

# The two sets are disjoint by construction — never add a word to one set
# that is also a common word in the other language.
_ES_WORDS = frozenset({
    "el", "la", "los", "las", "una", "unos", "unas", "de", "del", "al",
    "que", "se", "por", "como", "más", "su", "sus", "este", "esta",
    "ser", "son", "está", "y", "o", "en", "con", "para", "deberá",
    "podrá", "será", "artículo", "miembros", "cada", "cualquier",
})
_EN_WORDS = frozenset({
    "the", "of", "and", "to", "is", "are", "be", "that", "this", "for",
    "with", "by", "on", "from", "shall", "may", "must", "will", "not",
    "it", "as", "any", "all", "who", "have", "has", "or", "at",
    "members", "meeting",
})

_SAMPLE_CHARS = 15000


def detect_language(text: str) -> str:
    """Return 'es' or 'en' for the given text (default 'es' on tie/empty)."""
    words = _WORD_RE.findall(text[:_SAMPLE_CHARS].lower())
    es = sum(1 for w in words if w in _ES_WORDS)
    en = sum(1 for w in words if w in _EN_WORDS)
    return "en" if en > es else "es"
