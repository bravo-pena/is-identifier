"""
Institutional verb extractor using spaCy dependency parsing.

Identifies verbs that act as AIMs (institutional actions) or constitutive
functions in a sentence, cross-referenced against the verb lexicon.
"""

from .sentence_splitter import _load_model
from .verb_lexicon import lookup_verb_type

_MAIN_DEPS = {"ROOT", "conj", "xcomp", "ccomp", "advcl"}
_AUX_DEPS = {"aux", "auxpass"}
# Deontic auxiliaries worth surfacing directly in the output
_DEONTIC_LEMMAS_EN = {"shall", "must", "may", "can", "will", "should"}
_DEONTIC_LEMMAS_ES = {"deber", "poder", "tener", "haber", "corresponder"}


def extract_aims(sentence: str, language: str = "es") -> list[dict]:
    """
    Extract institutional verbs from a sentence.

    Returns a list of dicts with keys:
        verb           – surface form as it appears in the text
        lemma          – base form
        position       – character offset in the sentence
        suggested_type – institutional type from verb lexicon, or None
    """
    nlp = _load_model(language)
    doc = nlp(sentence)

    results = []
    seen_lemmas: set[str] = set()

    deontic_lemmas = _DEONTIC_LEMMAS_EN if language == "en" else _DEONTIC_LEMMAS_ES

    for token in doc:
        # ── Deontic auxiliary (AUX) ──────────────────────────────────────────
        # Capture modal auxiliaries directly (shall, must, may...) as they ARE
        # the institutional modality marker in IG, even though spaCy tags them AUX.
        if token.pos_ == "AUX" and token.dep_ in _AUX_DEPS:
            lemma = token.lemma_.lower()
            if lemma in deontic_lemmas and lemma not in seen_lemmas:
                seen_lemmas.add(lemma)
                results.append({
                    "verb": token.text,
                    "lemma": lemma,
                    "position": token.idx,
                    "suggested_type": None,   # modality, not a lexicon type
                })
            continue

        # ── Main lexical verb (VERB) ─────────────────────────────────────────
        if token.pos_ != "VERB":
            continue
        if token.dep_ not in _MAIN_DEPS:
            continue

        lemma = token.lemma_.lower()

        # Filter out noise: prefixes (re, de), single/two-char tokens
        if len(lemma) < 3 or not lemma[0].isalpha():
            continue

        # Skip repeated lemmas (coordinated identical verbs)
        if lemma in seen_lemmas:
            continue

        suggested_type = lookup_verb_type(lemma, language)

        # Include verb if it's in the lexicon OR if it has a deontic auxiliary
        has_deontic_aux = _has_deontic_aux(token, language)
        if suggested_type or has_deontic_aux:
            seen_lemmas.add(lemma)
            results.append({
                "verb": token.text,
                "lemma": lemma,
                "position": token.idx,
                "suggested_type": suggested_type,
            })

    return results


_DEONTICS_ES = {"debe", "deberá", "podrá", "puede", "tendrá", "habrá", "corresponde"}
_DEONTICS_EN = {"must", "shall", "should", "may", "will", "can"}


def _has_deontic_aux(token, language: str) -> bool:
    """Check if a verb token has a deontic auxiliary child."""
    deontics = _DEONTICS_ES if language == "es" else _DEONTICS_EN
    return any(
        child.dep_ in _AUX_DEPS and child.text.lower() in deontics
        for child in token.children
    )
