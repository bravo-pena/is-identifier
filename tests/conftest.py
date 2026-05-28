"""Shared fixtures for IS Identifier tests."""
import pytest


# Synthetic fixture sentences grouped by expected AIM.n.
# These cover: Spanish, different deontic types, zero-AIM headings, and consequence patterns.

SENTENCES_AIM_0 = [
    "Capítulo III. De la Junta de Gobierno.",
    "Sección 2. Disposiciones generales.",
    "Título IV. Del régimen sancionador.",
    "Capítulo I. Objeto y ámbito de aplicación.",
    "Disposición transitoria primera.",
]

SENTENCES_AIM_1_ES = [
    "Los regantes deberán pagar la cuota anual antes del 31 de marzo.",
    "La Junta de Gobierno estará compuesta por el Presidente, el Secretario y tres Vocales.",
    "El Presidente convocará la Asamblea General al menos una vez al año.",
    "Los miembros podrán solicitar información sobre el estado de la comunidad.",
    "El Secretario registrará las actas de todas las reuniones celebradas.",
]

SENTENCES_AIM_2_ES = [
    "Los propietarios deberán mantener las acequias en buen estado y comunicar cualquier avería al Presidente.",
    "El infractor será multado y, de lo contrario, podrá ser expulsado de la comunidad.",
    "La Junta aprobará el presupuesto anual y lo comunicará a todos los miembros.",
    "Los regantes deberán pagar la cuota y contribuir a las obras de mantenimiento.",
    "El Presidente convocará la reunión y elaborará el orden del día con antelación suficiente.",
]

SENTENCES_AIM_1_EN = [
    "The members shall pay the annual fee before March 31.",
    "The Board shall be composed of the President, Secretary, and three members.",
    "The President shall convene the General Assembly at least once a year.",
]

SENTENCES_CONSEQUENCE_ES = [
    "Los miembros deben pagar la cuota; de lo contrario, serán sancionados.",
    "El regante deberá limpiar su acequia; en caso contrario, se impondrá una multa.",
]


@pytest.fixture
def aim0_sentences():
    return SENTENCES_AIM_0


@pytest.fixture
def aim1_sentences_es():
    return SENTENCES_AIM_1_ES


@pytest.fixture
def aim2_sentences_es():
    return SENTENCES_AIM_2_ES


@pytest.fixture
def aim1_sentences_en():
    return SENTENCES_AIM_1_EN


@pytest.fixture
def consequence_sentences_es():
    return SENTENCES_CONSEQUENCE_ES
