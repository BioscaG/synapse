"""Intent detection from god messages."""

from orchestrator.god_mode import detect_intent


def test_stop_keywords():
    for text in ["callaos un poco", "basta ya", "parad por favor", "shhhh", "silencio chicos"]:
        assert detect_intent(text).kind == "stop", text


def test_brainstorm_keywords():
    for text in [
        "investigad esto bien",
        "buscad info por favor",
        "analizad la idea",
        "pensad bien lo que vamos a hacer",
        "va enserio chicos",
    ]:
        assert detect_intent(text).kind == "brainstorm", text


def test_summary_keywords():
    for text in ["dame un resumen", "que habéis hablado hoy", "qué habeis hablado", "recap please"]:
        assert detect_intent(text).kind == "summary", text


def test_debate_keywords():
    for text in ["qué pensáis de eso", "queréis dar vuestra opinión", "qué os parece la idea"]:
        assert detect_intent(text).kind == "debate", text


def test_normal_fallback():
    for text in ["hola chicos", "que tal", "vamos a pensar una idea"]:
        assert detect_intent(text).kind == "normal", text
