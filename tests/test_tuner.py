"""Deterministic n-gram repetition detector."""

from orchestrator.tuner import find_overused_phrases


def test_detects_repeated_catchphrase():
    messages = [
        "bro hazlo 100 pero validemos esto antes",
        "vale pero validemos esto bro",
        "creo q primero validemos esto",
        "bro hazlo 100 va",
        "jordi tu vas suelto bro hazlo 100",
        "pero validemos esto igual",
    ]
    overused = find_overused_phrases(messages, min_count=3)
    phrases = [p for p, _ in overused]
    assert any("validemos esto" in p for p in phrases), f"missed validemos: {phrases}"


def test_ignores_stopword_only_ngrams():
    messages = ["y q", "y q", "y q", "no es", "no es", "no es"] * 2
    overused = find_overused_phrases(messages, min_count=3)
    assert overused == [], f"flagged stopword ngrams: {overused}"


def test_min_count_threshold():
    messages = ["bro hazlo bien"] * 2  # only 2 occurrences
    overused = find_overused_phrases(messages, min_count=3)
    assert overused == []


def test_substring_dedup_keeps_longer():
    messages = ["validemos esto bro algo", "validemos esto bro algo", "validemos esto bro algo"]
    overused = find_overused_phrases(messages, min_count=3)
    phrases = [p for p, _ in overused]
    assert len(phrases) == 1, phrases
