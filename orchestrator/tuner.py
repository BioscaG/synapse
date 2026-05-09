"""StyleTuner: detect overused phrases per agent and feed them back into prompts.

Why
---
Even with personality prompts and the chained turn-taking in place, agents
fall into catchphrase loops ("vale pero validemos esto", "bro hazlo 100%").
Catchphrases that are charming once become robotic by the third repetition.

How
---
After every conversation the tuner reads each agent's last ~80 messages from
``agent_messages`` (a rolling per-agent log), counts 2–4 word phrases (skipping
stopword-only n-grams), keeps the ones that show up at least three times, and
writes them to ``style_feedback``. The generation prompt reads that table and
asks the agent to phrase things differently.

This is intentionally deterministic: no LLM call, no risk of feedback loops,
cheap to run on every conversation tick. The "bot manager" lives here as a
plain function, not as another personality with its own prompt.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Iterable

from memory.manager import MemoryManager

log = logging.getLogger(__name__)

# Common Spanish/chat stopwords that shouldn't count as "overused phrases" on
# their own. An n-gram is dropped only if EVERY word is a stopword — that way
# "validemos esto" survives even though "esto" is a stopword.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "al", "ante", "bajo", "con", "contra", "de", "del", "desde", "en",
        "entre", "hacia", "hasta", "para", "por", "segun", "según", "sin", "sobre",
        "tras", "el", "la", "los", "las", "un", "una", "unos", "unas",
        "y", "o", "u", "ni", "que", "como", "cuando", "donde", "si", "no",
        "es", "son", "ser", "estar", "estoy", "esta", "este", "esto", "esa",
        "ese", "eso", "esos", "esas", "estos", "estas",
        "yo", "tu", "tú", "el", "él", "ella", "nos", "os", "ellos", "ellas",
        "me", "te", "se", "le", "lo", "les", "su", "sus", "mi", "mis",
        "muy", "mas", "más", "ya", "tmb", "tb", "tambien", "también",
        "ns", "creo", "asi", "así", "bro", "xd", "bueno", "pues", "vale",
        "ah", "eh", "ay", "uy", "uf",
    }
)

WORD_RE = re.compile(r"[\wáéíóúüñçÁÉÍÓÚÜÑÇ]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def _ngrams(words: list[str], n: int) -> Iterable[tuple[str, ...]]:
    for i in range(len(words) - n + 1):
        yield tuple(words[i : i + n])


def find_overused_phrases(
    messages: Iterable[str],
    *,
    min_count: int = 3,
    n_range: tuple[int, int] = (2, 4),
    top_k: int = 8,
) -> list[tuple[str, int]]:
    """Return up to ``top_k`` (phrase, count) pairs the agent overuses.

    A phrase is "overused" when it appears in at least ``min_count`` distinct
    messages of the input. Substring duplicates are collapsed: if "validemos
    esto bro" appears 4 times, "validemos esto" is dropped.
    """
    per_message_phrases: list[set[str]] = []
    for text in messages:
        words = _tokenize(text)
        phrases: set[str] = set()
        for n in range(n_range[0], n_range[1] + 1):
            for ngram in _ngrams(words, n):
                if all(w in STOPWORDS for w in ngram):
                    continue
                phrase = " ".join(ngram)
                if len(phrase) < 7:
                    continue
                phrases.add(phrase)
        per_message_phrases.append(phrases)

    counter: Counter[str] = Counter()
    for phrases in per_message_phrases:
        counter.update(phrases)

    candidates = [(phrase, count) for phrase, count in counter.items() if count >= min_count]
    # Sort by count desc, then by phrase length desc (prefer longer / more specific).
    candidates.sort(key=lambda pair: (-pair[1], -len(pair[0])))

    selected: list[tuple[str, int]] = []
    for phrase, count in candidates:
        if any(phrase in s or s in phrase for s, _ in selected):
            continue
        selected.append((phrase, count))
        if len(selected) >= top_k:
            break
    return selected


class StyleTuner:
    """Background analyser that updates each agent's banned-phrase list."""

    def __init__(self, memory: MemoryManager, lookback: int = 80, min_count: int = 3) -> None:
        self.memory = memory
        self.lookback = lookback
        self.min_count = min_count

    def run_for_agent(self, agent_id: str) -> list[tuple[str, int]]:
        messages = self.memory.get_recent_agent_messages(agent_id, limit=self.lookback)
        if len(messages) < self.min_count * 2:
            log.debug("StyleTuner: only %d msgs for %s, skipping", len(messages), agent_id)
            return []
        phrases = find_overused_phrases(messages, min_count=self.min_count)
        self.memory.set_overused_phrases(agent_id, phrases)
        if phrases:
            log.info(
                "StyleTuner: %s overused phrases -> %s",
                agent_id,
                [f"{p} (x{c})" for p, c in phrases],
            )
        return phrases

    def run_for_all(self, agent_ids: Iterable[str]) -> dict[str, list[tuple[str, int]]]:
        return {agent_id: self.run_for_agent(agent_id) for agent_id in agent_ids}
