"""God-mode: detect intents in the human user's messages and act accordingly.

The "god" is the human user that owns the group. Their messages always hit
priority 1.0. A few keyword-based intents trigger special behaviour:

- ``stop_conversation``: keywords like "callaos", "basta", "parad" → silence all bots.
- ``brainstorm``: "investigad", "buscad", "analizad", "pensad bien" → switch
  to ``MODEL_DEEP`` and force ``need_search=True`` for at least one agent.
- ``debate``: "qué pensáis", "opinión", "ideas" → all bots respond.
- ``summary``: "resumen", "qué habéis hablado" → use ``doc_generator``.
- otherwise: normal response with high priority.

Detection is intentionally heuristic / keyword-based. No LLM call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

STOP_PATTERNS = re.compile(r"\b(callaos|callate|basta|parad|silencio|shhh+)\b", re.IGNORECASE)
BRAINSTORM_PATTERNS = re.compile(
    r"\b(investigad|investiga|buscad|busca|analizad|analiza|pensad bien|va enserio|brainstorm)\b",
    re.IGNORECASE,
)
DEBATE_PATTERNS = re.compile(r"\b(qu[eé] pens[aá]is|opini[oó]n|ideas|qu[eé] os parece)\b", re.IGNORECASE)
SUMMARY_PATTERNS = re.compile(
    r"\b(resumen|resume|qu[eé] hab[eé]is hablado|recap)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class GodIntent:
    kind: str  # 'stop' | 'brainstorm' | 'debate' | 'summary' | 'normal'
    raw: str


def detect_intent(text: str) -> GodIntent:
    if STOP_PATTERNS.search(text):
        return GodIntent("stop", text)
    if BRAINSTORM_PATTERNS.search(text):
        return GodIntent("brainstorm", text)
    if SUMMARY_PATTERNS.search(text):
        return GodIntent("summary", text)
    if DEBATE_PATTERNS.search(text):
        return GodIntent("debate", text)
    return GodIntent("normal", text)
