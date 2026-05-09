"""Technical-feasibility helper for Jordi.

Returns an informal estimate the bot can quote: stack, time-to-MVP, APIs and
complexity score (1-10). No web search needed — pure prompting.
"""

from __future__ import annotations

import logging

from anthropic import AsyncAnthropic

log = logging.getLogger(__name__)

PROMPT = """\
Eres un asistente para un bot de Telegram (Jordi — el ejecutor del grupo). Te paso el contexto reciente.
Estima la viabilidad técnica de la idea que se discute:

- Stack técnico (lenguajes / frameworks / servicios)
- Tiempo de desarrollo: MVP vs producto completo
- APIs / servicios externos necesarios
- Complejidad 1-10

Contexto:
{context}

Devuelve un PÁRRAFO BREVE (máx 4 frases) en español, estilo conversacional informal, sin markdown,
como Jordi diría: "esto lo monto en un finde con FastAPI y la API de X" o "esto es mínimo 3 meses bro".
"""


async def run_tech_estimator(client: AsyncAnthropic, model: str, context: str) -> str | None:
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=400,
            messages=[{"role": "user", "content": PROMPT.format(context=context)}],
        )
    except Exception:
        log.exception("Tech estimator call failed")
        return None

    text = response.content[0].text if response.content else ""
    return text.strip() or None
