"""Market-analysis helper for Víctor.

Wraps the same Anthropic web_search tool but with a different prompt: instead
of a generic summary, the model returns competitors, market size, entry
barriers and similar past attempts. The output is one short paragraph the bot
can paraphrase naturally.
"""

from __future__ import annotations

import logging

from anthropic import AsyncAnthropic

from tools.web_search import WEB_SEARCH_TOOL

log = logging.getLogger(__name__)

PROMPT = """\
Eres un asistente para un bot de Telegram (Víctor — el analista del grupo). Te paso el contexto reciente.
Busca en la web datos para evaluar la idea de negocio que se está discutiendo:

- Competidores existentes (2-3 nombres)
- Tamaño de mercado aproximado
- Barreras de entrada principales
- Casos similares que funcionaron o fracasaron

Contexto:
{context}

Devuelve un PÁRRAFO BREVE (máx 5 frases) en español, en estilo conversacional, con cifras y nombres
si los tienes. Nada de markdown ni bullets, lenguaje informal pero riguroso, como Víctor escribiría.
"""


async def run_market_analysis(client: AsyncAnthropic, model: str, context: str) -> str | None:
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=700,
            tools=[WEB_SEARCH_TOOL],
            messages=[{"role": "user", "content": PROMPT.format(context=context)}],
        )
    except Exception:
        log.exception("Market analysis call failed")
        return None

    chunks: list[str] = []
    for block in response.content or []:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip() or None
