"""Web search powered by Anthropic's hosted ``web_search`` tool.

Anthropic exposes a hosted ``web_search_20250305`` server-tool that the API
runs on its side. We call ``MODEL_DEEP`` with that tool enabled and ask it to
return a short, conversational summary the bot can quote naturally — never a
formal report.

If the API call fails (no entitlement, transient error, etc) we return ``None``
and let the caller fall back to writing without external info.
"""

from __future__ import annotations

import logging
from typing import Any

from anthropic import AsyncAnthropic

log = logging.getLogger(__name__)

WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 3,
}

PROMPT = """\
Eres un asistente para un bot de Telegram. Te paso el contexto reciente de la conversación.
Busca en la web la info que falta para que el bot pueda responder con datos reales.

Contexto:
{context}

Devuelve un PÁRRAFO BREVE (máx 4 frases) en español, en estilo conversacional, con los datos clave
y nombres de competidores/cifras si aplica. No formato bullets, no markdown, sin "como puedes ver" ni rollos.
"""


async def run_web_search(client: AsyncAnthropic, model: str, context: str) -> str | None:
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=600,
            tools=[WEB_SEARCH_TOOL],
            messages=[{"role": "user", "content": PROMPT.format(context=context)}],
        )
    except Exception:
        log.exception("Web search call failed")
        return None

    chunks: list[str] = []
    for block in response.content or []:
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip() or None
