"""Base ``Agent`` class.

Each agent owns:

- A *system prompt* describing its persona (Spanish, WhatsApp-style).
- A *personality config* with timing, burst probability and response priors.
- A list of tool names it is allowed to invoke.

Two LLM calls per turn:

1. ``evaluate_message`` — cheap Haiku call returning a JSON decision:
   does the agent want to reply? how urgent? how fast? burst? needs a tool?
2. ``generate_response`` — Haiku for normal replies, Sonnet when a tool was
   used. Returns a list of short Spanish messages (1 for a single reply,
   2-5 for a burst).
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from anthropic import AsyncAnthropic

if TYPE_CHECKING:
    from memory.manager import GroupMessage, MemoryManager

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Evaluation:
    """Result of an agent's "do I want to reply?" decision."""

    wants_to_respond: bool
    urgency: float
    estimated_delay: float
    is_rafaga: bool
    rafaga_count: int
    needs_tools: bool
    tool_to_use: str | None
    react_emoji: str | None
    reason: str

    @classmethod
    def silent(cls, reason: str = "") -> "Evaluation":
        return cls(
            wants_to_respond=False,
            urgency=0.0,
            estimated_delay=0.0,
            is_rafaga=False,
            rafaga_count=0,
            needs_tools=False,
            tool_to_use=None,
            react_emoji=None,
            reason=reason,
        )


EVALUATION_PROMPT = """\
Eres {name} en un grupo de Telegram con tus amigos.

Contexto reciente del grupo:
{context}

Nuevo mensaje a evaluar:
{new_message}

Tu memoria fría (creencias, opiniones, conocimiento):
{beliefs}

Decide si quieres responder a ese mensaje. Devuelve SOLO un JSON con esta forma:
{{
  "respond": true|false,
  "urgency": 0.0-1.0,
  "delay_seconds": <int>,
  "rafaga": <int 1-5>,
  "need_search": true|false,
  "tool": null|"web_search"|"market_analysis"|"tech_estimator",
  "react_emoji": null|"🔥"|"😭"|"🤔"|"👍"|"❌"|"😂"|"🤑"|"👀"|"💀",
  "reason": "una frase corta"
}}

REGLAS:
- Si te mencionan por nombre o te preguntan: respond=true, urgency alta.
- Si es solo un "xd" / reacción de otro: probablemente respond=false (puedes poner react_emoji).
- Si alguien lanza una idea jugosa: respond=true.
- Si llevas muchos mensajes seguidos sin que hablen los demás: respond=false.
- Si es el DIOS (etiqueta [DIOS]): respond=true SIEMPRE, urgency=1.0.
- rafaga es cuántos mensajes seguidos cortos quieres mandar (típico 1, a veces 2-5).
- need_search=true sólo si necesitas datos del mundo real que no sabes.
- NADA fuera del JSON.
"""


GENERATION_PROMPT = """\
{system_prompt}

Contexto del grupo (últimos mensajes, en orden):
{context}

Tu memoria fría:
{beliefs}

{tool_section}

Responde con tu siguiente intervención en el grupo. Reglas:
- Mensajes CORTOS, estilo WhatsApp (1-2 líneas como máx).
- Devuelve un JSON array de strings: ["msg1", "msg2", ...] (1 a {max_msgs} elementos).
- Si vas a mandar varios, son ráfaga: cada uno es una idea suelta.
- Usa TUS muletillas y errores ortográficos típicos.
- NUNCA escribas "jajaja" — usa la variante de risa que te corresponde.
- NO repitas frases que ya estén en el contexto.
- SOLO el JSON array, nada más fuera.
"""

SPONTANEOUS_PROMPT = """\
{system_prompt}

Son las {hour:02d}:00 del {weekday}. Estás en tu grupo de Telegram con tus amigos.
Llevas un rato sin hablar y te apetece soltar algo.

Tu memoria fría:
{beliefs}

Temas pendientes de conversaciones anteriores:
{pending}

{trigger_section}

Escribe TU primer mensaje al grupo para iniciar la conversación. Algo natural que dirías tú:
una idea, una pregunta, algo gracioso, un follow-up, una noticia que has visto…

Devuelve SOLO un JSON con esta forma:
{{
  "topic": "tema en una frase",
  "messages": ["msg1", "msg2", ...]
}}
1 a 3 mensajes para iniciar. Estilo WhatsApp, corto, en español.
"""


@dataclass
class Agent:
    """A single bot personality."""

    name: str
    agent_id: str
    system_prompt: str
    config: dict
    tools: list[str] = field(default_factory=list)
    cooldown_until: float = 0.0
    consecutive_msgs: int = 0
    last_msg_time: float = 0.0

    # ----------------------------------------------------------------- helpers
    def _format_message(self, message: "GroupMessage") -> str:
        tag = "[DIOS]" if message.is_from_god else f"[{message.sender_name}]"
        return f"{tag} {message.text}"

    def _record_outgoing(self, count: int) -> None:
        now = time.time()
        self.consecutive_msgs += count
        self.last_msg_time = now

    def reset_streak(self) -> None:
        self.consecutive_msgs = 0

    # ----------------------------------------------------------------- LLM-driven decisions
    async def evaluate_message(
        self,
        client: AsyncAnthropic,
        model: str,
        message: "GroupMessage",
        memory: "MemoryManager",
    ) -> Evaluation:
        """Ask Haiku whether this agent wants to respond to ``message``."""
        if message.sender_id == self.agent_id:
            return Evaluation.silent("self message")

        prompt = EVALUATION_PROMPT.format(
            name=self.name,
            context=memory.format_context(15),
            new_message=self._format_message(message),
            beliefs=memory.format_beliefs(self.agent_id),
        )
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=180,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:
            log.exception("Evaluation API call failed for %s", self.agent_id)
            return Evaluation.silent("api error")
        memory.record_api_call(getattr(response.usage, "output_tokens", 0))

        text = response.content[0].text if response.content else ""
        try:
            payload = _extract_json(text)
        except json.JSONDecodeError:
            log.warning("Evaluation JSON parse failed for %s: %r", self.agent_id, text[:160])
            return Evaluation.silent("parse error")

        return Evaluation(
            wants_to_respond=bool(payload.get("respond", False)),
            urgency=float(payload.get("urgency", 0.0)),
            estimated_delay=float(payload.get("delay_seconds", 2.0)),
            is_rafaga=int(payload.get("rafaga", 1)) > 1,
            rafaga_count=max(1, min(5, int(payload.get("rafaga", 1)))),
            needs_tools=bool(payload.get("need_search", False)),
            tool_to_use=payload.get("tool"),
            react_emoji=payload.get("react_emoji"),
            reason=str(payload.get("reason", "")),
        )

    async def generate_response(
        self,
        client: AsyncAnthropic,
        model: str,
        memory: "MemoryManager",
        max_msgs: int,
        tool_results: str | None = None,
    ) -> list[str]:
        """Ask the model for the next 1..max_msgs messages this agent will send."""
        tool_section = ""
        if tool_results:
            tool_section = f"Has buscado info y has encontrado:\n{tool_results}\n"

        prompt = GENERATION_PROMPT.format(
            system_prompt=self.system_prompt,
            context=memory.format_context(15),
            beliefs=memory.format_beliefs(self.agent_id),
            tool_section=tool_section,
            max_msgs=max_msgs,
        )
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:
            log.exception("Generation API call failed for %s", self.agent_id)
            return []
        memory.record_api_call(getattr(response.usage, "output_tokens", 0))

        text = response.content[0].text if response.content else ""
        messages = _parse_messages(text, max_msgs)
        if messages:
            self._record_outgoing(len(messages))
        return messages

    async def generate_spontaneous(
        self,
        client: AsyncAnthropic,
        model: str,
        memory: "MemoryManager",
        hour: int,
        weekday: str,
        trigger: str = "random_thought",
        trigger_payload: str | None = None,
    ) -> dict | None:
        """Ask the model to bootstrap a spontaneous conversation."""
        pending = memory.get_pending_topics(limit=8)
        pending_text = "\n".join(f"- {t}" for t in pending) if pending else "(ninguno)"

        if trigger == "follow_up" and trigger_payload:
            trigger_section = f"Ayer quedó pendiente: {trigger_payload}"
        elif trigger == "news_share" and trigger_payload:
            trigger_section = f"Has visto esta noticia: {trigger_payload}"
        else:
            trigger_section = "Trigger: random_thought (suelta lo que se te ocurra)."

        prompt = SPONTANEOUS_PROMPT.format(
            system_prompt=self.system_prompt,
            hour=hour,
            weekday=weekday,
            beliefs=memory.format_beliefs(self.agent_id),
            pending=pending_text,
            trigger_section=trigger_section,
        )
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:
            log.exception("Spontaneous API call failed for %s", self.agent_id)
            return None
        memory.record_api_call(getattr(response.usage, "output_tokens", 0))

        text = response.content[0].text if response.content else ""
        try:
            data = _extract_json(text)
        except json.JSONDecodeError:
            log.warning("Spontaneous parse failed for %s: %r", self.agent_id, text[:160])
            return None
        msgs = data.get("messages") or []
        if not isinstance(msgs, list) or not msgs:
            return None
        return {
            "topic": str(data.get("topic", "")),
            "messages": [str(m) for m in msgs[:3] if str(m).strip()],
        }

    # ----------------------------------------------------------------- timing helpers
    def burst_size(self) -> int:
        """Pick a burst size based on configured probability and range."""
        if random.random() > self.config.get("rafaga_probability", 0.3):
            return 1
        lo, hi = self.config.get("rafaga_size", (2, 4))
        return random.randint(lo, hi)

    def base_delay(self) -> float:
        lo, hi = self.config.get("delay_base", (1.0, 3.0))
        return random.uniform(lo, hi)

    def burst_delay(self) -> float:
        lo, hi = self.config.get("delay_rafaga", (0.3, 0.8))
        return random.uniform(lo, hi)


# ---------------------------------------------------------------- module helpers
def _extract_json(raw: str):
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"[\[{].*[\]}]", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _parse_messages(raw: str, max_msgs: int) -> list[str]:
    """Parse the JSON array of messages produced by ``generate_response``."""
    try:
        data = _extract_json(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip().strip("`").strip()
        if cleaned:
            return [cleaned[:300]]
        return []
    if isinstance(data, str):
        return [data][:max_msgs]
    if isinstance(data, list):
        return [str(m).strip() for m in data if str(m).strip()][:max_msgs]
    if isinstance(data, dict) and "messages" in data:
        msgs = data["messages"]
        if isinstance(msgs, list):
            return [str(m).strip() for m in msgs if str(m).strip()][:max_msgs]
    return []
