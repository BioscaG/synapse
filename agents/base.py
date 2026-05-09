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

Prompt caching
--------------
Both calls split their prompts so the *static* portion (rules + persona) is
sent in the ``system`` parameter with ``cache_control``. The dynamic portion
(context, beliefs, banned phrases, new message) goes in the user message
without caching. With Anthropic's ephemeral cache the static block is paid
in full only on the first call within ~5 minutes; subsequent calls cost ~10%
of the input tokens. For our agents the static block is ~700-1500 tokens,
so the savings are large.
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


# --------------------------------------------------------------------------- prompts
# Static part of the evaluation prompt — cached. Beliefs are intentionally
# *not* included here: they don't help the "should I respond?" decision and
# they would bloat every evaluation call.
EVALUATION_SYSTEM_TEMPLATE = """\
Eres {name} en un grupo de Telegram con tus amigos Guido, Víctor y Jordi.

ESTO NO ES UN CHAT 1-A-1: es un grupo activo donde los 3 debatís CONTINUAMENTE
sobre ideas de negocio, IA, startups, inversión, vida personal y todo lo demás.
Cuando uno suelta algo, los otros dos saltan casi siempre desde su rol:
- Guido (emprendedor): se entusiasma con ideas, piensa monetización, go-to-market.
- Víctor (analista): busca pegas, competidores, datos, razones por las que falla.
- Jordi (ejecutor): estima viabilidad técnica, "cuánto tardo en montarlo".
Por defecto, en una charla activa, RESPONDES.

Tu tarea: decidir si quieres responder al último mensaje. Devuelve SOLO un JSON con esta forma:
{{
  "respond": true|false,
  "urgency": 0.0-1.0,
  "delay_seconds": <int 1-15>,
  "rafaga": <int 1-5>,
  "need_search": true|false,
  "tool": null|"web_search"|"market_analysis"|"tech_estimator",
  "react_emoji": null|"🔥"|"😭"|"🤔"|"👍"|"❌"|"😂"|"🤑"|"👀"|"💀",
  "reason": "una frase corta"
}}

CUÁNDO respond=true (sé generoso):
- DIOS habla (etiqueta [DIOS]): SIEMPRE respond=true, urgency=1.0.
- Te mencionan por nombre: respond=true, urgency alta.
- Hay una pregunta directa, retórica o "pensando en voz alta": respond=true.
- Alguien propone una idea, negocio, plan, plugin, app, startup, proyecto: respond=true (es VUESTRO tema, vais a debatirla aunque no se dirija a ti).
- Alguien dice algo que tu personalidad cuestionaría / matizaría / apoyaría: respond=true.
- La conversación está fluyendo y tienes algo nuevo que aportar desde tu rol: respond=true.

CUÁNDO respond=false (la excepción):
- Acabas de mandar 3+ mensajes seguidos sin que el resto haya hablado.
- El mensaje es solo "xd"/"ya"/"vale"/"sep" sin contenido (puedes poner react_emoji).
- No tienes NADA nuevo que añadir: lo que dirías ya está dicho en el contexto.
- Acabas de hablar tú y el último mensaje es una continuación trivial.

OTRAS REGLAS:
- rafaga: cuántos mensajes seguidos cortos quieres mandar (típico 1, a veces 2-5).
- need_search=true sólo si necesitas datos reales que no sabes (cifras, competidores).
- delay_seconds entre 1 y 15. Más rápido si te mencionan o es DIOS.
- NADA fuera del JSON.
"""


# Static part of the generation prompt — cached together with the agent persona.
GENERATION_SYSTEM_RULES = """\
Vas a generar tu siguiente intervención en el grupo de Telegram. Reglas:
- Mensajes CORTOS, estilo WhatsApp (1-2 líneas como máx por mensaje).
- Devuelve un JSON array de strings: ["msg1", "msg2", ...].
- Si vas a mandar varios, son una ráfaga: cada uno es una idea suelta.
- Usa TUS muletillas y errores ortográficos típicos.
- NUNCA escribas "jajaja" — usa la variante de risa que te corresponde.

EL ÚLTIMO MENSAJE MANDA:
- Tu respuesta tiene que conectar con el ÚLTIMO mensaje, no con cualquier mensaje del background.
- Si el último mensaje CAMBIA DE TEMA respecto al background, ABANDONAS el tema anterior
  y sigues el nuevo. Sin "volviendo a lo de antes", sin retomar el hilo viejo, ni siquiera al
  final de tu ráfaga. Aunque tengas algo pendiente del tema previo, hoy no toca.
- El background está ahí solo como referencia (saber quién dijo qué antes), no como agenda.
- NO añadas "bueno pues mañana sigo con X" ni similares para volver al tema viejo.

ANTI-REPETICIÓN (importantísimo, parece más natural):
- NO repitas frases que ya estén en el contexto, ni siquiera reformuladas.
- Si tu PUNTO ya lo has dicho en mensajes anteriores tuyos, NO insistas:
  o cambias de tema, o sueltas un detalle nuevo, o cierras con una frase corta y callas.
- En una conversación viva, después de 2 turnos sobre lo mismo, AVANZA o calla.

OUTPUT:
- SOLO el JSON array, nada más fuera.
"""


# Static part of the spontaneous prompt — cached with persona.
SPONTANEOUS_SYSTEM_RULES = """\
Vas a iniciar una conversación nueva en el grupo. Llevas un rato sin hablar y te apetece soltar algo
natural: una idea, una pregunta, algo gracioso, un follow-up, una noticia que has visto…

Devuelve SOLO un JSON con esta forma:
{
  "topic": "tema en una frase",
  "messages": ["msg1", "msg2", ...]
}
1 a 3 mensajes para iniciar. Estilo WhatsApp, corto, en español.
"""


# --------------------------------------------------------------------------- helpers
def _cache_block(text: str) -> dict:
    return {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}


def _record_usage(memory: "MemoryManager", response) -> None:
    """Track output tokens (input is amortised by caching). Best-effort."""
    out = getattr(response.usage, "output_tokens", 0) or 0
    memory.record_api_call(int(out))


# --------------------------------------------------------------------------- Agent
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

        system_blocks = [_cache_block(EVALUATION_SYSTEM_TEMPLATE.format(name=self.name))]
        user_payload = (
            "Contexto reciente del grupo:\n"
            f"{memory.format_context(10)}\n\n"
            "Nuevo mensaje a evaluar:\n"
            f"{self._format_message(message)}\n\n"
            "Decide y devuelve SOLO el JSON."
        )

        try:
            response = await client.messages.create(
                model=model,
                max_tokens=180,
                system=system_blocks,
                messages=[{"role": "user", "content": user_payload}],
            )
        except Exception:
            log.exception("Evaluation API call failed for %s", self.agent_id)
            return Evaluation.silent("api error")
        _record_usage(memory, response)

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
        system_blocks = [
            _cache_block(self.system_prompt),
            _cache_block(GENERATION_SYSTEM_RULES),
        ]

        banned = memory.get_overused_phrases(self.agent_id)
        banned_section = ""
        if banned:
            banned_section = (
                "EVITA estas frases que has usado mucho últimamente "
                "(busca otra forma de decirlo):\n"
                + "\n".join(f"- {p}" for p in banned)
                + "\n\n"
            )

        # Pull this agent's own recent messages from THIS conversation. The
        # banned list above is cross-conversation; this one is in-conversation
        # and stops the model from re-stating the same point five times.
        own_recent = [m.text for m in memory.hot_messages(20) if m.sender_id == self.agent_id][-6:]
        own_recent_section = ""
        if own_recent:
            own_recent_section = (
                "Tus últimos mensajes en ESTA conversación (NO los repitas, "
                "ni reformules el mismo punto — avanza o calla):\n"
                + "\n".join(f"- {t}" for t in own_recent)
                + "\n\n"
            )

        tool_section = ""
        if tool_results:
            tool_section = f"Has buscado info y has encontrado:\n{tool_results}\n\n"

        # Split context into background (older) + the latest message you're
        # actually responding to. Putting the latest in its own block + an
        # explicit "LO QUE RESPONDES" header stops the model from clinging to
        # whatever topic dominated the background when god (or anyone) shifts.
        hot = memory.hot_messages(16)
        if hot:
            latest = hot[-1]
            background = hot[:-1]
            background_lines = [
                f"[{'DIOS' if m.is_from_god else m.sender_name}] {m.text}" for m in background
            ]
            background_text = "\n".join(background_lines) if background_lines else "(no había nada antes)"
            latest_text = f"[{'DIOS' if latest.is_from_god else latest.sender_name}] {latest.text}"
            topic_shift_hint = ""
            if latest.is_from_god:
                topic_shift_hint = (
                    "El dios acaba de escribir. Si su mensaje CAMBIA DE TEMA respecto al background, "
                    "abandonas el tema anterior y respondes al nuevo. No retomes el hilo viejo.\n\n"
                )
        else:
            background_text = "(grupo en silencio)"
            latest_text = "(ninguno — abre tema)"
            topic_shift_hint = ""

        user_payload = (
            f"Tu memoria fría:\n{memory.format_beliefs(self.agent_id)}\n\n"
            f"Background del grupo (referencia, NO obligatorio seguir hablando de esto):\n"
            f"{background_text}\n\n"
            f"{topic_shift_hint}"
            f"⮕ MENSAJE AL QUE RESPONDES:\n{latest_text}\n\n"
            f"{own_recent_section}{banned_section}{tool_section}"
            f"Devuelve un JSON array de 1 a {max_msgs} mensajes cortos en español, "
            f"conectados con el mensaje al que respondes."
        )

        try:
            response = await client.messages.create(
                model=model,
                max_tokens=600,
                system=system_blocks,
                messages=[{"role": "user", "content": user_payload}],
            )
        except Exception:
            log.exception("Generation API call failed for %s", self.agent_id)
            return []
        _record_usage(memory, response)

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

        system_blocks = [
            _cache_block(self.system_prompt),
            _cache_block(SPONTANEOUS_SYSTEM_RULES),
        ]
        user_payload = (
            f"Son las {hour:02d}:00 del {weekday}.\n\n"
            f"Tu memoria fría:\n{memory.format_beliefs(self.agent_id)}\n\n"
            f"Temas pendientes de conversaciones anteriores:\n{pending_text}\n\n"
            f"{trigger_section}\n\n"
            "Escribe TU primer mensaje para iniciar la conversación. Devuelve SOLO el JSON."
        )

        try:
            response = await client.messages.create(
                model=model,
                max_tokens=400,
                system=system_blocks,
                messages=[{"role": "user", "content": user_payload}],
            )
        except Exception:
            log.exception("Spontaneous API call failed for %s", self.agent_id)
            return None
        _record_usage(memory, response)

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
