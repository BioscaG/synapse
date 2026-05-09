"""Belief / conversation summariser.

After a conversation winds down, the summariser asks Haiku to extract:

- New beliefs the agent should remember
- Updates to existing beliefs (confidence shifts, refinements)
- Beliefs that should be removed (proven wrong, obsolete)
- Idea-board updates (status changes, new ideas)
- Pending items (things someone said "we should investigate")
- A 1-2 sentence narrative summary

The summariser runs asynchronously so it never blocks the live conversation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Iterable

from anthropic import AsyncAnthropic

from memory.manager import GroupMessage, MemoryManager

log = logging.getLogger(__name__)

PROMPT_TEMPLATE = """\
Eres el sistema de memoria de {name}.

Esta conversación acaba de ocurrir:
{transcript}

Creencias actuales de {name}:
{beliefs}

¿Qué ha cambiado? Devuelve SOLO un JSON válido con esta forma:
{{
  "new_beliefs": [{{"category": "opinion|knowledge|idea_status|personal|relationship", "content": "...", "confidence": 0.0-1.0}}],
  "updated_beliefs": [{{"id": <int>, "content": "nuevo contenido", "confidence": 0.0-1.0}}],
  "removed_beliefs": [<int>, ...],
  "idea_updates": [{{"title": "...", "status": "proposed|discussing|researching|promising|discarded|dead", "reason": "..."}}],
  "pending_items": ["..."],
  "conversation_summary": "1-2 frases en español, como lo contaría {name}"
}}

REGLAS:
- Solo cambios RELEVANTES, no trivialidades.
- La confianza sube si algo se confirma y baja si se cuestiona.
- Si una idea se descartó, ponla como "discarded" con el motivo.
- pending_items son cosas concretas que alguien dijo "hay que investigar/hacer".
- NADA fuera del JSON.
"""


def _format_transcript(messages: Iterable[GroupMessage]) -> str:
    return "\n".join(f"[{m.sender_name}] {m.text}" for m in messages) or "(vacío)"


def _extract_json(raw: str) -> dict:
    """Parse the model output, tolerating ```json fences and trailing prose."""
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


class BeliefSummariser:
    """Calls Haiku to update an agent's cold memory after a conversation."""

    def __init__(self, client: AsyncAnthropic, memory: MemoryManager, model: str) -> None:
        self.client = client
        self.memory = memory
        self.model = model

    async def update_for_agent(
        self,
        agent_id: str,
        agent_name: str,
        messages: list[GroupMessage],
    ) -> dict:
        """Update belief storage for a single agent and return the parsed result."""
        if not messages:
            return {}
        prompt = PROMPT_TEMPLATE.format(
            name=agent_name,
            transcript=_format_transcript(messages),
            beliefs=self.memory.format_beliefs(agent_id),
        )
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=900,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:
            log.exception("Belief summariser API call failed for %s", agent_id)
            return {}
        self.memory.record_api_call(getattr(response.usage, "output_tokens", 0))
        text = response.content[0].text if response.content else ""
        try:
            data = _extract_json(text)
        except json.JSONDecodeError:
            log.warning("Could not parse belief update JSON for %s: %r", agent_id, text[:200])
            return {}

        self._apply(agent_id, data)
        return data

    def _apply(self, agent_id: str, data: dict) -> None:
        for belief in data.get("new_beliefs", []) or []:
            self.memory.add_belief(
                agent_id=agent_id,
                category=belief.get("category", "opinion"),
                content=belief["content"],
                confidence=float(belief.get("confidence", 0.5)),
                source="summariser",
            )
        for belief in data.get("updated_beliefs", []) or []:
            self.memory.update_belief(
                belief_id=int(belief["id"]),
                content=belief.get("content"),
                confidence=float(belief["confidence"]) if "confidence" in belief else None,
            )
        if removed := data.get("removed_beliefs"):
            self.memory.remove_beliefs(int(i) for i in removed)
        for idea in data.get("idea_updates", []) or []:
            self.memory.upsert_idea(
                title=idea["title"],
                status=idea.get("status", "proposed"),
                kill_reason=idea.get("reason") if idea.get("status") in {"discarded", "dead"} else None,
                killed_by=agent_id if idea.get("status") in {"discarded", "dead"} else None,
            )
        if summary := data.get("conversation_summary"):
            self.memory.log_conversation(
                summary=summary,
                pending_items=data.get("pending_items") or [],
            )
