"""Plain-text formatters for summaries and idea boards.

These are deterministic — no LLM call. They consume rows from the memory
manager and render a Telegram-friendly message.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Iterable


STATUS_BADGES = {
    "proposed": "💡",
    "discussing": "💬",
    "researching": "🔍",
    "promising": "✅",
    "discarded": "❌",
    "dead": "💀",
}


def build_summary(history: Iterable[sqlite3.Row], ideas: Iterable[sqlite3.Row]) -> str:
    history = list(history)
    ideas = list(ideas)

    parts = ["📋 *Resumen de la sesión*", ""]

    if not history:
        parts.append("Nada relevante registrado en las últimas conversaciones.")
    else:
        parts.append("*Conversaciones recientes:*")
        for row in history:
            try:
                topics = json.loads(row["topics"] or "[]")
            except (json.JSONDecodeError, KeyError):
                topics = []
            tags = ", ".join(topics) if topics else "—"
            parts.append(f"• {row['date']} [{tags}] — {row['summary']}")

    parts.append("")
    if ideas:
        parts.append("*Board de ideas:*")
        for idea in ideas:
            badge = STATUS_BADGES.get(idea["status"], "•")
            extra = f" — {idea['kill_reason']}" if idea["kill_reason"] else ""
            parts.append(f"{badge} *{idea['title']}*{extra}")

    pending = []
    for row in history:
        try:
            pending.extend(json.loads(row["pending_items"] or "[]"))
        except (json.JSONDecodeError, KeyError):
            continue
    pending = list(dict.fromkeys(pending))[:8]
    if pending:
        parts.append("")
        parts.append("*Pendiente:*")
        parts.extend(f"• {item}" for item in pending)

    return "\n".join(parts).strip()


def build_idea_board(ideas: Iterable[sqlite3.Row]) -> str:
    ideas = list(ideas)
    if not ideas:
        return "💡 *Board de ideas* — vacío de momento."

    by_status: dict[str, list[sqlite3.Row]] = {}
    for idea in ideas:
        by_status.setdefault(idea["status"], []).append(idea)

    out = ["💡 *Board de ideas*", ""]
    order = ["promising", "researching", "discussing", "proposed", "discarded", "dead"]
    for status in order:
        rows = by_status.get(status, [])
        if not rows:
            continue
        badge = STATUS_BADGES.get(status, "•")
        for idea in rows:
            extra = f" — {idea['kill_reason']}" if idea["kill_reason"] else ""
            out.append(f"{badge} *{idea['title']}*{extra}")
        out.append("")
    return "\n".join(out).strip()
