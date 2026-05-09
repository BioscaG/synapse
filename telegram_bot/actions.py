"""High-level Telegram actions used by the orchestrator.

Currently a thin layer of conveniences:

- :func:`pick_emoji_for_agent`: the personality's "default" emoji palette,
  used as a fallback when the model didn't return ``react_emoji``.
- :func:`format_idea_board`: re-export of the doc-generator helper for use
  from the handler layer.

Most send/react logic lives in :mod:`telegram_bot.setup`; this module exists
so that future actions (stickers, polls, file uploads) can be added in one
obvious place.
"""

from __future__ import annotations

import random

from tools.doc_generator import build_idea_board, build_summary  # noqa: F401  (re-exports)

EMOJI_PALETTE = {
    "guido": ["😭", "🔥", "👀"],
    "victor": ["🤔", "👍", "❌"],
    "jordi": ["🔥", "😂", "🤑"],
}


def pick_emoji_for_agent(agent_id: str) -> str:
    palette = EMOJI_PALETTE.get(agent_id, ["👍"])
    return random.choice(palette)


__all__ = ["build_idea_board", "build_summary", "pick_emoji_for_agent"]
