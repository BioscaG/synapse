"""Slash commands for the human god.

Registered on the primary listener bot. Only the configured ``GOD_USER_ID``
in the configured group is allowed to invoke them — random members of the
group can't trigger admin actions.

Available commands
------------------
- ``/board``   — render the shared idea board (status badges + kill reasons).
- ``/stats``   — today's API call count, tokens, spontaneous-convo count.
- ``/summary`` — last few conversation summaries pulled from the cold log.
- ``/silence`` — silence the bots for 15 minutes (cleaner than typing
  "callaos").
- ``/wake``    — clear an active silence.
- ``/help``    — list the commands.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from tools.doc_generator import build_idea_board, build_summary

if TYPE_CHECKING:
    from orchestrator.manager import ConversationManager
    from telegram_bot.setup import TelegramHub

log = logging.getLogger(__name__)


HELP_TEXT = (
    "Commandos disponibles (solo para ti):\n"
    "/board — board de ideas\n"
    "/stats — uso del día\n"
    "/summary — resúmenes recientes\n"
    "/silence — callar a los bots 15 min\n"
    "/wake — quitar silencio\n"
    "/help — esto"
)


def register_commands(
    application: Application,
    hub: "TelegramHub",
    manager: "ConversationManager",
    primary_agent_id: str,
) -> None:
    """Wire ``/command`` handlers onto the primary application's update polling."""

    async def _is_authorized(update: Update) -> bool:
        chat = update.effective_chat
        user = update.effective_user
        if chat is None or chat.id != hub.group_chat_id:
            return False
        if user is None or user.id != hub.god_user_id:
            return False
        return True

    async def _send(text: str, parse_mode: str | None = "Markdown") -> None:
        await hub.send(primary_agent_id, text, parse_mode=parse_mode)

    async def board(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _is_authorized(update):
            return
        ideas = manager.memory.get_ideas()
        await _send(build_idea_board(ideas))

    async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _is_authorized(update):
            return
        usage = manager.memory.usage_today()
        # Rough cost estimate at Haiku 4.5 prices ($1/MTok input, $5/MTok output).
        # We don't separate input vs output so use a 1:5 average ~$3/MTok.
        approx_cost = (usage["tokens_used"] / 1_000_000) * 3.0
        text = (
            "📊 *Hoy*\n"
            f"• Llamadas API: {usage['api_calls']}\n"
            f"• Tokens (output): {usage['tokens_used']}\n"
            f"• Convos espontáneas: {usage['spontaneous_convos']}\n"
            f"• Coste aprox: ${approx_cost:.3f}\n"
            f"• Hot context: {manager.memory.hot_size()} mensajes\n"
            f"• Conversación actual: {manager._convo_message_count} msg / "
            f"{manager.config.max_messages_per_convo} cap"
        )
        await _send(text)

    async def summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _is_authorized(update):
            return
        ideas = manager.memory.get_ideas()
        history = manager.memory.get_recent_summaries(limit=5)
        await _send(build_summary(history, ideas))

    async def silence(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _is_authorized(update):
            return
        manager._cancel_pending()
        manager._silenced_until = time.time() + 60 * 15
        await _send("🤐 Silenciados 15 min. /wake para devolverlos.", parse_mode=None)

    async def wake(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _is_authorized(update):
            return
        manager._silenced_until = 0.0
        await _send("👀 Despiertos.", parse_mode=None)

    async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await _is_authorized(update):
            return
        await _send(HELP_TEXT, parse_mode=None)

    for name, handler in [
        ("board", board),
        ("stats", stats),
        ("summary", summary),
        ("silence", silence),
        ("wake", wake),
        ("help", help_cmd),
    ]:
        application.add_handler(CommandHandler(name, handler))

    log.info("Registered slash commands: /board /stats /summary /silence /wake /help")
