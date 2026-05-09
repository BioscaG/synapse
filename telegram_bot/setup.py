"""Telegram hub: three bots sharing one event loop.

Each personality is a separate ``python-telegram-bot`` ``Application`` with its
own token. To avoid processing every group message three times, we designate
one of them as the *primary listener* — the only application that registers
update handlers. The other two are only used to *send* messages, set
reactions, and emit typing indicators.

The three applications share the same asyncio event loop and the same
``ConversationManager``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from telegram import ReactionTypeEmoji, Update
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from memory.manager import GroupMessage

log = logging.getLogger(__name__)

PRIMARY_AGENT_ID = "guido"


class TelegramHub:
    """Owns the three Telegram applications and exposes send / react / typing."""

    def __init__(
        self,
        tokens: dict[str, str],
        group_chat_id: int,
        god_user_id: int,
    ) -> None:
        self.group_chat_id = group_chat_id
        self.god_user_id = god_user_id
        self.apps: dict[str, Application] = {
            agent_id: Application.builder().token(token).build()
            for agent_id, token in tokens.items()
        }
        self._bot_user_ids: dict[int, str] = {}
        self._on_human_message: Callable[[GroupMessage], Awaitable[None]] | None = None
        self._on_idle_tick: Callable[[], Awaitable[None]] | None = None
        self._idle_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------ wiring
    def attach_handlers(
        self,
        on_message: Callable[[GroupMessage], Awaitable[None]],
        on_idle_tick: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._on_human_message = on_message
        self._on_idle_tick = on_idle_tick
        primary = self.apps[PRIMARY_AGENT_ID]
        primary.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_update)
        )

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        for agent_id, app in self.apps.items():
            await app.initialize()
            me = await app.bot.get_me()
            self._bot_user_ids[me.id] = agent_id
            await app.start()
            log.info("Telegram bot started: %s (id=%s, @%s)", agent_id, me.id, me.username)

        # Only the primary application polls for updates.
        await self.apps[PRIMARY_AGENT_ID].updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )

        if self._on_idle_tick:
            self._idle_task = asyncio.create_task(self._idle_loop())

    async def run_until_stopped(self) -> None:
        await self._stop_event.wait()

    async def shutdown(self) -> None:
        self._stop_event.set()
        if self._idle_task:
            self._idle_task.cancel()
        try:
            await self.apps[PRIMARY_AGENT_ID].updater.stop()
        except Exception:
            log.debug("Updater stop raised", exc_info=True)
        for agent_id, app in self.apps.items():
            try:
                await app.stop()
                await app.shutdown()
            except Exception:
                log.debug("Shutdown of %s raised", agent_id, exc_info=True)

    # ------------------------------------------------------------------ outgoing
    async def send(
        self,
        agent_id: str,
        text: str,
        reply_to_message_id: int | None = None,
    ) -> int | None:
        bot = self.apps[agent_id].bot
        try:
            message = await bot.send_message(
                chat_id=self.group_chat_id,
                text=text,
                reply_to_message_id=reply_to_message_id,
            )
            return message.message_id
        except TelegramError:
            log.exception("send_message failed for %s", agent_id)
            return None

    async def react(self, agent_id: str, message_id: int, emoji: str) -> None:
        try:
            await self.apps[agent_id].bot.set_message_reaction(
                chat_id=self.group_chat_id,
                message_id=message_id,
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
        except TelegramError:
            log.debug("set_message_reaction failed for %s/%s", agent_id, emoji, exc_info=True)

    async def typing(self, agent_id: str) -> None:
        try:
            await self.apps[agent_id].bot.send_chat_action(
                chat_id=self.group_chat_id,
                action=ChatAction.TYPING,
            )
        except TelegramError:
            log.debug("send_chat_action failed for %s", agent_id, exc_info=True)

    # ------------------------------------------------------------------ incoming
    async def _handle_update(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if not message or not message.text:
            return
        if update.effective_chat is None or update.effective_chat.id != self.group_chat_id:
            return
        sender = update.effective_user
        if sender is None:
            return

        if sender.id in self._bot_user_ids:
            # The bot's own / sibling bot's messages already round-trip through
            # the manager via deliver_burst — ignore them on the listener side.
            return

        if sender.id != self.god_user_id:
            # Spec only models the human "god"; non-god humans are ignored.
            log.debug("Ignoring message from non-god human %s", sender.id)
            return

        if not self._on_human_message:
            return

        gm = GroupMessage(
            sender_id="god",
            sender_name="DIOS",
            text=message.text,
            is_from_god=True,
            telegram_message_id=message.message_id,
        )
        try:
            await self._on_human_message(gm)
        except Exception:
            log.exception("Manager raised on human message")

    # ------------------------------------------------------------------ idle ticks
    async def _idle_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(60)
                if self._on_idle_tick:
                    try:
                        await self._on_idle_tick()
                    except Exception:
                        log.exception("Idle tick raised")
        except asyncio.CancelledError:
            return
