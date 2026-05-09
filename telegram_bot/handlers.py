"""Wire :class:`ConversationManager` into :class:`TelegramHub`.

The manager is constructed with the hub's outgoing methods (``hub.send``,
``hub.react``, ``hub.typing``) as callbacks. Once both objects exist this
helper attaches the manager's incoming handlers to the hub.
"""

from __future__ import annotations

from orchestrator.manager import ConversationManager
from telegram_bot.setup import TelegramHub


def wire(manager: ConversationManager, hub: TelegramHub) -> None:
    """Register the manager's callbacks on the hub."""
    hub.attach_handlers(
        on_message=manager.handle_message,
        on_idle_tick=manager.maybe_summarise_dead_conversation,
    )
