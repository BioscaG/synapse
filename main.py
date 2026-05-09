"""Entry point — wires every component together and runs forever.

Boot sequence:

1. Load secrets from ``config.py`` (created from ``config.example.py``).
2. Initialise SQLite (creating the schema if missing) and seed initial memories.
3. Build the three :class:`Agent` instances.
4. Build the :class:`AsyncAnthropic` client and the :class:`TelegramHub`.
5. Build the :class:`ConversationManager`, wiring it to the hub's callbacks.
6. Start the spontaneous-conversation scheduler.
7. Start the Telegram applications and block until ``SIGINT`` / ``SIGTERM``.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from anthropic import AsyncAnthropic

from agents import build_guido, build_jordi, build_victor
from memory.manager import MemoryManager
from orchestrator.manager import ConversationManager, ManagerConfig
from orchestrator.scheduler import SpontaneousScheduler
from telegram_bot.commands import register_commands
from telegram_bot.handlers import wire
from telegram_bot.setup import PRIMARY_AGENT_ID, TelegramHub
from tools.registry import ToolRegistry


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Telegram's HTTPX logs are noisy at INFO; tone them down.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


async def _run() -> None:
    try:
        import config  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        print(
            "config.py not found. Copy config.example.py to config.py and fill in your tokens.",
            file=sys.stderr,
        )
        sys.exit(1)

    _configure_logging(getattr(config, "LOG_LEVEL", "INFO"))
    log = logging.getLogger("rials.main")

    memory = MemoryManager(
        db_path=config.DB_PATH,
        initial_memories_path=getattr(config, "INITIAL_MEMORIES_PATH", None),
    )

    agents = {
        "guido": build_guido(),
        "victor": build_victor(),
        "jordi": build_jordi(),
    }

    client = AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    tools = ToolRegistry(client=client, model=config.MODEL_DEEP)

    hub = TelegramHub(
        tokens=config.TELEGRAM_TOKENS,
        group_chat_id=config.GROUP_CHAT_ID,
        god_user_id=config.GOD_USER_ID,
    )

    manager = ConversationManager(
        agents=agents,
        memory=memory,
        client=client,
        config=ManagerConfig(
            model_fast=config.MODEL_FAST,
            model_deep=config.MODEL_DEEP,
            max_daily_calls=config.MAX_DAILY_API_CALLS,
            max_messages_per_convo=config.MAX_MESSAGES_PER_CONVO,
            burst_inter_delay=config.DELAY_RAFAGA,
        ),
        send_callback=hub.send,
        react_callback=hub.react,
        typing_callback=hub.typing,
        tool_registry=tools,
    )

    wire(manager, hub)
    register_commands(
        application=hub.apps[PRIMARY_AGENT_ID],
        hub=hub,
        manager=manager,
        primary_agent_id=PRIMARY_AGENT_ID,
    )

    scheduler = SpontaneousScheduler(
        manager=manager,
        active_hours=config.ACTIVE_HOURS,
        peak_hours=config.PEAK_HOURS,
        max_per_day=config.MAX_SPONTANEOUS_CONVOS_PER_DAY,
    )

    await hub.start()
    scheduler.start()
    log.info("RIALS is up. Press Ctrl+C to stop.")

    stop_event = asyncio.Event()

    def _on_signal() -> None:
        log.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            # Windows / non-asyncio-native platforms — fall back to default.
            signal.signal(sig, lambda *_: _on_signal())

    try:
        await stop_event.wait()
    finally:
        log.info("Shutting down...")
        scheduler.shutdown()
        await hub.shutdown()
        log.info("Bye.")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
