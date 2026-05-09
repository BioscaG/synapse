"""Spontaneous-conversation scheduler.

Periodically rolls a few dice to decide whether a bot should kick off a new
conversation in the group. Uses APScheduler for the periodic ticks.

Triggers (defined by the spec):

- ``random_thought``: a bot just felt like saying something random.
- ``follow_up``: there's a pending topic from a previous conversation.
- ``news_share``: Víctor has seen a relevant news item (currently a stub).

Real-chat distribution of "who starts a conversation" is used to weight the
random trigger: Víctor 39%, Jordi 33%, Guido 28%.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import TYPE_CHECKING, Awaitable, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler

if TYPE_CHECKING:
    from orchestrator.manager import ConversationManager

log = logging.getLogger(__name__)

WEEKDAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

INITIATOR_WEIGHTS = {"victor": 0.39, "jordi": 0.33, "guido": 0.28}


def is_within_active_hours(now: datetime, active_hours: tuple[int, int]) -> bool:
    start, end = active_hours
    hour = now.hour
    if start <= end:
        return start <= hour < end
    # wraps midnight (e.g. 22..3)
    return hour >= start or hour < end


def is_peak(now: datetime, peak_hours: tuple[int, int]) -> bool:
    start, end = peak_hours
    return start <= now.hour < end


def pick_initiator(weights: dict[str, float] | None = None) -> str:
    weights = weights or INITIATOR_WEIGHTS
    keys = list(weights.keys())
    return random.choices(keys, weights=[weights[k] for k in keys], k=1)[0]


class SpontaneousScheduler:
    """Owns an APScheduler job that periodically tries to start conversations."""

    def __init__(
        self,
        manager: "ConversationManager",
        active_hours: tuple[int, int],
        peak_hours: tuple[int, int],
        max_per_day: int,
        check_every_minutes: int = 30,
    ) -> None:
        self.manager = manager
        self.active_hours = active_hours
        self.peak_hours = peak_hours
        self.max_per_day = max_per_day
        self.check_every_minutes = check_every_minutes
        self._scheduler = AsyncIOScheduler()

    def start(self) -> None:
        self._scheduler.add_job(
            self._tick,
            "interval",
            minutes=self.check_every_minutes,
            id="spontaneous_tick",
            next_run_time=datetime.now(),
        )
        self._scheduler.start()
        log.info(
            "Spontaneous scheduler started (every %d min, active %s, peak %s)",
            self.check_every_minutes,
            self.active_hours,
            self.peak_hours,
        )

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    # ------------------------------------------------------------------ ticks
    async def _tick(self) -> None:
        now = datetime.now()
        if not is_within_active_hours(now, self.active_hours):
            return

        usage = self.manager.memory.usage_today()
        if usage["spontaneous_convos"] >= self.max_per_day:
            return
        if self.manager.is_silenced():
            return
        if self.manager.has_recent_activity(minutes=20):
            return

        # Probability bumps for peak hours / Tuesday & Friday (real-chat data).
        prob = 0.03
        if is_peak(now, self.peak_hours):
            prob *= 2
        if now.weekday() in (1, 4):  # Tuesday, Friday
            prob *= 1.4

        if random.random() > prob:
            return

        trigger, payload = self._pick_trigger()
        agent_id = pick_initiator()
        agent = self.manager.agents[agent_id]
        weekday = WEEKDAYS_ES[now.weekday()]

        result = await agent.generate_spontaneous(
            client=self.manager.client,
            model=self.manager.config.model_fast,
            memory=self.manager.memory,
            hour=now.hour,
            weekday=weekday,
            trigger=trigger,
            trigger_payload=payload,
        )
        if not result:
            return

        log.info("Spontaneous convo started by %s (trigger=%s)", agent_id, trigger)
        self.manager.memory.record_spontaneous_convo()
        await self.manager.deliver_burst(agent, result["messages"])

    def _pick_trigger(self) -> tuple[str, str | None]:
        pending = self.manager.memory.get_pending_topics(limit=5)
        if pending and random.random() < 0.4:
            return "follow_up", random.choice(pending)
        return "random_thought", None
