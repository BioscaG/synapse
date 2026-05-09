"""Central conversation orchestrator.

For every incoming message:

1. Push it into hot memory.
2. If it comes from the human god, run god-mode special handling and force
   priority 1.0 on every agent.
3. Otherwise: the two non-sender agents evaluate in *parallel* (asyncio.gather).
4. Apply turn-taking heuristics to each evaluation.
5. Schedule each agent's response as a separate asyncio task that waits for
   its delay, optionally invokes a tool, generates the burst and sends it.

The manager also exposes ``deliver_burst`` so the spontaneous scheduler can
push messages without going through the eval/generate cycle.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Awaitable, Callable

from anthropic import AsyncAnthropic

from agents.base import Agent, Evaluation
from memory.manager import GroupMessage, MemoryManager
from memory.summarizer import BeliefSummariser
from orchestrator.god_mode import detect_intent
from orchestrator.turns import adjust_evaluation, decide_burst
from tools.registry import ToolRegistry

log = logging.getLogger(__name__)

SendCallback = Callable[..., Awaitable[int | None]]


@dataclass(frozen=True, slots=True)
class ManagerConfig:
    model_fast: str
    model_deep: str
    max_daily_calls: int
    max_messages_per_convo: int
    burst_inter_delay: tuple[float, float]
    conversation_timeout_seconds: int = 300


class ConversationManager:
    """Coordinates the three agents on top of Telegram + memory + tools."""

    def __init__(
        self,
        agents: dict[str, Agent],
        memory: MemoryManager,
        client: AsyncAnthropic,
        config: ManagerConfig,
        send_callback: SendCallback,
        react_callback: Callable[[str, int, str], Awaitable[None]] | None = None,
        typing_callback: Callable[[str], Awaitable[None]] | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.agents = agents
        self.memory = memory
        self.client = client
        self.config = config
        self._send = send_callback
        self._react = react_callback
        self._typing = typing_callback
        self.tools = tool_registry or ToolRegistry(client=client, model=config.model_deep)
        self.summariser = BeliefSummariser(client=client, memory=memory, model=config.model_fast)

        self._silenced_until: float = 0.0
        self._convo_message_count: int = 0
        self._last_activity_at: datetime | None = None
        self._pending_tasks: set[asyncio.Task] = set()
        self._summary_lock = asyncio.Lock()

    # ------------------------------------------------------------------ public API
    async def handle_message(self, message: GroupMessage) -> None:
        self.memory.push_message(message)
        self._last_activity_at = datetime.utcnow()
        self._convo_message_count += 1

        if message.is_from_god:
            await self._handle_god(message)
            return

        if self.is_silenced():
            return

        if self._convo_message_count >= self.config.max_messages_per_convo:
            log.info("Bot quota for this conversation reached, agents will calm down")
            return

        if not self._can_call_api():
            return

        # Reset other agents' streaks, increment streak of sender
        for agent in self.agents.values():
            if agent.agent_id == message.sender_id:
                agent.consecutive_msgs += 1
            else:
                agent.reset_streak()

        # Evaluate in parallel with the two NON-SENDER agents.
        evaluators = [a for a in self.agents.values() if a.agent_id != message.sender_id]
        results = await asyncio.gather(
            *(a.evaluate_message(self.client, self.config.model_fast, message, self.memory) for a in evaluators),
            return_exceptions=True,
        )

        scheduled = 0
        for agent, evaluation in zip(evaluators, results):
            if isinstance(evaluation, BaseException):
                log.warning("Evaluation crashed for %s: %s", agent.agent_id, evaluation)
                continue
            evaluation = adjust_evaluation(agent, evaluation, message, self.memory)
            if not evaluation.wants_to_respond:
                if evaluation.react_emoji and self._react and message.telegram_message_id:
                    await self._react(agent.agent_id, message.telegram_message_id, evaluation.react_emoji)
                continue
            self._schedule_response(agent, evaluation)
            scheduled += 1

        log.debug("Scheduled %d response(s) for message from %s", scheduled, message.sender_id)

    async def deliver_burst(
        self,
        agent: Agent,
        messages: list[str],
        reply_to_message_id: int | None = None,
    ) -> None:
        """Send a pre-generated burst of messages from ``agent``."""
        for i, text in enumerate(messages):
            if not text:
                continue
            if i > 0:
                lo, hi = self.config.burst_inter_delay
                await asyncio.sleep(random.uniform(lo, hi))
            await self._typing_for(agent.agent_id)
            telegram_id = await self._send(agent.agent_id, text, reply_to_message_id if i == 0 else None)
            self.memory.push_message(
                GroupMessage(
                    sender_id=agent.agent_id,
                    sender_name=agent.name,
                    text=text,
                    telegram_message_id=telegram_id,
                )
            )
            agent.last_msg_time = time.time()

    # ------------------------------------------------------------------ god-mode
    async def _handle_god(self, message: GroupMessage) -> None:
        intent = detect_intent(message.text)
        log.info("God spoke (intent=%s): %r", intent.kind, message.text[:80])

        if intent.kind == "stop":
            self._cancel_pending()
            self._silenced_until = time.time() + 60 * 15  # 15 minutes of silence
            return

        if intent.kind == "summary":
            await self._handle_summary_request(message)
            return

        # Reset silence on any other god message.
        self._silenced_until = 0.0
        self._convo_message_count = 1

        # Brainstorm: nudge agents toward tool use & deep model in their generation.
        force_deep = intent.kind == "brainstorm"

        evaluators = list(self.agents.values())
        results = await asyncio.gather(
            *(a.evaluate_message(self.client, self.config.model_fast, message, self.memory) for a in evaluators),
            return_exceptions=True,
        )

        # All three answer the god. Order them by delay so the earliest one
        # may add the "ostia el jefe"-style intro.
        ordered: list[tuple[Agent, Evaluation]] = []
        for agent, evaluation in zip(evaluators, results):
            if isinstance(evaluation, BaseException):
                log.warning("God-mode eval crashed for %s: %s", agent.agent_id, evaluation)
                continue
            adjusted = adjust_evaluation(
                agent,
                Evaluation(
                    wants_to_respond=True,
                    urgency=1.0,
                    estimated_delay=max(0.5, evaluation.estimated_delay),
                    is_rafaga=evaluation.is_rafaga,
                    rafaga_count=evaluation.rafaga_count,
                    needs_tools=evaluation.needs_tools or force_deep,
                    tool_to_use=evaluation.tool_to_use,
                    react_emoji=None,
                    reason="god priority",
                ),
                message,
                self.memory,
            )
            if adjusted.wants_to_respond:
                ordered.append((agent, adjusted))

        ordered.sort(key=lambda pair: pair[1].estimated_delay)
        for agent, evaluation in ordered:
            self._schedule_response(agent, evaluation, force_deep=force_deep)

    async def _handle_summary_request(self, message: GroupMessage) -> None:
        from tools.doc_generator import build_summary

        ideas = self.memory.get_ideas()
        history = self.memory.get_recent_summaries(limit=5)
        text = build_summary(history, ideas)
        spokesperson_id = "victor" if "victor" in self.agents else next(iter(self.agents))
        agent = self.agents[spokesperson_id]
        await self.deliver_burst(agent, [text], reply_to_message_id=message.telegram_message_id)

    # ------------------------------------------------------------------ scheduling
    def _schedule_response(self, agent: Agent, evaluation: Evaluation, *, force_deep: bool = False) -> None:
        log.info(
            "Scheduling %s in %.1fs (urgency=%.2f, burst=%s, tool=%s)",
            agent.agent_id,
            evaluation.estimated_delay,
            evaluation.urgency,
            evaluation.rafaga_count,
            evaluation.tool_to_use,
        )
        task = asyncio.create_task(self._run_response(agent, evaluation, force_deep=force_deep))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _run_response(self, agent: Agent, evaluation: Evaluation, *, force_deep: bool = False) -> None:
        try:
            await asyncio.sleep(max(0.0, evaluation.estimated_delay))
            if self.is_silenced():
                log.info("%s skipped: bots silenced", agent.agent_id)
                return

            tool_results: str | None = None
            use_deep = force_deep or evaluation.needs_tools
            if evaluation.needs_tools and evaluation.tool_to_use:
                tool_input = self.memory.format_context(8)
                tool_results = await self.tools.run(evaluation.tool_to_use, tool_input)

            burst = decide_burst(agent, evaluation)
            burst = min(burst, agent.config.get("max_consecutive", 5))
            model = self.config.model_deep if use_deep else self.config.model_fast
            log.info("%s generating (model=%s, burst=%d)", agent.agent_id, model, burst)
            messages = await agent.generate_response(
                client=self.client,
                model=model,
                memory=self.memory,
                max_msgs=burst,
                tool_results=tool_results,
            )
            if not messages:
                log.warning("%s generated no messages — check API logs above", agent.agent_id)
                return
            log.info("%s sending %d message(s)", agent.agent_id, len(messages))
            await self.deliver_burst(agent, messages)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Response task crashed for %s", agent.agent_id)

    def _cancel_pending(self) -> None:
        for task in list(self._pending_tasks):
            task.cancel()
        self._pending_tasks.clear()

    async def _typing_for(self, agent_id: str) -> None:
        if self._typing:
            try:
                await self._typing(agent_id)
            except Exception:
                log.debug("Typing indicator failed for %s", agent_id, exc_info=True)

    # ------------------------------------------------------------------ state queries
    def is_silenced(self) -> bool:
        return time.time() < self._silenced_until

    def has_recent_activity(self, minutes: int = 20) -> bool:
        if not self._last_activity_at:
            return False
        return (datetime.utcnow() - self._last_activity_at) < timedelta(minutes=minutes)

    def _can_call_api(self) -> bool:
        usage = self.memory.usage_today()
        if usage["api_calls"] >= self.config.max_daily_calls:
            log.warning("Daily API call cap reached (%d)", usage["api_calls"])
            return False
        return True

    # ------------------------------------------------------------------ background maintenance
    async def maybe_summarise_dead_conversation(self) -> None:
        """If the conversation has been quiet for ``conversation_timeout``, summarise it."""
        if not self._last_activity_at:
            return
        elapsed = (datetime.utcnow() - self._last_activity_at).total_seconds()
        if elapsed < self.config.conversation_timeout_seconds:
            return
        if not self.memory.hot_messages(1):
            return

        async with self._summary_lock:
            messages = self.memory.hot_messages(40)
            for agent in self.agents.values():
                await self.summariser.update_for_agent(agent.agent_id, agent.name, messages)
            self._convo_message_count = 0
            self._last_activity_at = None
