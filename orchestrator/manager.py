"""Central conversation orchestrator.

For every incoming message:

1. Push it into hot memory.
2. If it comes from the human god, run god-mode special handling (intent
   detection for ``stop`` / ``summary`` / ``brainstorm``) and forward to
   :meth:`_dispatch` with priority forced to 1.0.
3. Otherwise call :meth:`_dispatch` directly.

:meth:`_dispatch` is the heart of turn-taking:

- Evaluate (in parallel) every non-sender agent on the new message.
- Apply turn-taking heuristics.
- Pick **one** winner (highest urgency, lowest delay) and schedule its
  response.

When that response is delivered, :meth:`deliver_burst` calls :meth:`_dispatch`
again on the bot's *own* last message — re-evaluating the **other** two
agents. This produces an organic chain: each bot sees what the previous one
just said and decides whether to chime in. The chain stops naturally when
nobody wants to respond, when ``_can_continue_conversation`` says so (silence,
message cap, daily API cap), or when the user (god) interjects.

The manager also exposes :meth:`deliver_burst` so the spontaneous scheduler
can push opening messages without going through the eval cycle.
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
from orchestrator.tuner import StyleTuner
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
    # If god speaks after more than this many seconds of silence, the hot
    # context is wiped before processing the new message — otherwise the bots
    # bring up the previous conversation as if it were still going.
    stale_gap_seconds: int = 120


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
        self.tuner = StyleTuner(memory=memory)

        self._silenced_until: float = 0.0
        self._convo_message_count: int = 0
        self._last_activity_at: datetime | None = None
        self._pending_tasks: set[asyncio.Task] = set()
        self._summary_lock = asyncio.Lock()

    # ------------------------------------------------------------------ public API
    async def handle_message(self, message: GroupMessage) -> None:
        # If god speaks after a long pause, the previous conversation is
        # effectively over. Wipe the hot context so the bots don't reply as
        # if the old thread were still alive.
        if message.is_from_god and self._is_conversation_stale():
            log.info("Stale context detected before god message — flushing")
            await self._flush_stale_conversation()

        self.memory.push_message(message)
        self._last_activity_at = datetime.utcnow()
        self._convo_message_count += 1

        if message.is_from_god:
            await self._handle_god(message)
            return

        if not self._can_continue_conversation():
            return

        await self._dispatch(message)

    async def deliver_burst(
        self,
        agent: Agent,
        messages: list[str],
        reply_to_message_id: int | None = None,
    ) -> None:
        """Send a pre-generated burst of messages from ``agent``.

        After the burst lands, re-dispatch the conversation so the *other*
        agents can react to what was just said. This is how the chain stays
        alive without firing every bot in parallel on the original message.
        """
        last_msg: GroupMessage | None = None
        for i, text in enumerate(messages):
            if not text:
                continue
            if i > 0:
                lo, hi = self.config.burst_inter_delay
                await asyncio.sleep(random.uniform(lo, hi))
            await self._typing_for(agent.agent_id)
            telegram_id = await self._send(
                agent.agent_id, text, reply_to_message_id if i == 0 else None
            )
            last_msg = GroupMessage(
                sender_id=agent.agent_id,
                sender_name=agent.name,
                text=text,
                telegram_message_id=telegram_id,
            )
            self.memory.push_message(last_msg)
            self._convo_message_count += 1
            agent.last_msg_time = time.time()

        if last_msg and self._can_continue_conversation():
            await self._dispatch(last_msg)

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

        # Reset silence/quota on any other god message — the user is alive.
        self._silenced_until = 0.0
        self._convo_message_count = 1
        force_deep = intent.kind == "brainstorm"
        await self._dispatch(message, force_priority=True, force_deep=force_deep)

    async def _handle_summary_request(self, message: GroupMessage) -> None:
        from tools.doc_generator import build_summary

        ideas = self.memory.get_ideas()
        history = self.memory.get_recent_summaries(limit=5)
        text = build_summary(history, ideas)
        spokesperson_id = "victor" if "victor" in self.agents else next(iter(self.agents))
        agent = self.agents[spokesperson_id]
        await self.deliver_burst(agent, [text], reply_to_message_id=message.telegram_message_id)

    # ------------------------------------------------------------------ dispatch
    async def _dispatch(
        self,
        message: GroupMessage,
        *,
        force_priority: bool = False,
        force_deep: bool = False,
    ) -> None:
        """Pick the most eager non-sender agent and schedule its response.

        ``force_priority`` is set when the message comes from god (or any
        message we want to guarantee gets answered): every evaluator's
        ``wants_to_respond`` is forced to ``True`` and ``urgency`` to 1.0
        before applying the turn-taking heuristics.
        """
        evaluators = [a for a in self.agents.values() if a.agent_id != message.sender_id]
        if not evaluators:
            return

        raw_evals = await asyncio.gather(
            *(
                a.evaluate_message(self.client, self.config.model_fast, message, self.memory)
                for a in evaluators
            ),
            return_exceptions=True,
        )

        candidates: list[tuple[Agent, Evaluation]] = []
        for agent, raw in zip(evaluators, raw_evals):
            if isinstance(raw, BaseException):
                log.warning("Eval crashed for %s: %s", agent.agent_id, raw)
                continue

            seed = raw
            if force_priority:
                seed = Evaluation(
                    wants_to_respond=True,
                    urgency=1.0,
                    estimated_delay=max(0.5, raw.estimated_delay),
                    is_rafaga=raw.is_rafaga,
                    rafaga_count=raw.rafaga_count,
                    needs_tools=raw.needs_tools or force_deep,
                    tool_to_use=raw.tool_to_use,
                    react_emoji=None,
                    reason="god priority",
                )

            adjusted = adjust_evaluation(agent, seed, message, self.memory)
            if adjusted.wants_to_respond:
                candidates.append((agent, adjusted))
            elif adjusted.react_emoji and self._react and message.telegram_message_id:
                await self._react(
                    agent.agent_id, message.telegram_message_id, adjusted.react_emoji
                )

        if not candidates:
            log.debug("Nobody wants to respond to %s", message.sender_id)
            return

        # Highest urgency wins; ties broken by shorter delay.
        candidates.sort(key=lambda pair: (-pair[1].urgency, pair[1].estimated_delay))
        chosen_agent, chosen_eval = candidates[0]
        runners_up = [c[0].agent_id for c in candidates[1:]] or "none"
        log.info(
            "Dispatch: %s wins (urgency=%.2f, delay=%.1fs) over %s",
            chosen_agent.agent_id,
            chosen_eval.urgency,
            chosen_eval.estimated_delay,
            runners_up,
        )
        self._schedule_response(chosen_agent, chosen_eval, force_deep=force_deep)

    # ------------------------------------------------------------------ scheduling
    def _schedule_response(
        self, agent: Agent, evaluation: Evaluation, *, force_deep: bool = False
    ) -> None:
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

    async def _run_response(
        self, agent: Agent, evaluation: Evaluation, *, force_deep: bool = False
    ) -> None:
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

    def _can_continue_conversation(self) -> bool:
        """True if the chain may continue: not silenced, under message cap, under API cap."""
        if self.is_silenced():
            return False
        if self._convo_message_count >= self.config.max_messages_per_convo:
            log.info(
                "Conversation cap reached (%d msgs) — chain stops",
                self._convo_message_count,
            )
            return False
        return self._can_call_api()

    def _is_conversation_stale(self) -> bool:
        """True if too much time passed since the last message to keep the context."""
        if not self._last_activity_at:
            return False
        if not self.memory.hot_messages(1):
            return False
        elapsed = (datetime.utcnow() - self._last_activity_at).total_seconds()
        return elapsed >= self.config.stale_gap_seconds

    async def _flush_stale_conversation(self) -> None:
        """Summarise the prior conversation in the background and clear hot context."""
        old_messages = self.memory.hot_messages(40)
        self.memory.clear_hot()
        self._convo_message_count = 0
        self._last_activity_at = None
        self._cancel_pending()
        if old_messages:
            asyncio.create_task(self._post_conversation_maintenance(old_messages))

    async def _post_conversation_maintenance(self, messages: list[GroupMessage]) -> None:
        """Summarise beliefs, then re-tune banned phrases — runs off the hot path."""
        try:
            async with self._summary_lock:
                for agent in self.agents.values():
                    await self.summariser.update_for_agent(agent.agent_id, agent.name, messages)
                self.tuner.run_for_all(self.agents.keys())
        except Exception:
            log.exception("Post-conversation maintenance failed")

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
            self.tuner.run_for_all(self.agents.keys())
            self.memory.clear_hot()
            self._convo_message_count = 0
            self._last_activity_at = None
