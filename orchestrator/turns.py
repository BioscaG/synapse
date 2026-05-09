"""Turn-taking heuristics.

These are pure helpers — no LLM calls — that adjust the *raw* evaluation each
agent returned. The numbers come from the real-chat statistics in the spec
(``RIALS_README.md``):

- Guido / Jordi reply more often to each other than to Víctor.
- Víctor's median gap between appearances is ~5.6 messages (he chimes in less).
- Burst-length distribution: Guido has the longest tails.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from agents.base import Evaluation

if TYPE_CHECKING:
    from agents.base import Agent
    from memory.manager import GroupMessage, MemoryManager

# Empirical "X replies to Y" frequencies from 25k-message dataset.
# Higher number = more likely to engage with that sender.
REPLY_AFFINITY = {
    "guido": {"jordi": 1.0, "victor": 0.72, "god": 1.0},
    "victor": {"guido": 1.0, "jordi": 0.93, "god": 1.0},
    "jordi": {"guido": 1.0, "victor": 0.66, "god": 1.0},
}

# Average gap (in messages) between own appearances. Used to nudge participation
# back when an agent has been silent for a while.
AVERAGE_GAP = {"guido": 4.1, "jordi": 4.6, "victor": 5.6}

# Probability that an agent stays quiet even when they want to talk — adds
# realistic randomness so the bots are not always all triggered together.
RANDOM_SILENCE_PROB = 0.10

# Probability of overriding the model's "no" when a message looks substantive.
OVERRIDE_PROBABILITY = 0.80

# Below this many messages in hot memory, every silent agent is force-included
# at least once. Guarantees a minimum 3-4 turn conversation after god speaks.
FRESH_CONVERSATION_DEPTH = 10


def is_reaction_message(text: str) -> bool:
    stripped = text.strip().lower()
    if not stripped:
        return True
    if len(stripped) <= 4:
        return True
    return stripped in {"xd", "xdd", "xddd", "bro", "yes", "no", "ya", "si", "sep", "vale", "rt"}


def is_substantive(text: str) -> bool:
    """Heuristic: longer-than-reaction messages are substantive in this group.

    The previous keyword-list approach was too narrow — it missed messages
    like "osea, automatizar respuestas a clientes, eso duele en el bolsillo"
    just because they didn't contain the word "idea". In a brainstorming
    group, anything with five+ words is something the others can build on.
    """
    stripped = text.strip()
    if len(stripped) < 25:
        return False
    if "?" in stripped:
        return True
    return len(stripped.split()) >= 5


EARLY_CONVERSATION_DEPTH = 10  # below this many hot messages, skip random silence
SCORE_NOISE_RANGE = (0.0, 0.55)  # softer threshold than uniform(0,1): score >=0.55 always passes


def adjust_evaluation(
    agent: "Agent",
    evaluation: Evaluation,
    message: "GroupMessage",
    memory: "MemoryManager",
) -> Evaluation:
    """Apply turn-taking heuristics on top of the model's raw evaluation."""
    is_question = "?" in message.text
    implicit_addressee = (
        memory.previous_speaker_excluding(message.sender_id) if is_question else None
    )
    is_implicitly_addressed = implicit_addressee == agent.agent_id

    # Force-engage every bot in a fresh conversation. If we're still in the
    # first few hot messages and this agent hasn't said anything yet, they
    # should chime in regardless of what the model said. Otherwise god's
    # opener gets one reply and the chain dies.
    fresh_convo_force = (
        memory.hot_size() < FRESH_CONVERSATION_DEPTH
        and memory.messages_since(agent.agent_id) >= memory.hot_size()
        and not message.is_from_god
        and message.sender_id != agent.agent_id
    )

    if not evaluation.wants_to_respond:
        # Override the model's "no" when:
        # - the agent hasn't spoken yet in a fresh conversation, OR
        # - the agent is the implicit addressee of a question, OR
        # - the message is substantive AND the agent isn't overexposed.
        should_override = (
            memory.last_speaker != agent.agent_id
            and memory.consecutive_count(agent.agent_id) < 3
            and (
                fresh_convo_force
                or is_implicitly_addressed
                or (is_substantive(message.text) and random.random() < OVERRIDE_PROBABILITY)
            )
        )
        if should_override:
            evaluation = Evaluation(
                wants_to_respond=True,
                urgency=(
                    0.9
                    if (fresh_convo_force or is_implicitly_addressed)
                    else 0.6
                ),
                estimated_delay=1.5 if is_implicitly_addressed else 2.0,
                is_rafaga=False,
                rafaga_count=1,
                needs_tools=False,
                tool_to_use=None,
                react_emoji=None,
                reason=(
                    "fresh conversation force-engage"
                    if fresh_convo_force
                    else "implicit addressee"
                    if is_implicitly_addressed
                    else "substantive message"
                ),
            )
        else:
            return evaluation

    score = evaluation.urgency or agent.config.get("response_probability_base", 0.5)

    text_lower = message.text.lower()
    if agent.name.lower() in text_lower or agent.agent_id in text_lower:
        score = max(score, agent.config.get("response_probability_mention", 0.95))

    if is_implicitly_addressed:
        # The previous-speaker-questioned signal is strong: someone is asking
        # this agent specifically even without naming them.
        score = max(score, 0.9)

    if is_question:
        score += 0.15

    consecutive = memory.consecutive_count(agent.agent_id)
    if consecutive >= 3:
        score -= 0.30 * (consecutive - 2)

    if memory.last_speaker == agent.agent_id:
        score -= 0.20

    msgs_since = memory.messages_since(agent.agent_id)
    if msgs_since > AVERAGE_GAP.get(agent.agent_id, 5):
        score += 0.20

    affinity = REPLY_AFFINITY.get(agent.agent_id, {}).get(message.sender_id, 0.85)
    score *= affinity

    if message.is_from_god:
        score = 1.0

    early_conversation = memory.hot_size() < EARLY_CONVERSATION_DEPTH
    if (
        random.random() < RANDOM_SILENCE_PROB
        and not message.is_from_god
        and not early_conversation
        and not is_implicitly_addressed
    ):
        return Evaluation.silent("random silence")

    if score <= random.uniform(*SCORE_NOISE_RANGE):
        return Evaluation.silent("score below threshold")

    delay = _calculate_delay(agent, evaluation, message)
    return Evaluation(
        wants_to_respond=True,
        urgency=min(1.0, score),
        estimated_delay=delay,
        is_rafaga=evaluation.is_rafaga,
        rafaga_count=evaluation.rafaga_count,
        needs_tools=evaluation.needs_tools,
        tool_to_use=evaluation.tool_to_use,
        react_emoji=evaluation.react_emoji,
        reason=evaluation.reason,
    )


def _calculate_delay(agent: "Agent", evaluation: Evaluation, message: "GroupMessage") -> float:
    base_min, base_max = agent.config.get("delay_base", (1.0, 3.0))
    delay = random.uniform(base_min, base_max)

    if message.is_from_god:
        delay *= 0.5

    text_lower = message.text.lower()
    if agent.name.lower() in text_lower or agent.agent_id in text_lower:
        # Mentioned by name → respond before the others. Aggressive multiplier
        # because base delays for slower agents (Víctor's 2-5s) need to drop
        # below faster agents' god-discounted base (Guido's 0.5-2s × 0.5).
        delay *= 0.15

    if is_reaction_message(message.text):
        delay *= 0.4

    if evaluation.needs_tools:
        delay += random.uniform(3.0, 8.0)

    delay *= random.uniform(0.7, 1.4)
    return max(0.3, delay)


def decide_burst(agent: "Agent", evaluation: Evaluation) -> int:
    """Pick the actual burst size, blending the model hint and config."""
    suggested = max(1, evaluation.rafaga_count)
    if suggested > 1:
        return suggested
    return agent.burst_size()
