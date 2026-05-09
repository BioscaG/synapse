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

# Words that flag a message as "substantive group fodder": when present, the
# heuristic will override a conservative ``respond=false`` from the model with
# a coin flip, because in this group an idea/business/plan always sparks debate.
SUBSTANTIVE_KEYWORDS = (
    "idea",
    "negocio",
    "startup",
    "plugin",
    " app ",
    " app?",
    " app.",
    "proyecto",
    "monto",
    "montar",
    "monetiza",
    "mercado",
    "competencia",
    "invert",
    "invierto",
    "inversi",
    "ai ",
    " ia ",
    " ia,",
    " ia.",
    "modelo de",
    "lanzar",
    "vender",
    "comprar",
)

OVERRIDE_PROBABILITY = 0.65


def is_reaction_message(text: str) -> bool:
    stripped = text.strip().lower()
    if not stripped:
        return True
    if len(stripped) <= 4:
        return True
    return stripped in {"xd", "xdd", "xddd", "bro", "yes", "no", "ya", "si", "sep", "vale", "rt"}


def is_substantive(text: str) -> bool:
    """Heuristic: looks like content the group would naturally debate."""
    lower = text.lower()
    if len(text) < 25:
        return False
    if "?" in text:
        return True
    return any(kw in lower for kw in SUBSTANTIVE_KEYWORDS)


def adjust_evaluation(
    agent: "Agent",
    evaluation: Evaluation,
    message: "GroupMessage",
    memory: "MemoryManager",
) -> Evaluation:
    """Apply turn-taking heuristics on top of the model's raw evaluation."""
    if not evaluation.wants_to_respond:
        # Override an over-conservative "no" when the message is substantive
        # AND the agent isn't already overexposed. Mention-checking + cooldown
        # keep this from spamming.
        if (
            is_substantive(message.text)
            and memory.last_speaker != agent.agent_id
            and memory.consecutive_count(agent.agent_id) < 3
            and random.random() < OVERRIDE_PROBABILITY
        ):
            evaluation = Evaluation(
                wants_to_respond=True,
                urgency=0.6,
                estimated_delay=2.0,
                is_rafaga=False,
                rafaga_count=1,
                needs_tools=False,
                tool_to_use=None,
                react_emoji=None,
                reason="heuristic override: substantive message",
            )
        else:
            return evaluation

    score = evaluation.urgency or agent.config.get("response_probability_base", 0.5)

    text_lower = message.text.lower()
    if agent.name.lower() in text_lower or agent.agent_id in text_lower:
        score = max(score, agent.config.get("response_probability_mention", 0.95))

    if "?" in message.text:
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

    if random.random() < RANDOM_SILENCE_PROB and not message.is_from_god:
        return Evaluation.silent("random silence")

    if score <= random.random():
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
