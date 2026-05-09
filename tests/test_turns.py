"""Turn-taking heuristics."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

from agents import build_guido, build_jordi, build_victor
from agents.base import Evaluation
from memory.manager import GroupMessage, MemoryManager
from orchestrator.turns import (
    adjust_evaluation,
    is_reaction_message,
    is_substantive,
)


INITIAL_MEMORIES = str(Path(__file__).parent.parent / "data" / "initial_memories.json")


def _fresh_memory() -> MemoryManager:
    tmpdir = tempfile.mkdtemp()
    return MemoryManager(db_path=os.path.join(tmpdir, "test.db"), initial_memories_path=INITIAL_MEMORIES)


def test_is_reaction_short_text():
    assert is_reaction_message("xd")
    assert is_reaction_message("ya")
    assert is_reaction_message("")
    assert is_reaction_message("vale")
    assert not is_reaction_message("vamos a pensar una idea de negocio")


def test_is_substantive_long_with_keyword():
    assert is_substantive("vamos a pensar en una idea de negocio")
    assert is_substantive("y q tal lo del plugin de wallapop???")
    assert not is_substantive("xd")
    assert not is_substantive("vale gracias")


def test_god_message_forced_through_heuristics():
    random.seed(0)
    mem = _fresh_memory()
    god_msg = GroupMessage(
        sender_id="god", sender_name="DIOS", text="hola chicos", is_from_god=True
    )
    mem.push_message(god_msg)
    guido = build_guido()
    seed = Evaluation(
        wants_to_respond=True,
        urgency=0.5,
        estimated_delay=2.0,
        is_rafaga=False,
        rafaga_count=1,
        needs_tools=False,
        tool_to_use=None,
        react_emoji=None,
        reason="seed",
    )
    pass_count = 0
    for _ in range(50):
        adjusted = adjust_evaluation(guido, seed, god_msg, mem)
        if adjusted.wants_to_respond:
            pass_count += 1
    assert pass_count >= 48, f"god boost should pass nearly always, got {pass_count}/50"


def test_implicit_addressee_from_question():
    """When a bot asks a question right after another bot spoke, the asked-of
    bot is force-engaged even if the model returned respond=false."""
    random.seed(1)
    mem = _fresh_memory()
    mem.push_message(GroupMessage(sender_id="god", sender_name="DIOS", text="vamos a pensar idea", is_from_god=True))
    mem.push_message(GroupMessage(sender_id="guido", sender_name="Guido", text="tengo una idea pendiente"))
    mem.push_message(GroupMessage(sender_id="jordi", sender_name="Jordi", text="bro y q idea es???"))

    guido = build_guido()
    silent = Evaluation.silent("model said no")
    msg = mem.hot_messages(1)[0]
    pass_count = 0
    for _ in range(50):
        adjusted = adjust_evaluation(guido, silent, msg, mem)
        if adjusted.wants_to_respond:
            pass_count += 1
    assert pass_count == 50, "implicit addressee must always pass"


def test_fresh_convo_force_engages_silent_agent():
    """An agent that hasn't spoken yet in a fresh conversation should be
    force-engaged even when the model says no."""
    random.seed(2)
    mem = _fresh_memory()
    mem.push_message(GroupMessage(sender_id="god", sender_name="DIOS", text="vamos a pensar idea", is_from_god=True))
    mem.push_message(GroupMessage(sender_id="guido", sender_name="Guido", text="osea estaba pensando algo de ecommerce"))

    victor = build_victor()  # has not spoken
    silent = Evaluation.silent("model said no")
    msg = mem.hot_messages(1)[0]
    pass_count = sum(adjust_evaluation(victor, silent, msg, mem).wants_to_respond for _ in range(50))
    assert pass_count >= 49, f"fresh-convo force should always pass, got {pass_count}/50"


def test_last_speaker_penalty_blocks_immediate_repeat():
    """An agent who just spoke shouldn't immediately respond to themselves —
    last_speaker penalty + score cutoff should mostly suppress."""
    random.seed(3)
    mem = _fresh_memory()
    mem.push_message(GroupMessage(sender_id="god", sender_name="DIOS", text="hola", is_from_god=True))
    mem.push_message(GroupMessage(sender_id="guido", sender_name="Guido", text="ola buenas q tal todo"))
    # Bump hot size above EARLY_CONVERSATION_DEPTH so random_silence_prob applies.
    for i in range(12):
        mem.push_message(GroupMessage(sender_id="jordi", sender_name="Jordi", text=f"jord msg {i} medianamente largo"))
        mem.push_message(GroupMessage(sender_id="guido", sender_name="Guido", text=f"gui msg {i} medianamente largo"))

    guido = build_guido()
    seed = Evaluation(
        wants_to_respond=True,
        urgency=0.5,
        estimated_delay=2.0,
        is_rafaga=False,
        rafaga_count=1,
        needs_tools=False,
        tool_to_use=None,
        react_emoji=None,
        reason="seed",
    )
    msg = mem.hot_messages(1)[0]  # last speaker is guido
    pass_count = sum(adjust_evaluation(guido, seed, msg, mem).wants_to_respond for _ in range(200))
    # last_speaker penalty (-0.20) + score noise check should drop pass-rate.
    assert pass_count < 130, f"last-speaker penalty should drop guido below 65%, got {pass_count}/200"


def test_named_in_text_boosts_score():
    random.seed(4)
    mem = _fresh_memory()
    god_msg = GroupMessage(
        sender_id="god",
        sender_name="DIOS",
        text="jordi piensa una idea de negocio va",
        is_from_god=True,
    )
    mem.push_message(god_msg)

    jordi = build_jordi()
    silent = Evaluation.silent("model said no")
    pass_count = sum(
        adjust_evaluation(jordi, silent, god_msg, mem).wants_to_respond for _ in range(50)
    )
    # Even with model saying no, named addressee on god message must pass.
    assert pass_count >= 48, f"named-jordi god message should always pass, got {pass_count}/50"
