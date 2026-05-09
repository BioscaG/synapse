"""Memory manager: hot deque, cold tables, helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from memory.manager import GroupMessage, MemoryManager


INITIAL_MEMORIES = str(Path(__file__).parent.parent / "data" / "initial_memories.json")


def _fresh() -> MemoryManager:
    tmpdir = tempfile.mkdtemp()
    return MemoryManager(db_path=os.path.join(tmpdir, "test.db"), initial_memories_path=INITIAL_MEMORIES)


def test_initial_memories_seeded_once():
    mem = _fresh()
    assert len(mem.get_beliefs("guido")) >= 5
    # Re-seeding should be a no-op (idempotent on first-run only).
    mem._seed_initial_memories(INITIAL_MEMORIES)
    assert len(mem.get_beliefs("guido")) >= 5


def test_hot_push_and_clear():
    mem = _fresh()
    mem.push_message(GroupMessage(sender_id="god", sender_name="DIOS", text="hola", is_from_god=True))
    mem.push_message(GroupMessage(sender_id="guido", sender_name="Guido", text="ey"))
    assert mem.hot_size() == 2
    mem.clear_hot()
    assert mem.hot_size() == 0


def test_consecutive_count_resets_when_other_speaks():
    mem = _fresh()
    mem.push_message(GroupMessage(sender_id="guido", sender_name="Guido", text="m1"))
    mem.push_message(GroupMessage(sender_id="guido", sender_name="Guido", text="m2"))
    mem.push_message(GroupMessage(sender_id="guido", sender_name="Guido", text="m3"))
    assert mem.consecutive_count("guido") == 3
    mem.push_message(GroupMessage(sender_id="jordi", sender_name="Jordi", text="ya"))
    assert mem.consecutive_count("guido") == 0
    assert mem.consecutive_count("jordi") == 1


def test_messages_since():
    mem = _fresh()
    mem.push_message(GroupMessage(sender_id="guido", sender_name="Guido", text="m1"))
    mem.push_message(GroupMessage(sender_id="jordi", sender_name="Jordi", text="m2"))
    mem.push_message(GroupMessage(sender_id="victor", sender_name="Víctor", text="m3"))
    assert mem.messages_since("guido") == 2
    assert mem.messages_since("victor") == 0
    assert mem.messages_since("nobody") == 3  # never spoke → equal to hot_size


def test_previous_speaker_excluding_skips_god():
    mem = _fresh()
    mem.push_message(GroupMessage(sender_id="god", sender_name="DIOS", text="g1", is_from_god=True))
    mem.push_message(GroupMessage(sender_id="guido", sender_name="Guido", text="m1"))
    mem.push_message(GroupMessage(sender_id="god", sender_name="DIOS", text="g2", is_from_god=True))
    mem.push_message(GroupMessage(sender_id="jordi", sender_name="Jordi", text="m2 q tal??"))
    # Looking at the question from Jordi, the implicit addressee should be Guido (not god).
    assert mem.previous_speaker_excluding("jordi") == "guido"


def test_agent_messages_persisted_for_bot_only():
    mem = _fresh()
    mem.push_message(GroupMessage(sender_id="god", sender_name="DIOS", text="hola", is_from_god=True))
    mem.push_message(GroupMessage(sender_id="guido", sender_name="Guido", text="bro hola"))
    mem.push_message(GroupMessage(sender_id="victor", sender_name="Víctor", text="Bro q tal"))
    assert mem.get_recent_agent_messages("guido") == ["bro hola"]
    assert mem.get_recent_agent_messages("victor") == ["Bro q tal"]
    # God's message must NOT have been logged.
    assert mem.get_recent_agent_messages("god") == []


def test_overused_phrases_roundtrip():
    mem = _fresh()
    mem.set_overused_phrases("victor", [("validemos esto", 5), ("hazlo 100", 3)])
    assert mem.get_overused_phrases("victor") == ["validemos esto", "hazlo 100"]
    # Setting again should replace, not append
    mem.set_overused_phrases("victor", [("nuevo bro", 4)])
    assert mem.get_overused_phrases("victor") == ["nuevo bro"]


def test_usage_today_increments():
    mem = _fresh()
    assert mem.usage_today()["api_calls"] == 0
    mem.record_api_call(tokens=100)
    mem.record_api_call(tokens=50)
    u = mem.usage_today()
    assert u["api_calls"] == 2
    assert u["tokens_used"] == 150
