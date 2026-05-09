"""Conversation simulator.

Drives the real ConversationManager / agents / heuristics with a *mocked*
Anthropic client so we can iterate on turn-taking and prompt logic without
spending real tokens or running Telegram.

Run: ``python scripts/simulate.py``

The mock client returns plausible JSON for both evaluate_message and
generate_response calls based on simple rules:

- evaluate_message: respond=true with high urgency if the agent is mentioned
  by name OR a question is in the text; otherwise 60% respond=true.
- generate_response: returns a canned burst for the agent based on the last
  message it's reacting to.

The point is to drive the manager through realistic call sequences and
verify the chain stays alive across multiple bot turns.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import build_guido, build_jordi, build_victor
from memory.manager import MemoryManager, GroupMessage
from orchestrator.manager import ConversationManager, ManagerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sim")
logging.getLogger("orchestrator.manager").setLevel(logging.INFO)
logging.getLogger("orchestrator.tuner").setLevel(logging.WARNING)


# --------------------------------------------------------------------------- mock anthropic
CANNED_BURSTS: dict[str, list[list[str]]] = {
    "guido": [
        ["a ver", "me he estado currando una idea estos dias", "es un plugin de chrome para ecommerce"],
        ["pero esq mola", "podriamos sacarle pasta facil"],
        ["BROOO si funciona"],
        ["mi gozo en un pozo"],
        ["nono", "tmb pero igual sale"],
        ["q mal lo explico en texto 😭"],
    ],
    "victor": [
        ["Bro pero ya hay 3 plugins así en chrome web store"],
        ["Ósea bien pero hay que ver márgenes"],
        ["Hace cuanto no validáis algo bro"],
        ["Mira esto no tiene moat"],
        ["Yo lo pondría a 9eur al mes minimo"],
    ],
    "jordi": [
        ["bro y q idea es??? me das spoiler???"],
        ["esto en un finde lo monto xd", "con FastAPI y la API de wallapop"],
        ["pero eso no era una liada???"],
        ["100000% bro", "LO HAGO ENTONCES"],
        ["ns me la pela ahora", "estoy con el TFG"],
        ["jajja menuda mierda no????"],
    ],
}

EVAL_LOG: list[tuple[str, str, dict]] = []


def _eval_response(agent_name: str, message_text: str, beliefs: str = "") -> dict:
    text = message_text.lower()
    name = agent_name.lower()

    # If god speaks, always respond.
    if "[dios]" in message_text:
        return {
            "respond": True,
            "urgency": 1.0,
            "delay_seconds": random.randint(1, 3),
            "rafaga": random.randint(1, 3),
            "need_search": False,
            "tool": None,
            "react_emoji": None,
            "reason": "god",
        }

    # Mention by name.
    if name in text:
        return {
            "respond": True,
            "urgency": 0.9,
            "delay_seconds": random.randint(1, 4),
            "rafaga": random.randint(1, 2),
            "need_search": False,
            "tool": None,
            "react_emoji": None,
            "reason": "mentioned",
        }

    # Question — moderate engagement.
    if "?" in message_text:
        if random.random() < 0.6:
            return {
                "respond": True,
                "urgency": 0.6,
                "delay_seconds": random.randint(1, 5),
                "rafaga": 1,
                "need_search": False,
                "tool": None,
                "react_emoji": None,
                "reason": "question",
            }

    # Default: 40% engaged. This deliberately mimics the conservative
    # behaviour we've been seeing from the real model so the heuristics get
    # exercised.
    if random.random() < 0.4:
        return {
            "respond": True,
            "urgency": 0.55,
            "delay_seconds": random.randint(2, 6),
            "rafaga": 1,
            "need_search": False,
            "tool": None,
            "react_emoji": None,
            "reason": "engaged",
        }
    return {
        "respond": False,
        "urgency": 0.0,
        "delay_seconds": 0,
        "rafaga": 1,
        "need_search": False,
        "tool": None,
        "react_emoji": None,
        "reason": "skip",
    }


class MockAnthropic:
    """Pretends to be ``AsyncAnthropic`` for the duration of the simulation."""

    def __init__(self) -> None:
        self.messages = self  # so client.messages.create works
        self._burst_idx: dict[str, int] = {"guido": 0, "victor": 0, "jordi": 0}

    async def create(self, *, model: str, system=None, messages=None, **kwargs) -> SimpleNamespace:
        # Recover the agent name from the cached system block.
        sys_text = ""
        if isinstance(system, list):
            sys_text = "\n".join(b.get("text", "") for b in system)
        elif isinstance(system, str):
            sys_text = system

        user_text = ""
        if messages and isinstance(messages, list) and messages:
            content = messages[0].get("content", "")
            if isinstance(content, str):
                user_text = content

        agent_name = _detect_agent(sys_text)

        # evaluate_message → JSON dict
        if "Decide y devuelve SOLO el JSON" in user_text or "decidir si quieres responder" in sys_text:
            new_msg = _extract_new_message(user_text)
            payload = _eval_response(agent_name, new_msg)
            EVAL_LOG.append((agent_name, new_msg[:60], payload))
            text = json.dumps(payload)
        # generate_response / spontaneous → JSON array of strings
        else:
            agent_id = agent_name.lower().replace("í", "i")
            bursts = CANNED_BURSTS.get(agent_id, [["[mock burst]"]])
            idx = self._burst_idx.get(agent_id, 0) % len(bursts)
            self._burst_idx[agent_id] = idx + 1
            text = json.dumps(bursts[idx])

        return SimpleNamespace(
            content=[SimpleNamespace(text=text)],
            usage=SimpleNamespace(output_tokens=80, input_tokens=300, cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )


def _detect_agent(system_text: str) -> str:
    for name in ("Guido", "Víctor", "Jordi"):
        if f"Eres {name}" in system_text:
            return name
    return "Guido"


def _extract_new_message(user_text: str) -> str:
    marker = "Nuevo mensaje a evaluar:"
    if marker in user_text:
        rest = user_text.split(marker, 1)[1]
        # take until next double newline
        return rest.split("\n\n", 1)[0].strip()
    return user_text[:120]


# --------------------------------------------------------------------------- transcript hook
TRANSCRIPT: list[tuple[str, str]] = []


async def fake_send(agent_id: str, text: str, reply_to_message_id: int | None = None) -> int:
    TRANSCRIPT.append((agent_id, text))
    print(f"  [{agent_id:6}] {text}")
    return random.randint(1000, 9999)


async def fake_react(*args, **kwargs) -> None:
    return None


async def fake_typing(agent_id: str) -> None:
    return None


# --------------------------------------------------------------------------- scenarios
async def run_scenario(name: str, god_messages: list[str]) -> dict:
    """Run a single conversation scenario and return aggregate metrics."""
    print(f"\n{'='*78}\nSCENARIO: {name}\n{'='*78}")
    TRANSCRIPT.clear()
    EVAL_LOG.clear()

    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "sim.db")
        memory = MemoryManager(db_path=db, initial_memories_path="data/initial_memories.json")
        agents = {
            "guido": build_guido(),
            "victor": build_victor(),
            "jordi": build_jordi(),
        }
        client = MockAnthropic()
        manager = ConversationManager(
            agents=agents,
            memory=memory,
            client=client,  # type: ignore[arg-type]
            config=ManagerConfig(
                model_fast="mock-haiku",
                model_deep="mock-sonnet",
                max_daily_calls=10000,
                max_messages_per_convo=25,
                burst_inter_delay=(0.0, 0.0),  # zero waits in sim
                stale_gap_seconds=120,
            ),
            send_callback=fake_send,
            react_callback=fake_react,
            typing_callback=fake_typing,
        )
        # Patch base-delay so each bot runs near-instantly during sim.
        for a in agents.values():
            a.config = dict(a.config, delay_base=(0.0, 0.05), delay_rafaga=(0.0, 0.0))

        for msg in god_messages:
            print(f"  [GOD   ] {msg}")
            gm = GroupMessage(
                sender_id="god",
                sender_name="DIOS",
                text=msg,
                is_from_god=True,
                telegram_message_id=random.randint(1, 999),
            )
            await manager.handle_message(gm)
            # Allow scheduled tasks to drain
            for _ in range(50):
                if not manager._pending_tasks:
                    break
                await asyncio.sleep(0.02)
            # Wait one more round in case chain spawned new tasks at the very end
            await asyncio.sleep(0.05)
            for _ in range(50):
                if not manager._pending_tasks:
                    break
                await asyncio.sleep(0.02)

    speakers = [s for s, _ in TRANSCRIPT]
    by_agent = {a: speakers.count(a) for a in ("guido", "victor", "jordi")}
    return {
        "name": name,
        "total_messages": len(TRANSCRIPT),
        "by_agent": by_agent,
        "evals": len(EVAL_LOG),
        "no_say": [a for a, c in by_agent.items() if c == 0],
    }


async def main() -> None:
    random.seed(7)
    results = []
    results.append(await run_scenario(
        "S1: god opens with idea brainstorm",
        ["Vamos a pensar en una idea para hacer"],
    ))
    results.append(await run_scenario(
        "S2: god says hello (short)",
        ["hola chicos"],
    ))
    results.append(await run_scenario(
        "S3: god directs at Jordi",
        ["jordi piensa en una idea de negocio va"],
    ))
    results.append(await run_scenario(
        "S4: god asks open question",
        ["que opinais de las criptos"],
    ))
    results.append(await run_scenario(
        "S5: two consecutive god messages",
        ["chicos vamos a pensar una idea", "algo para sellers de ecommerce"],
    ))

    print("\n\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for r in results:
        flag = " ⚠️" if r["no_say"] or r["total_messages"] < 3 else ""
        print(
            f"  {r['name']:55s}  msgs={r['total_messages']:2d}  "
            f"{r['by_agent']}  silent={r['no_say']}{flag}"
        )


if __name__ == "__main__":
    asyncio.run(main())
