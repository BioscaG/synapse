"""Real-API simulator.

Drives the actual ConversationManager with the actual Anthropic client (so we
get real Haiku responses) but with a *mocked* Telegram layer. Lets us judge
naturalness, repetition and conversation length on real model output without
running the live bot or sending messages to Telegram.

Run:  ``python scripts/simulate_real.py``

Cost: each scenario = ~20-50 Haiku calls. With prompt caching the input
tokens after the first call within a 5-min window are billed at ~10%, so a
full run of all scenarios is in the order of $0.05-$0.20.

Usage:
- Reads ``config.py`` from the project root (must already be set up).
- Loads ``data/initial_memories.json`` (read-only, into a temp DB).
- Runs each scenario in fresh state.
- Prints transcripts and an aggregate report:
    - total messages, per-agent counts
    - duplicate-line detection
    - n-gram repetition score per agent
    - average message length
    - first / last speaker
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import AsyncAnthropic

from agents import build_guido, build_jordi, build_victor
from memory.manager import GroupMessage, MemoryManager
from orchestrator.manager import ConversationManager, ManagerConfig
from orchestrator.tuner import find_overused_phrases

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("orchestrator.manager").setLevel(logging.INFO)


# --------------------------------------------------------------------------- transcript hook
class Transcript:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def append(self, agent_id: str, text: str) -> None:
        self.lines.append((agent_id, text))


def build_callbacks(transcript: Transcript):
    next_id = [1000]

    async def fake_send(agent_id: str, text: str, reply_to_message_id=None) -> int:
        transcript.append(agent_id, text)
        # Real-time print so we can watch convo unfold
        prefix = {"guido": "G", "victor": "V", "jordi": "J"}.get(agent_id, "?")
        print(f"  [{prefix}] {text}")
        next_id[0] += 1
        return next_id[0]

    async def fake_react(*args, **kwargs):
        return None

    async def fake_typing(*args, **kwargs):
        return None

    return fake_send, fake_react, fake_typing


# --------------------------------------------------------------------------- scenario runner
async def run_scenario(
    name: str,
    god_messages: list[str],
    *,
    api_key: str,
    model_fast: str,
    model_deep: str,
    initial_memories_path: str,
    wait_per_msg: float = 8.0,
) -> dict:
    print(f"\n{'='*78}\nSCENARIO: {name}\n{'='*78}")

    transcript = Transcript()
    send, react, typing = build_callbacks(transcript)

    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "sim.db")
        memory = MemoryManager(db_path=db, initial_memories_path=initial_memories_path)
        agents = {
            "guido": build_guido(),
            "victor": build_victor(),
            "jordi": build_jordi(),
        }
        client = AsyncAnthropic(api_key=api_key)
        manager = ConversationManager(
            agents=agents,
            memory=memory,
            client=client,
            config=ManagerConfig(
                model_fast=model_fast,
                model_deep=model_deep,
                max_daily_calls=10000,
                max_messages_per_convo=25,
                burst_inter_delay=(0.0, 0.0),  # no waits inside burst
                stale_gap_seconds=120,
            ),
            send_callback=send,
            react_callback=react,
            typing_callback=typing,
        )
        # Compress real-life delays so the sim runs in ~30s, not minutes.
        for a in agents.values():
            a.config = dict(a.config, delay_base=(0.05, 0.3), delay_rafaga=(0.0, 0.05))

        for msg in god_messages:
            print(f"  [DIOS] {msg}")
            gm = GroupMessage(
                sender_id="god",
                sender_name="DIOS",
                text=msg,
                is_from_god=True,
                telegram_message_id=random.randint(1, 999),
            )
            await manager.handle_message(gm)
            # Drain pending tasks (chain) — wait long enough for API calls.
            deadline = asyncio.get_event_loop().time() + wait_per_msg * 8
            while manager._pending_tasks and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.2)

        usage = memory.usage_today()

    return analyse(name, transcript.lines, usage)


# --------------------------------------------------------------------------- analysis
def analyse(name: str, lines: list[tuple[str, str]], usage: dict) -> dict:
    if not lines:
        return {"name": name, "total": 0, "lines": []}

    by_agent: dict[str, list[str]] = {"guido": [], "victor": [], "jordi": []}
    for agent_id, text in lines:
        by_agent.setdefault(agent_id, []).append(text)

    # Duplicate-line detection (same agent says exact same thing twice)
    duplicates: list[tuple[str, str]] = []
    for agent_id, texts in by_agent.items():
        seen = Counter(texts)
        for t, n in seen.items():
            if n > 1:
                duplicates.append((agent_id, f"{t!r} x{n}"))

    # N-gram repetition score per agent (on this convo only; the real tuner
    # runs over ~80 messages across conversations so it's stricter).
    overused: dict[str, list[tuple[str, int]]] = {}
    for agent_id, texts in by_agent.items():
        if len(texts) >= 3:
            overused[agent_id] = find_overused_phrases(texts, min_count=2, top_k=4)
        else:
            overused[agent_id] = []

    avg_len = sum(len(t) for _, t in lines) / len(lines)

    counts = {a: len(by_agent.get(a, [])) for a in ("guido", "victor", "jordi")}
    silent = [a for a, c in counts.items() if c == 0]

    return {
        "name": name,
        "total": len(lines),
        "by_agent": counts,
        "silent": silent,
        "first_speaker": lines[0][0] if lines else None,
        "duplicates": duplicates,
        "overused": overused,
        "avg_msg_len": round(avg_len, 1),
        "tokens": usage["tokens_used"],
        "api_calls": usage["api_calls"],
    }


# --------------------------------------------------------------------------- main
SCENARIOS = [
    ("S1: brainstorm idea genérica", ["Vamos a pensar en una idea de negocio"]),
    ("S2: dios solo dice hola", ["hola chicos que tal"]),
    ("S3: dios dirige a Jordi", ["jordi piensa en una idea de negocio va"]),
    ("S4: pregunta abierta", ["que opinais de las criptos ahora mismo"]),
    ("S5: dios habla dos veces seguidas", [
        "chicos vamos a pensar una idea",
        "algo para sellers de ecommerce mejor",
    ]),
    ("S6: dios pide brainstorm explícito", [
        "investigad bien una idea de app movil que mole",
    ]),
    ("S7: dios cuenta algo personal", [
        "tio he tenido un dia de mierda en el curro",
    ]),
]


def print_report(results: list[dict]) -> None:
    print("\n\n" + "=" * 78)
    print("REPORT")
    print("=" * 78)
    for r in results:
        flag = ""
        silent = r.get("silent") or []
        if silent:
            flag += " ⚠️silent=" + ",".join(silent)
        if r.get("total", 0) < 4:
            flag += " ⚠️short"
        dups = r.get("duplicates") or []
        if dups:
            flag += f" ⚠️dups={len(dups)}"
        print(
            f"\n{r['name']}\n"
            f"  total={r.get('total', 0):2d}  by_agent={r.get('by_agent', {})}  "
            f"avg_len={r.get('avg_msg_len', 0)}  api_calls={r.get('api_calls', 0)}{flag}"
        )
        for who, line in dups[:3]:
            print(f"    DUP {who}: {line}")
        for who, phrases in (r.get("overused") or {}).items():
            if phrases:
                print(f"    overuse {who}: " + ", ".join(f"{p!r}×{c}" for p, c in phrases))


async def main() -> None:
    try:
        import config  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        print("config.py not found", file=sys.stderr)
        sys.exit(1)

    base_dir = Path(__file__).parent.parent
    initial = str(base_dir / config.INITIAL_MEMORIES_PATH)

    results = []
    for i, (name, msgs) in enumerate(SCENARIOS):
        if i > 0:
            # Short cooldown between scenarios so we don't burst into the
            # Anthropic per-minute rate limit during a full sweep.
            await asyncio.sleep(8.0)
        random.seed(hash(name) % 1000)
        try:
            r = await run_scenario(
                name,
                msgs,
                api_key=config.ANTHROPIC_API_KEY,
                model_fast=config.MODEL_FAST,
                model_deep=config.MODEL_DEEP,
                initial_memories_path=initial,
            )
            results.append(r)
        except Exception as e:
            print(f"  scenario failed: {e}")
            results.append({"name": name, "total": 0, "by_agent": {}, "silent": ["all"], "duplicates": [], "overused": {}, "avg_msg_len": 0, "tokens": 0, "api_calls": 0, "first_speaker": None})

    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())
