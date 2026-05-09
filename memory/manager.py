"""Memory manager.

Two layers of memory:

- **Hot memory**: the last N messages of the group, kept in RAM as a deque.
  Lost on restart by design (matches the spec).
- **Cold memory**: per-agent beliefs, the shared idea board, conversation
  summaries and daily-usage counters. Persisted in SQLite.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from memory.models import init_db

log = logging.getLogger(__name__)

# Maximum messages kept in the hot window. Anthropic prompts only see the last
# 15, but we keep a few extra for re-evaluation when bursts arrive.
HOT_WINDOW = 40


@dataclass(slots=True)
class GroupMessage:
    """A single message in the group's hot memory."""

    sender_id: str          # 'guido', 'victor', 'jordi', or 'god'
    sender_name: str        # display name for the prompt
    text: str
    is_from_god: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)
    telegram_message_id: int | None = None
    # Optional image attachment (god can send a photo with caption). The
    # generation prompt will include the image as a content block; older
    # messages don't keep the image in hot memory beyond their own turn.
    image_b64: str | None = None
    image_media_type: str | None = None


class MemoryManager:
    """Coordinates hot and cold memory for the three agents."""

    def __init__(self, db_path: str, initial_memories_path: str | None = None) -> None:
        self.db_path = db_path
        self.initial_memories_path = initial_memories_path
        self._hot: deque[GroupMessage] = deque(maxlen=HOT_WINDOW)
        init_db(db_path)
        if initial_memories_path:
            self._seed_initial_memories(initial_memories_path)

    # ------------------------------------------------------------------ DB
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _seed_initial_memories(self, path: str) -> None:
        """Load initial memories on first run (no-op if any belief exists)."""
        with self._connect() as conn:
            existing = conn.execute("SELECT COUNT(*) FROM beliefs").fetchone()[0]
        if existing:
            return
        seed_path = Path(path)
        if not seed_path.exists():
            log.warning("Initial memories file not found at %s — skipping seed", path)
            return
        data = json.loads(seed_path.read_text(encoding="utf-8"))
        with self._connect() as conn:
            for agent_id, payload in data.items():
                for belief in payload.get("beliefs", []):
                    conn.execute(
                        """
                        INSERT INTO beliefs (agent_id, category, content, confidence, source)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            agent_id,
                            belief["category"],
                            belief["content"],
                            float(belief.get("confidence", 0.5)),
                            "initial_memories.json",
                        ),
                    )
            conn.commit()
        log.info("Seeded initial memories from %s", path)

    # ------------------------------------------------------------------ hot memory
    def push_message(self, message: GroupMessage) -> None:
        self._hot.append(message)
        # Persist bot messages so the StyleTuner can analyse them across
        # conversations. God messages stay RAM-only by design (the spec).
        if not message.is_from_god and message.sender_id in {"guido", "victor", "jordi"}:
            self._record_agent_message(message.sender_id, message.text)

    def clear_hot(self) -> None:
        """Empty the in-RAM context window (used when starting a new topic)."""
        self._hot.clear()

    def hot_messages(self, n: int = 15) -> list[GroupMessage]:
        if n >= len(self._hot):
            return list(self._hot)
        return list(self._hot)[-n:]

    def hot_size(self) -> int:
        return len(self._hot)

    def previous_speaker_excluding(self, sender_id: str) -> str | None:
        """Walk hot memory backwards and return the first sender that is neither
        ``sender_id`` nor god. Used to detect the implicit addressee of a question."""
        for msg in reversed(self._hot):
            if msg.sender_id == sender_id:
                continue
            if msg.is_from_god or msg.sender_id == "god":
                continue
            return msg.sender_id
        return None

    def format_context(self, n: int = 15) -> str:
        """Render the last `n` messages for inclusion in a prompt."""
        lines = []
        for msg in self.hot_messages(n):
            tag = "[DIOS]" if msg.is_from_god else f"[{msg.sender_name}]"
            lines.append(f"{tag} {msg.text}")
        return "\n".join(lines) if lines else "(grupo en silencio)"

    def messages_since(self, agent_id: str) -> int:
        """Number of messages since `agent_id` last spoke (large if never)."""
        count = 0
        for msg in reversed(self._hot):
            if msg.sender_id == agent_id:
                return count
            count += 1
        return count

    @property
    def last_speaker(self) -> str | None:
        return self._hot[-1].sender_id if self._hot else None

    def consecutive_count(self, agent_id: str) -> int:
        """How many of the most recent messages were from `agent_id` in a row."""
        count = 0
        for msg in reversed(self._hot):
            if msg.sender_id == agent_id:
                count += 1
            else:
                break
        return count

    # ------------------------------------------------------------------ cold memory: beliefs
    def get_beliefs(self, agent_id: str) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM beliefs WHERE agent_id = ? ORDER BY confidence DESC, updated_at DESC",
                    (agent_id,),
                )
            )

    def format_beliefs(self, agent_id: str) -> str:
        rows = self.get_beliefs(agent_id)
        if not rows:
            return "(sin memoria previa)"
        by_cat: dict[str, list[str]] = {}
        for row in rows:
            by_cat.setdefault(row["category"], []).append(
                f"- {row['content']} (conf {row['confidence']:.2f})"
            )
        out = []
        for cat, items in by_cat.items():
            out.append(f"## {cat}")
            out.extend(items)
        return "\n".join(out)

    def add_belief(
        self,
        agent_id: str,
        category: str,
        content: str,
        confidence: float = 0.5,
        source: str | None = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO beliefs (agent_id, category, content, confidence, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                (agent_id, category, content, confidence, source),
            )
            conn.commit()
            return cur.lastrowid

    def update_belief(self, belief_id: int, content: str | None = None, confidence: float | None = None) -> None:
        sets, params = [], []
        if content is not None:
            sets.append("content = ?")
            params.append(content)
        if confidence is not None:
            sets.append("confidence = ?")
            params.append(confidence)
        if not sets:
            return
        sets.append("updated_at = CURRENT_TIMESTAMP")
        params.append(belief_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE beliefs SET {', '.join(sets)} WHERE id = ?", params)
            conn.commit()

    def remove_beliefs(self, ids: Iterable[int]) -> None:
        ids = list(ids)
        if not ids:
            return
        with self._connect() as conn:
            placeholders = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM beliefs WHERE id IN ({placeholders})", ids)
            conn.commit()

    # ------------------------------------------------------------------ cold memory: ideas
    def upsert_idea(
        self,
        title: str,
        description: str | None = None,
        status: str = "proposed",
        proposed_by: str | None = None,
        kill_reason: str | None = None,
        killed_by: str | None = None,
    ) -> int:
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM ideas WHERE LOWER(title) = LOWER(?)",
                (title,),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE ideas
                    SET status = ?, kill_reason = COALESCE(?, kill_reason),
                        killed_by = COALESCE(?, killed_by),
                        description = COALESCE(?, description),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (status, kill_reason, killed_by, description, existing["id"]),
                )
                conn.commit()
                return existing["id"]
            cur = conn.execute(
                """
                INSERT INTO ideas (title, description, status, proposed_by, killed_by, kill_reason)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (title, description, status, proposed_by, killed_by, kill_reason),
            )
            conn.commit()
            return cur.lastrowid

    def get_ideas(self, statuses: Iterable[str] | None = None) -> list[sqlite3.Row]:
        with self._connect() as conn:
            if statuses:
                statuses = list(statuses)
                placeholders = ",".join("?" * len(statuses))
                return list(
                    conn.execute(
                        f"SELECT * FROM ideas WHERE status IN ({placeholders}) ORDER BY updated_at DESC",
                        statuses,
                    )
                )
            return list(conn.execute("SELECT * FROM ideas ORDER BY updated_at DESC"))

    # ------------------------------------------------------------------ cold memory: conversations
    def log_conversation(
        self,
        summary: str,
        topics: Iterable[str] | None = None,
        mood: str | None = None,
        pending_items: Iterable[str] | None = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO conversations (date, summary, topics, mood, pending_items)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    date.today().isoformat(),
                    summary,
                    json.dumps(list(topics or []), ensure_ascii=False),
                    mood,
                    json.dumps(list(pending_items or []), ensure_ascii=False),
                ),
            )
            conn.commit()
            return cur.lastrowid

    def get_pending_topics(self, limit: int = 10) -> list[str]:
        """Aggregate pending items from recent conversations."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT pending_items FROM conversations ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        seen, out = set(), []
        for row in rows:
            try:
                items = json.loads(row["pending_items"] or "[]")
            except json.JSONDecodeError:
                continue
            for item in items:
                if item and item not in seen:
                    seen.add(item)
                    out.append(item)
        return out

    def get_recent_summaries(self, limit: int = 5) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM conversations ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
            )

    # ------------------------------------------------------------------ cost / usage
    def _today_key(self) -> str:
        return date.today().isoformat()

    def record_api_call(self, tokens: int = 0) -> None:
        today = self._today_key()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO daily_usage (date, api_calls, tokens_used)
                VALUES (?, 1, ?)
                ON CONFLICT(date) DO UPDATE SET
                    api_calls = api_calls + 1,
                    tokens_used = tokens_used + excluded.tokens_used
                """,
                (today, tokens),
            )
            conn.commit()

    def record_spontaneous_convo(self) -> None:
        today = self._today_key()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO daily_usage (date, spontaneous_convos)
                VALUES (?, 1)
                ON CONFLICT(date) DO UPDATE SET
                    spontaneous_convos = spontaneous_convos + 1
                """,
                (today,),
            )
            conn.commit()

    def usage_today(self) -> dict[str, int]:
        today = self._today_key()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT api_calls, tokens_used, spontaneous_convos FROM daily_usage WHERE date = ?",
                (today,),
            ).fetchone()
        if not row:
            return {"api_calls": 0, "tokens_used": 0, "spontaneous_convos": 0}
        return {
            "api_calls": row["api_calls"],
            "tokens_used": row["tokens_used"],
            "spontaneous_convos": row["spontaneous_convos"],
        }

    # ------------------------------------------------------------------ tuner: agent message log
    def _record_agent_message(self, agent_id: str, text: str, max_keep: int = 800) -> None:
        """Persist a bot-emitted message and trim the per-agent log."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO agent_messages (agent_id, text) VALUES (?, ?)",
                (agent_id, text),
            )
            # Keep only the last `max_keep` messages per agent.
            conn.execute(
                """
                DELETE FROM agent_messages
                WHERE agent_id = ?
                  AND id NOT IN (
                    SELECT id FROM agent_messages
                    WHERE agent_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                  )
                """,
                (agent_id, agent_id, max_keep),
            )
            conn.commit()

    def get_recent_agent_messages(self, agent_id: str, limit: int = 100) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT text FROM agent_messages WHERE agent_id = ? ORDER BY id DESC LIMIT ?",
                (agent_id, limit),
            ).fetchall()
        return [row["text"] for row in rows]

    # ------------------------------------------------------------------ tuner: style feedback
    def set_overused_phrases(self, agent_id: str, phrases: Iterable[tuple[str, int]]) -> None:
        """Replace the agent's banned-phrases list."""
        phrases = list(phrases)
        with self._connect() as conn:
            conn.execute("DELETE FROM style_feedback WHERE agent_id = ?", (agent_id,))
            for phrase, occurrences in phrases:
                conn.execute(
                    """
                    INSERT INTO style_feedback (agent_id, phrase, occurrences)
                    VALUES (?, ?, ?)
                    """,
                    (agent_id, phrase, int(occurrences)),
                )
            conn.commit()

    def get_overused_phrases(self, agent_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT phrase FROM style_feedback
                WHERE agent_id = ?
                ORDER BY occurrences DESC, detected_at DESC
                """,
                (agent_id,),
            ).fetchall()
        return [row["phrase"] for row in rows]
