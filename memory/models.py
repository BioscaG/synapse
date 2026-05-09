"""SQLite schema for the memory layer.

The schema follows the spec in ``RIALS_README.md`` plus the StyleTuner
extension:

- ``beliefs``: per-agent cold memory (opinions, knowledge, relationships).
- ``ideas``: shared idea board across the three agents.
- ``conversations``: short summaries of past conversations (NOT full message logs).
- ``daily_usage``: per-day counters for cost / safety limits.
- ``agent_messages``: rolling log of bot-emitted messages, used by the
  StyleTuner to detect overused phrases across conversations.
- ``style_feedback``: per-agent list of phrases the StyleTuner has flagged as
  overused. The generation prompt reads this list and asks the agent to vary
  its phrasing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS beliefs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    category TEXT NOT NULL,
    content TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    source TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_beliefs_agent ON beliefs(agent_id);
CREATE INDEX IF NOT EXISTS idx_beliefs_category ON beliefs(agent_id, category);

CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'proposed',
    proposed_by TEXT,
    killed_by TEXT,
    kill_reason TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ideas_status ON ideas(status);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    summary TEXT NOT NULL,
    topics TEXT,
    mood TEXT,
    pending_items TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_conversations_date ON conversations(date);

CREATE TABLE IF NOT EXISTS daily_usage (
    date TEXT PRIMARY KEY,
    api_calls INTEGER NOT NULL DEFAULT 0,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    spontaneous_convos INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS agent_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_agent_messages_agent ON agent_messages(agent_id, id DESC);

CREATE TABLE IF NOT EXISTS style_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    phrase TEXT NOT NULL,
    occurrences INTEGER NOT NULL DEFAULT 1,
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(agent_id, phrase)
);

CREATE INDEX IF NOT EXISTS idx_style_feedback_agent ON style_feedback(agent_id);
"""


def init_db(db_path: str) -> None:
    """Create the database file (if missing) and apply the schema."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()
