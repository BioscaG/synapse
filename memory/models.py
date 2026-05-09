"""SQLite schema for the memory layer.

The schema follows the spec in ``RIALS_README.md``:

- ``beliefs``: per-agent cold memory (opinions, knowledge, relationships).
- ``ideas``: shared idea board across the three agents.
- ``conversations``: short summaries of past conversations (NOT full message logs).
- ``daily_usage``: per-day counters for cost / safety limits.
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
"""


def init_db(db_path: str) -> None:
    """Create the database file (if missing) and apply the schema."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()
