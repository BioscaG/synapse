"""Example configuration. Copy to ``config.py`` and fill in real secrets.

The real ``config.py`` is git-ignored so it never ends up in the repo.
"""

from __future__ import annotations

# === API keys ===
ANTHROPIC_API_KEY = "sk-ant-..."

# === Telegram bot tokens (get them from @BotFather) ===
TELEGRAM_TOKENS = {
    "guido": "TOKEN_GUIDO",
    "victor": "TOKEN_VICTOR",
    "jordi": "TOKEN_JORDI",
}

# === Telegram group ===
# ID of the group where the three bots live.
GROUP_CHAT_ID = -1000000000000
# Telegram user ID of the human "god" of the group.
GOD_USER_ID = 0

# === Models ===
# Fast, cheap model used for ~90% of interactions (turn evaluation, normal replies).
MODEL_FAST = "claude-haiku-4-5-20251001"
# Deeper model used for analysis, web search, document generation, brainstorm mode.
MODEL_DEEP = "claude-sonnet-4-20250514"

# === Cost / safety limits ===
MAX_SPONTANEOUS_CONVOS_PER_DAY = 3
MAX_MESSAGES_PER_CONVO = 25
# Each bot turn consumes roughly 3 Haiku calls (2 evaluations from the
# non-sender bots + 1 generation). 1000 covers ~10 conversations of 25 msgs,
# leaving plenty of headroom for spontaneous chats and belief updates.
MAX_DAILY_API_CALLS = 1000
SPONTANEOUS_MSG_BUDGET = 20

# === Timing (seconds) ===
DELAY_GUIDO_BASE = (0.5, 2.0)
DELAY_VICTOR_BASE = (2.0, 5.0)
DELAY_JORDI_BASE = (1.0, 3.0)
DELAY_RAFAGA = (0.3, 0.8)

# === Active hours (local time, 24h) ===
ACTIVE_HOURS = (9, 24)
PEAK_HOURS = (17, 23)

# === Storage ===
DB_PATH = "data/rials.db"
INITIAL_MEMORIES_PATH = "data/initial_memories.json"

# === Logging ===
LOG_LEVEL = "INFO"
