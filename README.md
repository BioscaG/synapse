# Synapse / RIALS — Telegram group with 3 autonomous AI agents

Three independent AI agents (Guido, Víctor, Jordi) live in a Telegram group and
behave like the real friends they are modelled after. Each agent has its own
Anthropic API call, its own memory, its own tools and its own opinion. When one
agent speaks, the other two evaluate the message in parallel and decide
independently whether to reply.

The human user that created the group is treated as the "god" of the group:
their messages have maximum priority and the bots react with deference.

> Bot output is **always in Spanish** (informal, WhatsApp-style). Code,
> comments, file names and documentation are in English.

## Tech stack

- Python 3.11+
- `python-telegram-bot` v21+ (async)
- `anthropic` Python SDK
- `asyncio` for parallel agent evaluation
- SQLite for persistent memory
- APScheduler for spontaneous conversations

### Models

- `claude-haiku-4-5-20251001` — ~90% of interactions (turn evaluation, normal
  replies, reactions).
- `claude-sonnet-4-20250514` — deep analysis, web search, document generation,
  brainstorm mode.

Estimated cost: $3-5 / month.

## Layout

```
synapse/
├── main.py                   # entry point
├── config.example.py         # copy to config.py and fill secrets
├── requirements.txt
│
├── agents/                   # 3 personalities + base class
├── orchestrator/             # turn heuristics, scheduler, god mode
├── memory/                   # SQLite-backed memory + summariser
├── tools/                    # web search, market analysis, etc.
├── telegram_bot/             # bot setup, handlers, actions
└── data/                     # SQLite db + initial memories JSON
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.py config.py    # edit with real tokens
python main.py
```

## Production

Run as a `systemd` service on a small VPS (Hetzner, DigitalOcean) or a
Raspberry Pi. The bots reconnect on transient failures with exponential
backoff, so a simple `Restart=always` unit is enough.
