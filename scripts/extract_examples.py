"""Extract real-message bursts from a WhatsApp export.

Parses ``chat.txt`` (the original WhatsApp export the personalities are
modelled after), groups consecutive messages from the same sender into
"bursts", and prints a curated sample per agent. The output is meant to be
hand-picked into ``agents/{guido,victor,jordi}.py`` as few-shot examples.

Usage:  ``python scripts/extract_examples.py``

The script is read-only and prints to stdout — it never modifies the agent
files automatically. The user (or future me) hand-picks which bursts to use.
"""

from __future__ import annotations

import random
import re
import sys
from collections import defaultdict
from pathlib import Path

CHAT_PATH = Path(__file__).parent.parent / "chat.txt"

# Map WhatsApp display names -> agent ids.
SENDER_MAP = {
    "Guido": "guido",
    "Víctor UPC": "victor",
    "Jordi UPC": "jordi",
}

LINE_RE = re.compile(
    r"\[(?P<date>\d{2}/\d{2}/\d{4}), (?P<time>\d{2}:\d{2}:\d{2})\] (?P<sender>[^:]+): (?P<text>.+)$"
)

# Skip messages that are media placeholders / system events.
SKIP_PATTERNS = (
    "Messages and calls are end-to-end encrypted",
    "This message was deleted",
    "This message was edited",
    "<Media omitted>",
    "image omitted",
    "audio omitted",
    "video omitted",
    "document omitted",
    "sticker omitted",
    "GIF omitted",
    "added you",
    "created group",
    "changed the subject",
    "joined using",
    "left",
)


def parse() -> list[tuple[str, str, str]]:
    """Return list of (timestamp, sender_id, text)."""
    if not CHAT_PATH.exists():
        print(f"chat.txt not found at {CHAT_PATH}", file=sys.stderr)
        sys.exit(1)
    out: list[tuple[str, str, str]] = []
    for line in CHAT_PATH.read_text(encoding="utf-8").splitlines():
        m = LINE_RE.match(line)
        if not m:
            # Continuation of a previous multiline message — append to last.
            if out:
                ts, agent, text = out[-1]
                out[-1] = (ts, agent, text + " " + line.strip())
            continue
        sender = m.group("sender")
        agent = SENDER_MAP.get(sender)
        if not agent:
            continue
        text = m.group("text").strip()
        if any(pat in text for pat in SKIP_PATTERNS):
            continue
        if not text:
            continue
        out.append((f"{m.group('date')} {m.group('time')}", agent, text))
    return out


def group_bursts(messages: list[tuple[str, str, str]], gap_seconds: int = 60) -> list[tuple[str, list[str]]]:
    """Group consecutive messages from the same sender into bursts.

    A burst ends when a different sender speaks. We also split a single
    sender's long stream when there's a >gap_seconds pause.
    """
    bursts: list[tuple[str, list[str]]] = []
    current_sender: str | None = None
    current_texts: list[str] = []
    for _, sender, text in messages:
        if sender == current_sender:
            current_texts.append(text)
        else:
            if current_sender and current_texts:
                bursts.append((current_sender, current_texts))
            current_sender = sender
            current_texts = [text]
    if current_sender and current_texts:
        bursts.append((current_sender, current_texts))
    return bursts


def pick_examples(bursts: list[tuple[str, list[str]]], agent: str, n: int = 6) -> list[list[str]]:
    """Pick diverse, illustrative bursts for an agent.

    Heuristics:
    - At least 2 messages in the burst (boring single replies excluded).
    - At most 6 messages (avoid super-long monologues).
    - Total characters between 30 and 280 (representative size).
    - Prefer bursts that contain at least one of the agent's known catchphrases.
    """
    catchphrases = {
        "guido": ["a ver", "osea", "creo q", "tmb", "esq", "en plan", "BROO", "xddd", "hshs", "mi gozo"],
        "victor": ["Bro", "Ósea", "creo que", "ns", "Hahahah", "Jeje", "Sep", "Rt"],
        "jordi": ["xd", "ns", "es q", "creo q", "bueno", "pues", "jajja", "broooo", "victor", "guido"],
    }
    keys = catchphrases.get(agent, [])

    candidates = []
    for sender, texts in bursts:
        if sender != agent:
            continue
        if not (2 <= len(texts) <= 6):
            continue
        total = sum(len(t) for t in texts)
        if not (30 <= total <= 280):
            continue
        score = sum(1 for t in texts for k in keys if k.lower() in t.lower())
        candidates.append((score, texts))

    # Sort by catchphrase richness, take a diverse top-N
    candidates.sort(key=lambda x: (-x[0], -len(x[1])))
    seen_first_words = set()
    picked: list[list[str]] = []
    for _, texts in candidates:
        first = texts[0].split()[0].lower() if texts[0].split() else ""
        if first in seen_first_words and len(picked) >= 2:
            continue
        seen_first_words.add(first)
        picked.append(texts)
        if len(picked) >= n:
            break
    return picked


def main() -> None:
    random.seed(0)
    messages = parse()
    print(f"Total messages: {len(messages)}")
    by_agent = defaultdict(int)
    for _, a, _ in messages:
        by_agent[a] += 1
    print(f"Per agent: {dict(by_agent)}\n")

    bursts = group_bursts(messages)
    print(f"Total bursts: {len(bursts)}\n")

    for agent in ("guido", "victor", "jordi"):
        print("=" * 78)
        print(f"AGENT: {agent}")
        print("=" * 78)
        picks = pick_examples(bursts, agent, n=6)
        for i, texts in enumerate(picks, 1):
            print(f"\nEXAMPLE {i}:")
            for t in texts:
                print(f"  > {t}")
        print()


if __name__ == "__main__":
    main()
