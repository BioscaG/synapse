"""Tool registry: maps a tool name to the actual coroutine.

Agents only emit a tool *name* (string) in their evaluation. The orchestrator
asks the registry to resolve and invoke the right helper. Centralising the
mapping keeps the agent code free of imports of every individual tool module.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from anthropic import AsyncAnthropic

from tools.market_analysis import run_market_analysis
from tools.tech_estimator import run_tech_estimator
from tools.web_search import run_web_search

log = logging.getLogger(__name__)

ToolFn = Callable[[AsyncAnthropic, str, str], Awaitable[str | None]]


class ToolRegistry:
    """Resolve and invoke tools by name."""

    def __init__(self, client: AsyncAnthropic, model: str) -> None:
        self.client = client
        self.model = model
        self._tools: dict[str, ToolFn] = {
            "web_search": run_web_search,
            "market_analysis": run_market_analysis,
            "tech_estimator": run_tech_estimator,
        }

    def has(self, name: str) -> bool:
        return name in self._tools

    async def run(self, name: str, context: str) -> str | None:
        fn = self._tools.get(name)
        if not fn:
            log.warning("Unknown tool requested: %s", name)
            return None
        return await fn(self.client, self.model, context)
