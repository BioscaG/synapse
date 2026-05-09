"""Agent definitions: base class and three personalities."""

from agents.base import Agent
from agents.guido import build_guido
from agents.jordi import build_jordi
from agents.victor import build_victor

__all__ = ["Agent", "build_guido", "build_victor", "build_jordi"]
