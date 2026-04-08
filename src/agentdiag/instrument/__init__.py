"""Auto-instrumentation for AI frameworks."""

from agentdiag.instrument.anthropic import watch_anthropic
from agentdiag.instrument.openai import watch_openai

__all__ = ["watch_openai", "watch_anthropic"]
