"""AgentDiag — Diagnose why AI agents fail."""

__version__ = "0.1.0"

from agentdiag.instrument.anthropic import watch_anthropic
from agentdiag.instrument.openai import watch_openai
from agentdiag.tracer import Tracer

__all__ = ["Tracer", "watch_openai", "watch_anthropic", "__version__"]
