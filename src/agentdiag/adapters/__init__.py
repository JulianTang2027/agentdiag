"""Adapter auto-detection and registry."""

from __future__ import annotations

from typing import Any

from agentdiag.adapters.base import BaseAdapter
from agentdiag.adapters.langchain import LangChainAdapter
from agentdiag.adapters.openai_sdk import OpenAIAgentsAdapter
from agentdiag.adapters.raw import RawAdapter
from agentdiag.schema import Trace

# Ordered by specificity — more specific adapters first, raw as fallback
ADAPTERS: list[type[BaseAdapter]] = [
    LangChainAdapter,
    OpenAIAgentsAdapter,
    RawAdapter,
]


def detect_and_parse(raw: dict[str, Any]) -> Trace:
    """Auto-detect the trace format and parse it into a universal Trace."""
    for adapter_cls in ADAPTERS:
        if adapter_cls.can_handle(raw):
            adapter = adapter_cls()
            return adapter.parse(raw)
    raise ValueError(
        "Could not detect trace format. Ensure your JSON has 'task' and 'steps' fields, "
        "or use a supported framework format (LangChain, OpenAI Agents SDK, CrewAI)."
    )
