"""Raw JSON adapter — parses traces already in the AgentDiag universal format."""

from __future__ import annotations

from typing import Any

from agentdiag.adapters.base import BaseAdapter
from agentdiag.schema import Trace


class RawAdapter(BaseAdapter):
    """Adapter for traces already in the AgentDiag universal format."""

    name = "raw"

    def parse(self, raw: dict[str, Any]) -> Trace:
        return Trace.model_validate(raw)

    @classmethod
    def can_handle(cls, raw: dict[str, Any]) -> bool:
        return "task" in raw and "steps" in raw
