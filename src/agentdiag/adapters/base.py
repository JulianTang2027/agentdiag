"""Base adapter interface for converting framework traces to universal format."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentdiag.schema import Trace


class BaseAdapter(ABC):
    """Abstract base class for trace adapters."""

    name: str

    @abstractmethod
    def parse(self, raw: dict[str, Any]) -> Trace:
        """Convert a raw JSON trace into the universal Trace format."""
        ...

    @classmethod
    def can_handle(cls, raw: dict[str, Any]) -> bool:
        """Return True if this adapter can parse the given raw JSON."""
        return False
