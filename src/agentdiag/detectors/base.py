"""Base detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentdiag.schema import Finding, Trace


class BaseDetector(ABC):
    """Abstract base class for failure detectors."""

    name: str
    description: str

    @abstractmethod
    def detect(self, trace: Trace) -> list[Finding]:
        """Analyze a trace and return any findings."""
        ...
