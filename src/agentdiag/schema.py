"""Universal trace schema for agent diagnostics."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Step(BaseModel):
    """A single step in an agent's execution trace."""

    index: int
    type: Literal["thought", "tool_call", "observation", "result"]
    content: str
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None
    error: bool = False
    timestamp: datetime | None = None


class Trace(BaseModel):
    """A complete agent execution trace."""

    task: str = Field(description="What the agent was trying to do")
    steps: list[Step]
    outcome: Literal["success", "failure", "unknown"] = "unknown"
    model: str | None = None
    framework: str | None = None
    available_tools: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Finding(BaseModel):
    """A diagnostic finding from a detector."""

    detector: str = Field(description="Detector name, e.g. LOOP")
    severity: Literal["high", "medium", "low"]
    step_range: tuple[int, int] = Field(
        description="Start and end step indices (inclusive)"
    )
    summary: str = Field(description="Human-readable explanation")
    suggestion: str = Field(description="Actionable fix suggestion")


class DiagnosticReport(BaseModel):
    """Complete diagnostic report for a trace."""

    trace: Trace
    findings: list[Finding] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
