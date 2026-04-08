"""Lightweight SDK for capturing agent traces and running diagnostics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from rich.console import Console

from agentdiag.detectors import run_all
from agentdiag.report import build_report, render_json, render_rich
from agentdiag.schema import DiagnosticReport, Step, Trace


class Tracer:
    """Capture agent execution steps and diagnose failures.

    Usage::

        from agentdiag import Tracer

        tracer = Tracer(task="Book a flight", model="gpt-4o")
        tracer.thought("I should search for flights")
        tracer.tool_call("search_flights", args={"origin": "NYC"})
        tracer.observation("Found 3 flights")
        tracer.result("Booked flight AA123")
        report = tracer.diagnose()

    Or as a context manager (auto-diagnoses on exit)::

        with Tracer(task="Book a flight") as t:
            t.tool_call("search", args={"q": "flights"})
            t.observation("results")
            t.result("done")
    """

    def __init__(
        self,
        task: str,
        model: str | None = None,
        available_tools: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._task = task
        self._model = model
        self._available_tools = available_tools
        self._metadata = metadata or {}
        self._steps: list[Step] = []
        self._outcome: Literal["success", "failure", "unknown"] = "unknown"
        self._index = 0

    def _add_step(self, **kwargs: Any) -> None:
        kwargs["index"] = self._index
        kwargs.setdefault("timestamp", datetime.now(timezone.utc))
        self._steps.append(Step(**kwargs))
        self._index += 1

    def thought(self, content: str) -> None:
        """Log a reasoning/thinking step."""
        self._add_step(type="thought", content=content)

    def tool_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        content: str | None = None,
    ) -> None:
        """Log a tool invocation."""
        self._add_step(
            type="tool_call",
            content=content or f"Tool call: {tool_name}",
            tool_name=tool_name,
            tool_args=args,
        )

    def observation(
        self,
        content: str,
        error: bool = False,
        tool_result: str | None = None,
    ) -> None:
        """Log a tool result or observation."""
        self._add_step(
            type="observation",
            content=content,
            error=error,
            tool_result=tool_result or content,
        )

    def result(self, content: str) -> None:
        """Log the agent's final answer."""
        self._add_step(type="result", content=content)

    def set_outcome(self, outcome: Literal["success", "failure", "unknown"]) -> None:
        """Explicitly set the trace outcome."""
        self._outcome = outcome

    def to_trace(self) -> Trace:
        """Convert captured steps into a Trace object."""
        return Trace(
            task=self._task,
            steps=list(self._steps),
            outcome=self._outcome,
            model=self._model,
            available_tools=self._available_tools,
            metadata=self._metadata,
        )

    def diagnose(
        self,
        format: str = "rich",
        console: Console | None = None,
    ) -> DiagnosticReport:
        """Run all detectors and return a diagnostic report.

        Args:
            format: "rich" for terminal output, "json" for JSON, "quiet" for no output.
            console: Optional Rich console to print to.
        """
        trace = self.to_trace()
        findings = run_all(trace)
        report = build_report(trace, findings)

        if format == "json":
            c = console or Console()
            c.print(render_json(report), highlight=False)
        elif format == "rich":
            c = console or Console()
            render_rich(report, c)
        # "quiet" — no output, just return the report

        return report

    def save(self, path: str | Path) -> None:
        """Export the trace as JSON."""
        trace = self.to_trace()
        data = trace.model_dump(mode="json")
        Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def __enter__(self) -> Tracer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.diagnose()
