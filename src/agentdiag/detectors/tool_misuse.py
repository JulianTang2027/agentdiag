"""Tool Misuse detector — identifies calls to nonexistent or wrong tools."""

from __future__ import annotations

from agentdiag.detectors.base import BaseDetector
from agentdiag.schema import Finding, Trace


class ToolMisuseDetector(BaseDetector):
    """Detects when an agent calls tools that aren't in the available tools list."""

    name = "TOOL_MISUSE"
    description = "Agent calls nonexistent or unavailable tools"

    def detect(self, trace: Trace) -> list[Finding]:
        findings: list[Finding] = []

        if trace.available_tools is None:
            return findings

        available = set(trace.available_tools)

        for step in trace.steps:
            if step.type != "tool_call" or step.tool_name is None:
                continue

            if step.tool_name not in available:
                findings.append(
                    Finding(
                        detector=self.name,
                        severity="high",
                        step_range=(step.index, step.index),
                        summary=(
                            f"Agent called '{step.tool_name}' which is not in the "
                            f"available tools: {sorted(available)}."
                        ),
                        suggestion=(
                            f"Ensure the agent's system prompt lists available tools clearly. "
                            f"Consider adding few-shot examples of correct tool names. "
                            f"Available tools: {', '.join(sorted(available))}."
                        ),
                    )
                )

        return findings
