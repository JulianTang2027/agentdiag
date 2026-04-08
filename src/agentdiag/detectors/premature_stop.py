"""Premature Stop detector — identifies when an agent quits before task completion."""

from __future__ import annotations

from agentdiag.detectors.base import BaseDetector
from agentdiag.schema import Finding, Trace


class PrematureStopDetector(BaseDetector):
    """Detects when an agent stops without completing the task or hitting an error."""

    name = "PREMATURE_STOP"
    description = "Agent quits before completing the task"

    def detect(self, trace: Trace) -> list[Finding]:
        findings: list[Finding] = []

        if trace.outcome != "failure" or not trace.steps:
            return findings

        last_step = trace.steps[-1]

        # If the last step is a result (not an error observation), the agent
        # chose to stop rather than being forced to stop by an error
        if last_step.type == "result" and not last_step.error:
            # Check if there were recent errors that might explain giving up
            recent_errors = sum(
                1 for s in trace.steps[-4:] if s.type == "observation" and s.error
            )

            if recent_errors == 0:
                severity = "high"
                summary = (
                    "Agent returned a final result but the task outcome is 'failure'. "
                    "No recent errors were observed — the agent may have stopped prematurely."
                )
            else:
                severity = "medium"
                summary = (
                    "Agent returned a final result but the task outcome is 'failure'. "
                    f"There were {recent_errors} recent error(s) that may have caused "
                    "the agent to give up too early."
                )

            findings.append(
                Finding(
                    detector=self.name,
                    severity=severity,
                    step_range=(last_step.index, last_step.index),
                    summary=summary,
                    suggestion=(
                        "The agent should exhaust alternative approaches before giving up. "
                        "Consider adding persistence logic or fallback strategies so the "
                        "agent tries different tools or parameters before stopping."
                    ),
                )
            )

        return findings
