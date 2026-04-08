"""Recovery Failure detector — identifies errors followed by no strategy change."""

from __future__ import annotations

from agentdiag.detectors.base import BaseDetector
from agentdiag.schema import Finding, Trace


class RecoveryFailureDetector(BaseDetector):
    """Detects when an agent encounters an error but doesn't change its approach."""

    name = "RECOVERY_FAILURE"
    description = "Agent fails to adapt after encountering an error"

    def detect(self, trace: Trace) -> list[Finding]:
        findings: list[Finding] = []
        steps = trace.steps

        for i, step in enumerate(steps):
            if not (step.type == "observation" and step.error):
                continue

            # Find the tool_call that caused this error (look backward)
            failed_call = None
            for j in range(i - 1, -1, -1):
                if steps[j].type == "tool_call":
                    failed_call = steps[j]
                    break

            if failed_call is None:
                continue

            # Find the next tool_call after this error (look forward)
            next_call = None
            for k in range(i + 1, len(steps)):
                if steps[k].type == "tool_call":
                    next_call = steps[k]
                    break

            if next_call is None:
                continue

            # Check if the agent changed strategy
            same_tool = failed_call.tool_name == next_call.tool_name
            same_args = failed_call.tool_args == next_call.tool_args

            if same_tool and same_args:
                findings.append(
                    Finding(
                        detector=self.name,
                        severity="high",
                        step_range=(failed_call.index, next_call.index),
                        summary=(
                            f"Agent retried '{failed_call.tool_name}' with identical "
                            f"arguments after receiving an error at step {step.index}."
                        ),
                        suggestion=(
                            "After an error, the agent should change its approach: "
                            "modify parameters, try a different tool, add backoff/retry "
                            "logic, or gracefully report the failure."
                        ),
                    )
                )

        return findings
