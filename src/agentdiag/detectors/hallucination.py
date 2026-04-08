"""Hallucinated Success detector — identifies when an agent claims success but failed."""

from __future__ import annotations

import re

from agentdiag.detectors.base import BaseDetector
from agentdiag.schema import Finding, Trace

SUCCESS_PATTERNS = [
    r"\bdone\b",
    r"\bcompleted?\b",
    r"\bsuccessful(?:ly)?\b",
    r"\bfinished\b",
    r"\bhere(?:'s| is| are) (?:the|your)\b",
    r"\btask (?:is )?(?:done|complete)\b",
]

SUCCESS_RE = re.compile("|".join(SUCCESS_PATTERNS), re.IGNORECASE)


class HallucinatedSuccessDetector(BaseDetector):
    """Detects when an agent claims success in its final message but the outcome is failure."""

    name = "HALLUCINATED_SUCCESS"
    description = "Agent claims task is done but outcome is failure"

    def detect(self, trace: Trace) -> list[Finding]:
        findings: list[Finding] = []

        if trace.outcome != "failure" or not trace.steps:
            return findings

        # Check the last result step
        result_steps = [s for s in trace.steps if s.type == "result"]
        if not result_steps:
            return findings

        last_result = result_steps[-1]

        if SUCCESS_RE.search(last_result.content):
            findings.append(
                Finding(
                    detector=self.name,
                    severity="high",
                    step_range=(last_result.index, last_result.index),
                    summary=(
                        "Agent's final response suggests task completion "
                        f"(\"{last_result.content[:80]}...\") but the trace outcome "
                        "is 'failure'."
                    ),
                    suggestion=(
                        "Add output validation to verify the agent actually accomplished "
                        "the task before claiming success. Consider checking for concrete "
                        "deliverables (e.g., a booking confirmation, a file written, "
                        "a query result) rather than trusting the agent's self-assessment."
                    ),
                )
            )

        return findings
