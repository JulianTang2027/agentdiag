"""Loop detector — identifies repeated identical or similar tool calls."""

from __future__ import annotations

from difflib import SequenceMatcher

from agentdiag.detectors.base import BaseDetector
from agentdiag.schema import Finding, Trace

# Minimum consecutive identical/similar tool calls to flag
MIN_REPEAT_COUNT = 3
# Similarity threshold for "nearly identical" tool calls
SIMILARITY_THRESHOLD = 0.85


class LoopDetector(BaseDetector):
    """Detects when an agent repeats the same or very similar tool call multiple times."""

    name = "LOOP"
    description = "Repeated identical or near-identical tool calls"

    def detect(self, trace: Trace) -> list[Finding]:
        findings: list[Finding] = []
        tool_calls = [s for s in trace.steps if s.type == "tool_call"]

        if len(tool_calls) < MIN_REPEAT_COUNT:
            return findings

        i = 0
        while i < len(tool_calls):
            run_start = i
            run_end = i

            for j in range(i + 1, len(tool_calls)):
                if self._is_similar(tool_calls[i], tool_calls[j]):
                    run_end = j
                else:
                    break

            run_length = run_end - run_start + 1
            if run_length >= MIN_REPEAT_COUNT:
                start_idx = tool_calls[run_start].index
                end_idx = tool_calls[run_end].index
                findings.append(
                    Finding(
                        detector=self.name,
                        severity="high",
                        step_range=(start_idx, end_idx),
                        summary=(
                            f"Agent repeated '{tool_calls[i].tool_name}' "
                            f"{run_length} times with identical or near-identical arguments."
                        ),
                        suggestion=(
                            "Add deduplication logic or force parameter variation after "
                            "failed attempts. Consider adding backoff, trying alternative "
                            "tools, or aborting after N retries."
                        ),
                    )
                )
                i = run_end + 1
            else:
                i += 1

        return findings

    @staticmethod
    def _is_similar(a, b) -> bool:
        """Check if two tool call steps are identical or near-identical."""
        if a.tool_name != b.tool_name:
            return False
        # Exact match on args
        if a.tool_args == b.tool_args:
            return True
        # Fuzzy match on stringified args
        str_a = str(a.tool_args or {})
        str_b = str(b.tool_args or {})
        return SequenceMatcher(None, str_a, str_b).ratio() >= SIMILARITY_THRESHOLD
