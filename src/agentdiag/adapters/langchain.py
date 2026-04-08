"""LangChain trace adapter.

Parses traces exported from LangChain's callback system or LangSmith.
Expected format: a JSON object with a top-level "runs" array, where each run
has fields like "name", "run_type", "inputs", "outputs", "error", etc.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agentdiag.adapters.base import BaseAdapter
from agentdiag.schema import Step, Trace


class LangChainAdapter(BaseAdapter):
    """Adapter for LangChain callback/LangSmith trace exports."""

    name = "langchain"

    @classmethod
    def can_handle(cls, raw: dict[str, Any]) -> bool:
        # LangChain traces typically have "runs" with "run_type" fields
        if "runs" in raw and isinstance(raw["runs"], list):
            if raw["runs"] and "run_type" in raw["runs"][0]:
                return True
        return False

    def parse(self, raw: dict[str, Any]) -> Trace:
        runs: list[dict[str, Any]] = raw["runs"]
        steps: list[Step] = []
        index = 0
        available_tools: list[str] = []

        for run in runs:
            run_type = run.get("run_type", "")
            name = run.get("name", "")
            inputs = run.get("inputs", {})
            outputs = run.get("outputs", {})
            error = run.get("error") is not None
            timestamp = self._parse_timestamp(run.get("start_time"))

            if run_type == "llm":
                # LLM call — treat as a thought step
                output_text = self._extract_text(outputs)
                steps.append(Step(
                    index=index,
                    type="thought",
                    content=output_text or f"LLM call: {name}",
                    error=error,
                    timestamp=timestamp,
                ))
                index += 1

            elif run_type == "tool":
                available_tools.append(name)
                # Tool invocation
                tool_input = inputs.get("input", inputs.get("query", str(inputs)))
                steps.append(Step(
                    index=index,
                    type="tool_call",
                    content=f"Tool call: {name}",
                    tool_name=name,
                    tool_args=inputs if isinstance(inputs, dict) else {"input": inputs},
                    error=False,
                    timestamp=timestamp,
                ))
                index += 1

                # Tool result as observation
                tool_output = self._extract_text(outputs)
                steps.append(Step(
                    index=index,
                    type="observation",
                    content=tool_output or "(no output)",
                    tool_result=tool_output,
                    error=error,
                    timestamp=timestamp,
                ))
                index += 1

            elif run_type == "chain":
                # Top-level or sub-chain — extract if it has meaningful output
                output_text = self._extract_text(outputs)
                if output_text and not any(r.get("run_type") in ("llm", "tool") for r in runs):
                    steps.append(Step(
                        index=index,
                        type="result",
                        content=output_text,
                        error=error,
                        timestamp=timestamp,
                    ))
                    index += 1

        # Determine task from the first chain's input
        task = raw.get("task", "")
        if not task:
            for run in runs:
                if run.get("run_type") == "chain":
                    inp = run.get("inputs", {})
                    task = inp.get("input", inp.get("question", inp.get("query", "")))
                    if task:
                        break
        if not task:
            task = "(unknown task)"

        # Determine outcome
        last_run = runs[-1] if runs else {}
        has_error = any(r.get("error") is not None for r in runs)
        outcome = "failure" if has_error else "success"

        # Deduplicate available_tools
        unique_tools = list(dict.fromkeys(available_tools)) or None

        return Trace(
            task=task,
            steps=steps,
            outcome=raw.get("outcome", outcome),
            model=raw.get("model"),
            framework="langchain",
            available_tools=raw.get("available_tools", unique_tools),
            metadata=raw.get("metadata", {}),
        )

    @staticmethod
    def _extract_text(data: Any) -> str:
        """Pull a readable string out of LangChain output dicts."""
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for key in ("output", "text", "result", "content", "answer"):
                if key in data:
                    val = data[key]
                    return val if isinstance(val, str) else str(val)
            # Try generations format
            if "generations" in data:
                gens = data["generations"]
                if gens and isinstance(gens[0], list) and gens[0]:
                    return gens[0][0].get("text", str(gens[0][0]))
        return str(data) if data else ""

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
