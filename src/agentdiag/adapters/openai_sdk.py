"""OpenAI Agents SDK trace adapter.

Parses traces from the OpenAI Agents SDK (formerly Swarm). Expected format:
a JSON object with a "steps" array where each step has "type" (e.g.
"model_response", "tool_call", "tool_output", "handoff") and related fields.
"""

from __future__ import annotations

from typing import Any

from agentdiag.adapters.base import BaseAdapter
from agentdiag.schema import Step, Trace


class OpenAIAgentsAdapter(BaseAdapter):
    """Adapter for OpenAI Agents SDK trace exports."""

    name = "openai_agents_sdk"

    @classmethod
    def can_handle(cls, raw: dict[str, Any]) -> bool:
        # OpenAI Agents SDK traces have "steps" with "type" values like
        # "model_response", "tool_call", "tool_output", "handoff"
        if "steps" not in raw or not isinstance(raw["steps"], list):
            return False
        if not raw["steps"]:
            return False
        first = raw["steps"][0]
        # Distinguish from our raw format: raw format steps have "index" field
        if "index" in first:
            return False
        openai_types = {"model_response", "tool_call", "tool_output", "handoff", "reasoning"}
        return first.get("type") in openai_types

    def parse(self, raw: dict[str, Any]) -> Trace:
        raw_steps: list[dict[str, Any]] = raw["steps"]
        steps: list[Step] = []
        index = 0
        available_tools: list[str] = []

        for raw_step in raw_steps:
            step_type = raw_step.get("type", "")

            if step_type == "model_response":
                content = raw_step.get("content", "")
                if isinstance(content, list):
                    # Content can be a list of message parts
                    content = " ".join(
                        p.get("text", str(p)) for p in content if isinstance(p, dict)
                    ) or str(content)
                steps.append(Step(
                    index=index,
                    type="thought",
                    content=content or "(model response)",
                ))
                index += 1

            elif step_type == "tool_call":
                tool_name = raw_step.get("name", raw_step.get("function", {}).get("name", "unknown"))
                tool_args = raw_step.get("arguments", raw_step.get("function", {}).get("arguments", {}))
                if isinstance(tool_args, str):
                    import json
                    try:
                        tool_args = json.loads(tool_args)
                    except (json.JSONDecodeError, TypeError):
                        tool_args = {"raw": tool_args}

                if tool_name not in available_tools:
                    available_tools.append(tool_name)

                steps.append(Step(
                    index=index,
                    type="tool_call",
                    content=f"Tool call: {tool_name}",
                    tool_name=tool_name,
                    tool_args=tool_args,
                ))
                index += 1

            elif step_type == "tool_output":
                output = raw_step.get("output", raw_step.get("content", ""))
                error = raw_step.get("error", False)
                if isinstance(error, str):
                    error = True
                steps.append(Step(
                    index=index,
                    type="observation",
                    content=str(output),
                    tool_result=str(output),
                    error=bool(error),
                ))
                index += 1

            elif step_type == "handoff":
                target = raw_step.get("target", raw_step.get("agent", "unknown"))
                steps.append(Step(
                    index=index,
                    type="thought",
                    content=f"Handoff to agent: {target}",
                ))
                index += 1

            elif step_type == "reasoning":
                steps.append(Step(
                    index=index,
                    type="thought",
                    content=raw_step.get("content", "(reasoning)"),
                ))
                index += 1

        task = raw.get("task", raw.get("input", "(unknown task)"))

        # Determine outcome
        has_error = any(s.error for s in steps)
        outcome = raw.get("outcome", "failure" if has_error else "unknown")

        return Trace(
            task=task,
            steps=steps,
            outcome=outcome,
            model=raw.get("model"),
            framework="openai_agents_sdk",
            available_tools=raw.get("available_tools", available_tools or None),
            metadata=raw.get("metadata", {}),
        )
