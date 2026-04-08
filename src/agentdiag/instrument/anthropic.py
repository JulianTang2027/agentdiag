"""Auto-instrumentation for the Anthropic Python SDK."""

from __future__ import annotations

import json
from functools import wraps
from typing import Any

from agentdiag.tracer import Tracer


def watch_anthropic(
    client: Any,
    task: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[Any, Tracer]:
    """Wrap an Anthropic client to automatically capture agent traces.

    Usage::

        from anthropic import Anthropic
        from agentdiag import watch_anthropic

        client = Anthropic()
        client, tracer = watch_anthropic(client, task="Summarize emails")

        response = client.messages.create(...)

        tracer.diagnose()

    Args:
        client: An Anthropic client instance.
        task: Description of what the agent is trying to do.
        metadata: Optional metadata dict.

    Returns:
        Tuple of (client, tracer).
    """
    tracer = Tracer(task=task, metadata=metadata)
    _seen_tool_result_count = [0]

    original_create = client.messages.create

    @wraps(original_create)
    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        # --- Extract metadata ---
        model = kwargs.get("model")
        if model and tracer._model is None:
            tracer._model = model

        # Extract available tools
        tools = kwargs.get("tools")
        if tools and tracer._available_tools is None:
            tool_names = []
            for t in tools:
                if isinstance(t, dict):
                    name = t.get("name")
                    if name:
                        tool_names.append(name)
            if tool_names:
                tracer._available_tools = tool_names

        # --- Capture tool results from input messages ---
        messages = kwargs.get("messages", [])
        if isinstance(messages, list):
            tool_results = []
            for m in messages:
                if not isinstance(m, dict):
                    continue
                if m.get("role") != "user":
                    continue
                content = m.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            tool_results.append(block)

            new_count = len(tool_results)
            if new_count > _seen_tool_result_count[0]:
                for block in tool_results[_seen_tool_result_count[0]:]:
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_content = " ".join(
                            b.get("text", str(b))
                            for b in result_content
                            if isinstance(b, dict)
                        ) or str(result_content)
                    is_error = block.get("is_error", False) or _looks_like_error(str(result_content))
                    tracer.observation(str(result_content), error=is_error)
                _seen_tool_result_count[0] = new_count

        # --- Call the real API ---
        response = original_create(*args, **kwargs)

        # --- Capture from response ---
        # Anthropic responses have content blocks: text, tool_use
        content_blocks = getattr(response, "content", [])
        has_tool_use = any(
            getattr(b, "type", None) == "tool_use" for b in content_blocks
        )

        for block in content_blocks:
            block_type = getattr(block, "type", None)

            if block_type == "text":
                text = getattr(block, "text", "")
                if has_tool_use:
                    # Reasoning before tool calls
                    if text.strip():
                        tracer.thought(text)
                else:
                    # Final answer
                    if text.strip():
                        tracer.result(text)

            elif block_type == "tool_use":
                name = getattr(block, "name", "unknown")
                tool_input = getattr(block, "input", {})
                tracer.tool_call(name, args=tool_input if isinstance(tool_input, dict) else {})

        return response

    client.messages.create = wrapped_create
    return client, tracer


def _looks_like_error(content: str) -> bool:
    """Heuristic: does this tool result look like an error?"""
    lower = content.lower()
    return any(kw in lower for kw in ("error", "exception", "failed", "timeout", "403", "404", "500", "rate limit"))
