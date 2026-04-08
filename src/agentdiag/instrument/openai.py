"""Auto-instrumentation for the OpenAI Python SDK."""

from __future__ import annotations

import json
from functools import wraps
from typing import Any

from agentdiag.tracer import Tracer


def watch_openai(
    client: Any,
    task: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[Any, Tracer]:
    """Wrap an OpenAI client to automatically capture agent traces.

    Usage::

        from openai import OpenAI
        from agentdiag import watch_openai

        client = OpenAI()
        client, tracer = watch_openai(client, task="Book a flight")

        # Use client normally — all calls are auto-traced
        response = client.chat.completions.create(...)

        tracer.diagnose()

    Args:
        client: An OpenAI client instance.
        task: Description of what the agent is trying to do.
        metadata: Optional metadata dict.

    Returns:
        Tuple of (client, tracer). The client is the same object with
        chat.completions.create wrapped.
    """
    tracer = Tracer(task=task, metadata=metadata)
    _seen_tool_message_count = [0]  # mutable counter in closure

    original_create = client.chat.completions.create

    @wraps(original_create)
    def wrapped_create(*args: Any, **kwargs: Any) -> Any:
        # --- Extract metadata on first call ---
        model = kwargs.get("model")
        if model and tracer._model is None:
            tracer._model = model

        # Extract available tools from tools param
        tools = kwargs.get("tools")
        if tools and tracer._available_tools is None:
            tool_names = []
            for t in tools:
                if isinstance(t, dict):
                    func = t.get("function", {})
                    name = func.get("name") if isinstance(func, dict) else None
                    if name:
                        tool_names.append(name)
            if tool_names:
                tracer._available_tools = tool_names

        # --- Capture tool results from input messages ---
        messages = kwargs.get("messages", args[0] if args else [])
        if isinstance(messages, list):
            tool_messages = [m for m in messages if _get_role(m) == "tool"]
            new_count = len(tool_messages)
            if new_count > _seen_tool_message_count[0]:
                for m in tool_messages[_seen_tool_message_count[0]:]:
                    content = _get_content(m)
                    is_error = _looks_like_error(content)
                    tracer.observation(content, error=is_error)
                _seen_tool_message_count[0] = new_count

        # --- Call the real API ---
        response = original_create(*args, **kwargs)

        # --- Capture from response ---
        choice = response.choices[0] if response.choices else None
        if choice is None:
            return response

        msg = choice.message

        if msg.tool_calls:
            # Model is reasoning + calling tools
            if msg.content:
                tracer.thought(msg.content)
            for tc in msg.tool_calls:
                tc_args = tc.function.arguments
                if isinstance(tc_args, str):
                    try:
                        tc_args = json.loads(tc_args)
                    except (json.JSONDecodeError, TypeError):
                        tc_args = {"raw": tc_args}
                tracer.tool_call(tc.function.name, args=tc_args)
        else:
            # Final answer — no tool calls
            if msg.content:
                tracer.result(msg.content)

        return response

    client.chat.completions.create = wrapped_create
    return client, tracer


def _get_role(message: Any) -> str:
    """Get role from a message (dict or object)."""
    if isinstance(message, dict):
        return message.get("role", "")
    return getattr(message, "role", "")


def _get_content(message: Any) -> str:
    """Get content from a message (dict or object)."""
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _looks_like_error(content: str) -> bool:
    """Heuristic: does this tool result look like an error?"""
    lower = content.lower()
    return any(kw in lower for kw in ("error", "exception", "failed", "timeout", "403", "404", "500", "rate limit"))
