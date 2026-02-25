"""
Hook/Trigger System (OpenClaw P3.4)
====================================

Event-based extension points for the pipeline. Handlers are registered
per event type and executed in priority order with error isolation.

Usage:
    from core.hooks import register_hook, emit_hook

    # Register a handler
    def on_spec_created(context):
        print(f"New spec: {context['spec_dir']}")

    register_hook("spec:created", on_spec_created, priority=10)

    # Emit from pipeline
    await emit_hook("spec:created", {"spec_dir": "/path/to/spec"})
"""

from __future__ import annotations

import asyncio
import copy
import sys
from collections import defaultdict
from typing import Any, Callable


class HookRegistry:
    """Registry for event handlers with priority ordering and error isolation.

    Handlers are stored per event type and sorted by priority (lower = earlier).
    Each handler runs in a try/except to prevent one broken handler from
    blocking others.

    Supports both sync and async handler functions.
    """

    def __init__(self) -> None:
        # event_type → [(priority, handler)]
        self._handlers: dict[str, list[tuple[int, Callable]]] = defaultdict(list)

    def register(self, event_type: str, handler: Callable, priority: int = 0) -> None:
        """Register a handler for an event type.

        Args:
            event_type: Event name (e.g., "spec:created", "phase:started").
            handler: Callable that accepts a context dict. Can be sync or async.
            priority: Lower numbers execute first. Default 0.
        """
        handlers = self._handlers[event_type]
        handlers.append((priority, handler))
        handlers.sort(key=lambda x: x[0])

    def unregister(self, event_type: str, handler: Callable) -> None:
        """Remove a handler for an event type.

        Args:
            event_type: Event name.
            handler: The handler function to remove.
        """
        handlers = self._handlers.get(event_type, [])
        self._handlers[event_type] = [
            (p, h) for p, h in handlers if h is not handler
        ]

    async def emit(self, event_type: str, context: dict | None = None) -> list[Any]:
        """Emit an event, calling all registered handlers.

        Each handler receives a deep copy of the context dict to prevent
        mutation of the original data. Errors in individual handlers are
        caught and logged to stderr without blocking other handlers.

        Args:
            event_type: Event name to emit.
            context: Data dict passed to each handler.

        Returns:
            List of handler return values (None for failed handlers).
        """
        handlers = self._handlers.get(event_type, [])
        if not handlers:
            return []

        # Freeze context to prevent handler-to-handler mutation
        frozen_context = copy.deepcopy(context) if context else {}
        results: list[Any] = []

        for _priority, handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(frozen_context)
                else:
                    result = handler(frozen_context)
                results.append(result)
            except Exception as e:
                results.append(None)
                try:
                    handler_name = getattr(handler, "__name__", repr(handler))
                    sys.stderr.write(
                        f"[hooks] Handler {handler_name} for '{event_type}' "
                        f"raised {type(e).__name__}: {e}\n"
                    )
                    sys.stderr.flush()
                except (OSError, UnicodeEncodeError):
                    pass

        return results

    def emit_sync(self, event_type: str, context: dict | None = None) -> list[Any]:
        """Emit an event synchronously, calling only sync handlers.

        Designed for hot paths (e.g., tool call hooks) where asyncio
        overhead is unacceptable. Async handlers are skipped with a
        warning to stderr.

        Args:
            event_type: Event name to emit.
            context: Data dict passed to each handler.

        Returns:
            List of handler return values (None for skipped/failed handlers).
        """
        handlers = self._handlers.get(event_type, [])
        if not handlers:
            return []

        frozen_context = copy.deepcopy(context) if context else {}
        results: list[Any] = []

        for _priority, handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    # Skip async handlers in sync path
                    results.append(None)
                    try:
                        handler_name = getattr(handler, "__name__", repr(handler))
                        sys.stderr.write(
                            f"[hooks] Skipping async handler {handler_name} "
                            f"in sync emit for '{event_type}'\n"
                        )
                        sys.stderr.flush()
                    except (OSError, UnicodeEncodeError):
                        pass
                    continue

                result = handler(frozen_context)
                results.append(result)
            except Exception as e:
                results.append(None)
                try:
                    handler_name = getattr(handler, "__name__", repr(handler))
                    sys.stderr.write(
                        f"[hooks] Handler {handler_name} for '{event_type}' "
                        f"raised {type(e).__name__}: {e}\n"
                    )
                    sys.stderr.flush()
                except (OSError, UnicodeEncodeError):
                    pass

        return results

    def list_events(self) -> list[str]:
        """Return all event types that have registered handlers."""
        return sorted(self._handlers.keys())

    def handler_count(self, event_type: str) -> int:
        """Return the number of handlers for an event type."""
        return len(self._handlers.get(event_type, []))

    def clear(self, event_type: str | None = None) -> None:
        """Clear handlers. If event_type is None, clear all.

        Args:
            event_type: Specific event to clear, or None for all.
        """
        if event_type is None:
            self._handlers.clear()
        else:
            self._handlers.pop(event_type, None)


# ── Singleton + convenience functions ──────────────────────────────────────

hook_registry = HookRegistry()
register_hook = hook_registry.register
unregister_hook = hook_registry.unregister
emit_hook = hook_registry.emit
emit_hook_sync = hook_registry.emit_sync


# ── Event type constants ──────────────────────────────────────────────────

SPEC_CREATED = "spec:created"
SPEC_STARTED = "spec:started"
SPEC_COMPLETED = "spec:completed"
SPEC_FAILED = "spec:failed"

PHASE_STARTED = "phase:started"
PHASE_COMPLETED = "phase:completed"

AGENT_SESSION_START = "agent:session_start"
AGENT_SESSION_END = "agent:session_end"

DAEMON_TASK_PICKED = "daemon:task_picked"
DAEMON_TASK_DONE = "daemon:task_done"

TOOL_BEFORE_CALL = "tool:before_call"
TOOL_AFTER_CALL = "tool:after_call"
TOOL_BLOCKED = "tool:blocked"
