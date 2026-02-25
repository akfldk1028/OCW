"""
Task event protocol for frontend XState synchronization.

Protocol: __TASK_EVENT__:{...}

This module handles:
1. Event emission via stdout for frontend XState machine
2. Status persistence to implementation_plan.json for CLI-only execution
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

TASK_EVENT_PREFIX = "__TASK_EVENT__:"
_DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")


def _fire_hook(event_type: str, context: dict) -> None:
    """Fire registered hooks for a task event (best-effort, non-blocking).

    Uses asyncio if a loop is running, otherwise fires synchronously.
    Errors are silently ignored to never block the pipeline.
    """
    try:
        from core.hooks import hook_registry

        if not hook_registry.handler_count(event_type):
            return

        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(hook_registry.emit(event_type, context))
        except RuntimeError:
            # No running loop — skip async hooks
            pass
    except Exception:
        pass  # Non-critical: hooks must never break the pipeline


def append_event_log(spec_dir: Path, event: dict) -> None:
    """Append event to immutable JSONL log file (OpenClaw pattern).

    This provides an append-only event history that:
    - Avoids file lock issues (append is atomic on most OS)
    - Preserves complete event history for debugging/replay
    - Coexists with implementation_plan.json (which stores final snapshot)

    Each line is one JSON object with timestamp.
    """
    log_file = spec_dir / "events.jsonl"
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
    except (OSError, UnicodeEncodeError):
        pass  # Non-critical: best-effort logging

# Status mapping for XState state to TaskStatus
XSTATE_TO_TASK_STATUS = {
    "backlog": "backlog",
    "planning": "in_progress",
    "plan_review": "human_review",
    "coding": "in_progress",
    "qa_review": "ai_review",
    "qa_fixing": "ai_review",
    "human_review": "human_review",
    "pr_created": "pr_created",
    "done": "done",
    "error": "error",
}


def _persist_status_to_plan(
    spec_dir: Path,
    status: str,
    xstate_state: str | None = None,
    review_reason: str | None = None,
    execution_phase: str | None = None,
) -> bool:
    """Persist status fields to implementation_plan.json.

    This ensures CLI-only execution (without UI) still saves status for
    later UI sync and Kanban card positioning.

    Args:
        spec_dir: Path to the spec directory
        status: TaskStatus value (e.g., 'in_progress', 'human_review')
        xstate_state: XState machine state (e.g., 'planning', 'coding')
        review_reason: Review reason (e.g., 'completed', 'plan_review')
        execution_phase: Execution phase (e.g., 'planning', 'coding', 'qa_review')

    Returns:
        True if persisted successfully, False otherwise
    """
    plan_path = spec_dir / "implementation_plan.json"

    try:
        # Read existing plan or create minimal one
        if plan_path.exists():
            with open(plan_path, encoding="utf-8") as f:
                plan = json.load(f)
        else:
            # Create minimal plan if it doesn't exist yet
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan = {"phases": [], "created_at": datetime.now(timezone.utc).isoformat()}

        # Update status fields
        plan["status"] = status
        plan["planStatus"] = _map_status_to_plan_status(status)

        if xstate_state:
            plan["xstateState"] = xstate_state
        if review_reason:
            plan["reviewReason"] = review_reason
        if execution_phase:
            plan["executionPhase"] = execution_phase

        plan["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Schema validation before write (OpenClaw pattern: config-as-code with validation)
        from core.schema import validate_plan

        is_valid, schema_errors = validate_plan(plan)
        if not is_valid and _DEBUG:
            try:
                sys.stderr.write(
                    f"[task_event] Schema warnings before persist: {schema_errors}\n"
                )
                sys.stderr.flush()
            except (OSError, UnicodeEncodeError):
                pass
        # Write even if schema warns — don't block status updates, just log

        # Write atomically using temp file pattern
        from core.file_utils import write_json_atomic

        write_json_atomic(plan_path, plan, indent=2)

        if _DEBUG:
            sys.stderr.write(
                f"[task_event] Persisted status={status} xstate={xstate_state} to {plan_path}\n"
            )
            sys.stderr.flush()

        return True

    except Exception as e:
        if _DEBUG:
            try:
                sys.stderr.write(f"[task_event] persist_status failed: {e}\n")
                sys.stderr.flush()
            except (OSError, UnicodeEncodeError):
                pass
        return False


def _map_status_to_plan_status(status: str) -> str:
    """Map UI TaskStatus to Python-compatible planStatus."""
    mapping = {
        "queue": "queued",
        "in_progress": "in_progress",
        "ai_review": "review",
        "human_review": "review",
        "done": "completed",
        "error": "failed",
    }
    return mapping.get(status, "pending")


@dataclass
class TaskEventContext:
    task_id: str
    spec_id: str
    project_id: str
    sequence_start: int = 0


def _load_task_metadata(spec_dir: Path) -> dict:
    metadata_path = spec_dir / "task_metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with open(metadata_path, encoding="utf-8") as f:
            data = json.load(f)
        # Schema validation (OpenClaw pattern: catch dependency field conflicts early)
        if _DEBUG:
            from core.schema import validate_task_metadata

            is_valid, schema_errors = validate_task_metadata(data)
            if not is_valid:
                try:
                    sys.stderr.write(
                        f"[task_event] task_metadata schema issues: {schema_errors}\n"
                    )
                    sys.stderr.flush()
                except (OSError, UnicodeEncodeError):
                    pass
        return data
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _load_last_sequence(spec_dir: Path) -> int:
    plan_path = spec_dir / "implementation_plan.json"
    if not plan_path.exists():
        return 0
    try:
        with open(plan_path, encoding="utf-8") as f:
            plan = json.load(f)
        last_event = plan.get("lastEvent") or {}
        seq = last_event.get("sequence")
        if isinstance(seq, int) and seq >= 0:
            return seq + 1
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return 0
    return 0


def load_task_event_context(spec_dir: Path) -> TaskEventContext:
    metadata = _load_task_metadata(spec_dir)
    task_id = metadata.get("taskId") or metadata.get("task_id") or spec_dir.name
    spec_id = metadata.get("specId") or metadata.get("spec_id") or spec_dir.name
    project_id = metadata.get("projectId") or metadata.get("project_id") or ""
    sequence_start = _load_last_sequence(spec_dir)
    return TaskEventContext(
        task_id=str(task_id),
        spec_id=str(spec_id),
        project_id=str(project_id),
        sequence_start=sequence_start,
    )


class TaskEventEmitter:
    def __init__(self, context: TaskEventContext, spec_dir: Path | None = None) -> None:
        self._context = context
        self._sequence = context.sequence_start
        self._spec_dir = spec_dir

    @classmethod
    def from_spec_dir(cls, spec_dir: Path) -> TaskEventEmitter:
        return cls(load_task_event_context(spec_dir), spec_dir)

    def emit(self, event_type: str, payload: dict | None = None) -> None:
        event = {
            "type": event_type,
            "taskId": self._context.task_id,
            "specId": self._context.spec_id,
            "projectId": self._context.project_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "eventId": str(uuid4()),
            "sequence": self._sequence,
        }
        if payload:
            event.update(payload)

        try:
            print(f"{TASK_EVENT_PREFIX}{json.dumps(event, default=str)}", flush=True)
            self._sequence += 1
        except (OSError, UnicodeEncodeError) as e:
            if _DEBUG:
                try:
                    sys.stderr.write(f"[task_event] emit failed: {e}\n")
                    sys.stderr.flush()
                except (OSError, UnicodeEncodeError):
                    pass  # Silent on complete I/O failure

        # Append to immutable JSONL event log (OpenClaw pattern)
        if self._spec_dir:
            append_event_log(self._spec_dir, event)

        # Fire registered hooks (OpenClaw P3: event-based extension)
        _fire_hook(event_type, event)

        # Auto-persist status for key events (CLI-only execution support)
        if self._spec_dir:
            self._auto_persist_status(event_type, payload)

    def _auto_persist_status(self, event_type: str, payload: dict | None) -> None:
        """Auto-persist status to implementation_plan.json for key events.

        This ensures CLI-only execution (without UI) still saves status for
        later UI sync and Kanban card positioning.
        """
        # Map event types to status fields
        event_status_map = {
            "PLANNING_STARTED": {
                "status": "in_progress",
                "xstate_state": "planning",
                "execution_phase": "planning",
            },
            "PLANNING_COMPLETE": {
                "status": "human_review",
                "xstate_state": "plan_review",
                "review_reason": "plan_review",
                "execution_phase": "planning",
            },
            "PLAN_APPROVED": {
                "status": "in_progress",
                "xstate_state": "coding",
                "execution_phase": "coding",
            },
            "ALL_SUBTASKS_DONE": {
                "status": "ai_review",
                "xstate_state": "qa_review",
                "execution_phase": "qa_review",
            },
            "QA_STARTED": {
                "status": "ai_review",
                "xstate_state": "qa_review",
                "execution_phase": "qa_review",
            },
            "QA_PASSED": {
                "status": "human_review",
                "xstate_state": "human_review",
                "review_reason": "completed",
                "execution_phase": "complete",
            },
            "QA_FAILED": {
                "status": "in_progress",
                "xstate_state": "qa_fixing",
                "execution_phase": "qa_fixing",
            },
            "QA_FIXING_COMPLETE": {
                "status": "ai_review",
                "xstate_state": "qa_review",
                "execution_phase": "qa_review",
            },
            "QA_MAX_ITERATIONS": {
                "status": "human_review",
                "xstate_state": "human_review",
                "review_reason": "max_iterations",
                "execution_phase": "complete",
            },
            "CODING_FAILED": {
                "status": "error",
                "xstate_state": "error",
                "review_reason": "errors",
                "execution_phase": "failed",
            },
            "PLANNING_FAILED": {
                "status": "error",
                "xstate_state": "error",
                "review_reason": "errors",
                "execution_phase": "failed",
            },
            "MARK_DONE": {
                "status": "done",
                "xstate_state": "done",
                "execution_phase": "complete",
            },
        }

        if event_type not in event_status_map:
            return

        status_info = event_status_map[event_type]
        _persist_status_to_plan(
            self._spec_dir,
            status=status_info["status"],
            xstate_state=status_info.get("xstate_state"),
            review_reason=status_info.get("review_reason"),
            execution_phase=status_info.get("execution_phase"),
        )

    def emit_status_change(
        self,
        status: str,
        review_reason: str | None = None,
        execution_phase: str | None = None,
    ) -> None:
        """Emit a TASK_STATUS_CHANGE event for XState machine transition.

        This is a convenience method that emits a status change event with
        the correct payload structure for the frontend to sync Kanban cards.

        Args:
            status: The new status (e.g., 'in_progress', 'human_review', 'done')
            review_reason: Optional reason for review state (e.g., 'completed', 'plan_review')
            execution_phase: Optional execution phase (e.g., 'planning', 'coding', 'qa_review')
        """
        payload: dict = {"status": status}
        if review_reason:
            payload["reviewReason"] = review_reason
        if execution_phase:
            payload["executionPhase"] = execution_phase
        self.emit("TASK_STATUS_CHANGE", payload)


def emit_status_change(
    emitter: TaskEventEmitter,
    status: str,
    review_reason: str | None = None,
    execution_phase: str | None = None,
) -> None:
    """Convenience function to emit a status change event.

    This is a standalone function wrapper around TaskEventEmitter.emit_status_change()
    for cases where you have an emitter instance but prefer functional style.

    Args:
        emitter: The TaskEventEmitter instance
        status: The new status (e.g., 'in_progress', 'human_review', 'done')
        review_reason: Optional reason for review state
        execution_phase: Optional execution phase
    """
    emitter.emit_status_change(status, review_reason, execution_phase)
