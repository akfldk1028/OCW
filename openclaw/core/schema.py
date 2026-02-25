"""
Plan schema validation (OpenClaw pattern: config-as-code with validation).

Validates implementation_plan.json structure to catch malformed data early
instead of failing silently at runtime.
"""

from __future__ import annotations

from typing import Any

VALID_STATUSES = {"queue", "in_progress", "ai_review", "human_review", "done", "error", "complete"}
VALID_XSTATE_STATES = {
    "backlog", "planning", "plan_review", "coding",
    "qa_review", "qa_fixing", "human_review", "pr_created", "done", "error",
}
VALID_EXECUTION_PHASES = {
    "backlog", "planning", "coding", "qa_review", "qa_fixing", "complete", "failed",
}
VALID_TASK_TYPES = {"default", "design", "architecture", "research", "mcts"}


def validate_plan(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate implementation_plan.json structure.

    Returns:
        (is_valid, list_of_errors)
    """
    errors: list[str] = []

    # Required fields
    if "phases" not in data and "subtasks" not in data:
        errors.append("Missing 'phases' or 'subtasks' field")

    # Status validation (if present)
    status = data.get("status")
    if status and status not in VALID_STATUSES:
        errors.append(f"Invalid status '{status}', expected one of: {VALID_STATUSES}")

    xstate = data.get("xstateState")
    if xstate and xstate not in VALID_XSTATE_STATES:
        errors.append(f"Invalid xstateState '{xstate}', expected one of: {VALID_XSTATE_STATES}")

    exec_phase = data.get("executionPhase")
    if exec_phase and exec_phase not in VALID_EXECUTION_PHASES:
        errors.append(f"Invalid executionPhase '{exec_phase}', expected one of: {VALID_EXECUTION_PHASES}")

    task_type = data.get("taskType")
    if task_type and task_type not in VALID_TASK_TYPES:
        errors.append(f"Invalid taskType '{task_type}', expected one of: {VALID_TASK_TYPES}")

    # Phase structure validation
    phases = data.get("phases", [])
    if not isinstance(phases, list):
        errors.append(f"'phases' must be a list, got {type(phases).__name__}")
    else:
        for i, phase in enumerate(phases):
            if not isinstance(phase, dict):
                errors.append(f"Phase {i} must be a dict, got {type(phase).__name__}")
                continue
            if "id" not in phase and "name" not in phase:
                errors.append(f"Phase {i} missing 'id' or 'name'")

    # Subtask status validation
    for phase in phases if isinstance(phases, list) else []:
        for subtask in phase.get("subtasks", []):
            if isinstance(subtask, dict):
                st_status = subtask.get("status")
                if st_status and st_status not in {"pending", "in_progress", "completed", "failed", "skipped"}:
                    errors.append(f"Subtask '{subtask.get('id', '?')}' has invalid status: '{st_status}'")

    return len(errors) == 0, errors


def validate_requirements(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate requirements.json structure."""
    errors: list[str] = []

    if "task_description" not in data and "description" not in data:
        errors.append("Missing 'task_description' or 'description'")

    return len(errors) == 0, errors


def validate_task_metadata(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate task_metadata.json structure."""
    errors: list[str] = []

    # Check dependency field naming consistency
    dep_fields = [k for k in data if k.lower().replace("_", "") in ("dependson", "dependencies")]
    if len(dep_fields) > 1:
        errors.append(f"Multiple dependency fields found: {dep_fields}. Use 'dependsOn' only.")

    return len(errors) == 0, errors
