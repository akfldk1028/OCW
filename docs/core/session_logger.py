"""
Per-Agent Session Logger (OpenClaw P2.3)
========================================

Tracks agent session lifecycle with start/end events in events.jsonl.
Provides a context manager that captures session metadata (agent type,
model, duration, status, tool/message counts) without modifying the
streaming loop.

Usage:
    async with agent_session_logger(spec_dir, agent_type="coder", phase="coding") as session:
        # ... run agent session ...
        session.record_result(status="complete", tool_count=5, message_count=12)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.task_event import append_event_log


@dataclass
class SessionRecord:
    """Mutable record for collecting session metrics during execution."""

    session_id: str = field(default_factory=lambda: str(uuid4()))
    status: str = "running"
    tool_count: int = 0
    message_count: int = 0
    response_length: int = 0
    error_type: str | None = None
    error_message: str | None = None

    def record_result(
        self,
        status: str,
        tool_count: int = 0,
        message_count: int = 0,
        response_length: int = 0,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update session record with results."""
        self.status = status
        self.tool_count = tool_count
        self.message_count = message_count
        self.response_length = response_length
        self.error_type = error_type
        self.error_message = error_message


@asynccontextmanager
async def agent_session_logger(
    spec_dir: Path,
    agent_type: str,
    phase: str = "",
    subtask_id: str = "",
    session_num: int = 1,
    model: str = "",
):
    """
    Context manager that emits AGENT_SESSION_START/END to events.jsonl.

    Yields a SessionRecord that the caller populates with results.
    On exit, emits the END event with duration and metrics.

    Args:
        spec_dir: Spec directory for events.jsonl
        agent_type: Agent type (planner, coder, qa_reviewer, qa_fixer)
        phase: Execution phase (planning, coding, qa_review, qa_fixing)
        subtask_id: Current subtask ID (if applicable)
        session_num: Session/iteration number
        model: Model name used for this session
    """
    record = SessionRecord()
    start_time = time.monotonic()

    # Emit session start
    start_event: dict[str, Any] = {
        "type": "AGENT_SESSION_START",
        "sessionId": record.session_id,
        "agentType": agent_type,
        "phase": phase,
        "sessionNum": session_num,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if subtask_id:
        start_event["subtaskId"] = subtask_id
    if model:
        start_event["model"] = model

    append_event_log(spec_dir, start_event)

    try:
        yield record
    finally:
        duration = time.monotonic() - start_time

        # Emit session end
        end_event: dict[str, Any] = {
            "type": "AGENT_SESSION_END",
            "sessionId": record.session_id,
            "agentType": agent_type,
            "phase": phase,
            "sessionNum": session_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "durationSeconds": round(duration, 2),
            "status": record.status,
            "toolCount": record.tool_count,
            "messageCount": record.message_count,
            "responseLength": record.response_length,
        }
        if subtask_id:
            end_event["subtaskId"] = subtask_id
        if model:
            end_event["model"] = model
        if record.error_type:
            end_event["errorType"] = record.error_type
        if record.error_message:
            end_event["errorMessage"] = record.error_message[:500]

        append_event_log(spec_dir, end_event)
