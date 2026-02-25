"""
Daemon Status Bridge
====================

Writes daemon_status.json in the same format as TaskDaemon.get_status()
so that the Electron UI's DaemonStatusWatcher can track builds started
from the CLI (run.py) without the full TaskDaemon running.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class DaemonStatusBridge:
    """CLI standalone bridge that writes daemon_status.json for UI real-time sync."""

    def __init__(self, project_dir: Path, spec_id: str, spec_dir: Path, *, task_type: str = "impl"):
        self.status_path = project_dir / "daemon_status.json"
        self.project_dir = project_dir
        self.spec_id = spec_id
        self.spec_dir = spec_dir
        self.task_type = task_type
        self._active = False
        self._started_at: str | None = None
        self._completed = 0

    def start(self) -> None:
        """Create or merge into daemon_status.json at build start."""
        existing = self._read_existing()

        # If daemon is already managing this task, skip to avoid double writes
        if existing and self.spec_id in existing.get("running_tasks", {}):
            task_info = existing["running_tasks"][self.spec_id]
            # Check if it looks like a real daemon (has pid field with a live process)
            pid = task_info.get("pid")
            if pid and self._is_pid_alive(pid):
                logger.info(
                    "daemon_status_bridge: task %s already managed by daemon (pid %s), skipping",
                    self.spec_id,
                    pid,
                )
                return

        self._active = True
        self._started_at = datetime.now(timezone.utc).isoformat()

        # Merge into existing status or create new
        if existing and existing.get("running"):
            # Daemon is running — merge our task into its running_tasks
            existing["running_tasks"][self.spec_id] = self._make_task_info()
            existing["stats"]["running"] = len(existing["running_tasks"])
            existing["timestamp"] = datetime.now(timezone.utc).isoformat()
            self._write(existing)
        else:
            # No daemon — create fresh status
            self._write(self._make_status())

        logger.info("daemon_status_bridge: started for %s", self.spec_id)

    def update(
        self,
        *,
        subtask_id: str | None = None,
        phase: str | None = None,
        session: int | None = None,
    ) -> None:
        """Update running_tasks entry on subtask progress."""
        if not self._active:
            return

        status = self._read_existing()
        if not status:
            status = self._make_status()

        task_info = status.get("running_tasks", {}).get(
            self.spec_id, self._make_task_info()
        )
        task_info["last_update"] = datetime.now(timezone.utc).isoformat()
        if subtask_id:
            task_info["current_subtask"] = subtask_id
        if phase:
            task_info["phase"] = phase
        if session is not None:
            task_info["session"] = session

        status.setdefault("running_tasks", {})[self.spec_id] = task_info
        status["stats"]["running"] = len(
            [t for t in status["running_tasks"].values() if t.get("is_running")]
        )
        status["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._write(status)

    def complete(self) -> None:
        """Remove task from running_tasks and increment completed count."""
        if not self._active:
            return

        status = self._read_existing()
        if not status:
            return

        status.get("running_tasks", {}).pop(self.spec_id, None)
        self._completed += 1
        status["stats"]["completed"] = status["stats"].get("completed", 0) + 1
        status["stats"]["running"] = len(
            [
                t
                for t in status.get("running_tasks", {}).values()
                if t.get("is_running")
            ]
        )
        status["timestamp"] = datetime.now(timezone.utc).isoformat()

        # If no more running tasks, mark daemon as stopped
        if not status.get("running_tasks"):
            status["running"] = False

        self._write(status)
        self._active = False
        logger.info("daemon_status_bridge: completed %s", self.spec_id)

    def close(self) -> None:
        """Clean up on build exit/failure."""
        if not self._active:
            return

        status = self._read_existing()
        if not status:
            self._active = False
            return

        status.get("running_tasks", {}).pop(self.spec_id, None)
        status["stats"]["running"] = len(
            [
                t
                for t in status.get("running_tasks", {}).values()
                if t.get("is_running")
            ]
        )
        status["timestamp"] = datetime.now(timezone.utc).isoformat()

        if not status.get("running_tasks"):
            status["running"] = False

        self._write(status)
        self._active = False
        logger.info("daemon_status_bridge: closed %s", self.spec_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_task_info(self) -> dict:
        """Create a running_tasks entry matching TaskState.to_dict() format."""
        now = datetime.now(timezone.utc).isoformat()
        return {
            "spec_id": self.spec_id,
            "spec_dir": str(self.spec_dir),
            "status": "in_progress",
            "is_running": True,
            "started_at": self._started_at or now,
            "last_update": now,
            "task_type": self.task_type,
        }

    def _make_status(self) -> dict:
        """Create full daemon_status.json matching TaskDaemon.get_status() format."""
        return {
            "project_dir": str(self.project_dir),
            "running": True,
            "started_at": self._started_at or datetime.now(timezone.utc).isoformat(),
            "config": {
                "max_concurrent_tasks": 1,
                "headless_mode": True,
            },
            "running_tasks": {
                self.spec_id: self._make_task_info(),
            },
            "queued_tasks": [],
            "stats": {
                "running": 1,
                "queued": 0,
                "completed": self._completed,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _read_existing(self) -> dict | None:
        """Read existing daemon_status.json, return None if missing/invalid."""
        if not self.status_path.exists():
            return None
        try:
            content = self.status_path.read_text(encoding="utf-8-sig")
            return json.loads(content)
        except (OSError, json.JSONDecodeError):
            return None

    def _write(self, data: dict) -> None:
        """Atomic write: PID-unique .tmp → rename.

        Uses PID in temp filename to prevent collisions when multiple
        run.py processes write concurrently.
        """
        tmp_path = self.status_path.with_suffix(f".{os.getpid()}.tmp")
        try:
            tmp_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_path.replace(self.status_path)
        except OSError as e:
            logger.warning("daemon_status_bridge: write failed: %s", e)
            # Clean up temp file on failure
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        """Check if a process with given PID is alive.

        On Windows, os.kill(pid, 0) calls TerminateProcess which would KILL
        the target process.  Use ctypes OpenProcess + CloseHandle instead.
        """
        import sys

        if sys.platform == "win32":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False

        # Unix: signal 0 just checks existence, does not kill
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False
