"""
Structured Logger with Redaction (OpenClaw P3.2)
=================================================

JSON-structured logging with automatic API key masking.
Outputs to spec_dir/logs/{component}.jsonl with rolling file support.

Usage:
    from core.logging_ext import get_logger

    log = get_logger(spec_dir, "orchestrator")
    log.info("Phase started", phase="planning", model="sonnet")
    log.error("Agent failed", error=exc, attempt=2)
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Maximum log file size before rotation (bytes)
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Redaction patterns: (regex, replacement)
_REDACT_PATTERNS = [
    (re.compile(r"sk-ant-[a-zA-Z0-9\-]{20,}"), "sk-ant-****"),
    (re.compile(r"sk-proj-[a-zA-Z0-9]{20,}"), "sk-proj-****"),
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "sk-****"),
    (re.compile(r"ghp_[a-zA-Z0-9]{36}"), "ghp_****"),
    (re.compile(r"gho_[a-zA-Z0-9]{36}"), "gho_****"),
    (re.compile(r"Bearer [a-zA-Z0-9\-._~+/]+=*"), "Bearer ****"),
    (
        re.compile(r"-----BEGIN [A-Z ]+ KEY-----[\s\S]*?-----END [A-Z ]+ KEY-----"),
        "[REDACTED_PEM]",
    ),
    (re.compile(r"api[_-]?key[\"']?\s*[:=]\s*[\"'][^\"']{8,}[\"']"), "api_key=****"),
]


def redact(text: str) -> str:
    """Mask sensitive information in text.

    Handles API keys (Anthropic, OpenAI, GitHub), Bearer tokens,
    and PEM private keys.

    Args:
        text: Input text potentially containing secrets.

    Returns:
        Text with secrets replaced by masked placeholders.
    """
    if not text or not isinstance(text, str):
        return text
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _redact_value(value: Any) -> Any:
    """Recursively redact sensitive values in nested structures."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_redact_value(v) for v in value)
    return value


class StructuredLogger:
    """JSON-structured logger writing to rolling JSONL files.

    Each log entry is a single JSON line with timestamp, level, component,
    message, and arbitrary context fields. All string values are automatically
    redacted for sensitive information.

    Attributes:
        spec_dir: Spec directory root.
        component: Logger name (e.g., "orchestrator", "agent_runner").
        max_size: Max file size in bytes before rotation.
    """

    def __init__(self, spec_dir: Path, component: str, max_size_mb: int = 10):
        self.spec_dir = spec_dir
        self.component = component
        self.max_size = max_size_mb * 1024 * 1024
        self._log_dir = spec_dir / "logs"

    def _get_log_path(self) -> Path:
        return self._log_dir / f"{self.component}.jsonl"

    def _rotate_if_needed(self, log_path: Path) -> None:
        """Rotate log file if it exceeds max size."""
        try:
            if log_path.exists() and log_path.stat().st_size > self.max_size:
                rotated = log_path.with_suffix(".jsonl.1")
                # Keep only one rotated file
                if rotated.exists():
                    rotated.unlink()
                log_path.rename(rotated)
        except OSError:
            pass  # Non-critical: continue logging to same file

    def _write(self, level: str, message: str, **context: Any) -> None:
        """Write a single log entry."""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self._get_log_path()
            self._rotate_if_needed(log_path)

            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": level,
                "component": self.component,
                "msg": redact(message),
            }

            # Add redacted context
            for k, v in context.items():
                if k == "error" and isinstance(v, Exception):
                    entry[k] = redact(str(v))
                    entry["error_type"] = type(v).__name__
                else:
                    entry[k] = _redact_value(v)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except (OSError, UnicodeEncodeError):
            pass  # Non-critical: best-effort logging

    def info(self, message: str, **context: Any) -> None:
        """Log at INFO level."""
        self._write("info", message, **context)

    def warn(self, message: str, **context: Any) -> None:
        """Log at WARN level."""
        self._write("warn", message, **context)

    def error(self, message: str, error: Exception | None = None, **context: Any) -> None:
        """Log at ERROR level with optional exception."""
        if error is not None:
            context["error"] = error
        self._write("error", message, **context)

    def debug(self, message: str, **context: Any) -> None:
        """Log at DEBUG level (only if DEBUG env var is set)."""
        if os.environ.get("DEBUG", "").lower() in ("1", "true", "yes"):
            self._write("debug", message, **context)


# Logger cache to avoid creating duplicates
_loggers: dict[str, StructuredLogger] = {}


def get_logger(spec_dir: Path, component: str) -> StructuredLogger:
    """Get or create a StructuredLogger for the given component.

    Args:
        spec_dir: Spec directory for log file placement.
        component: Logger name (e.g., "orchestrator", "agent_runner").

    Returns:
        A StructuredLogger instance.
    """
    key = f"{spec_dir}:{component}"
    if key not in _loggers:
        _loggers[key] = StructuredLogger(spec_dir, component)
    return _loggers[key]
