"""
Session Diagnostics + Cache Trace (OpenClaw P3.5)
==================================================

Message fingerprinting for tracking prompt cache efficiency.
Outputs to spec_dir/logs/cache_trace.jsonl for cost optimization insights.

Activation: Set AUTO_CLAUDE_CACHE_TRACE=1 environment variable.

Usage:
    from core.cache_trace import CacheTrace

    trace = CacheTrace(spec_dir, session_id="abc123")
    trace.trace_message("user", content, cached=False)
    trace.trace_message("assistant", response, cached=True)
    summary = trace.summary()
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def is_cache_trace_enabled() -> bool:
    """Check if cache tracing is enabled via environment variable."""
    return os.environ.get("AUTO_CLAUDE_CACHE_TRACE", "").lower() in ("1", "true", "yes")


def fingerprint(content: str) -> str:
    """Generate a SHA-256 fingerprint of message content (first 12 chars).

    Privacy-safe: only stores the hash prefix, not the content itself.

    Args:
        content: Message content to fingerprint.

    Returns:
        12-character hex hash prefix.
    """
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:12]


@dataclass
class CacheTrace:
    """Tracks message fingerprints and cache hits for a session.

    Records each message's role, content fingerprint, cache status,
    and content length to spec_dir/logs/cache_trace.jsonl.

    Attributes:
        spec_dir: Spec directory for log output.
        session_id: Unique session identifier.
        enabled: Whether tracing is active (checks env var by default).
    """

    spec_dir: Path
    session_id: str
    enabled: bool = field(default_factory=is_cache_trace_enabled)
    _total: int = field(default=0, init=False, repr=False)
    _cached: int = field(default=0, init=False, repr=False)
    _total_tokens_saved: int = field(default=0, init=False, repr=False)

    def trace_message(
        self,
        role: str,
        content: str,
        cached: bool = False,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> str:
        """Record a message trace entry.

        Args:
            role: Message role ("user", "assistant", "system").
            content: Message content for fingerprinting.
            cached: Whether this message hit the prompt cache.
            cache_creation_tokens: Tokens used for cache creation.
            cache_read_tokens: Tokens read from cache (savings).

        Returns:
            The content fingerprint (12-char hex).
        """
        if not self.enabled:
            return ""

        fp = fingerprint(content)
        self._total += 1
        if cached:
            self._cached += 1
        self._total_tokens_saved += cache_read_tokens

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "sessionId": self.session_id,
            "role": role,
            "fingerprint": fp,
            "cached": cached,
            "contentLength": len(content),
        }
        if cache_creation_tokens:
            entry["cacheCreationTokens"] = cache_creation_tokens
        if cache_read_tokens:
            entry["cacheReadTokens"] = cache_read_tokens

        self._write(entry)
        return fp

    def trace_tool_use(
        self,
        tool_name: str,
        input_hash: str,
        cached: bool = False,
    ) -> None:
        """Record a tool use trace entry.

        Args:
            tool_name: Name of the tool invoked.
            input_hash: Fingerprint of the tool input.
            cached: Whether the tool result was cached.
        """
        if not self.enabled:
            return

        self._total += 1
        if cached:
            self._cached += 1

        self._write({
            "ts": datetime.now(timezone.utc).isoformat(),
            "sessionId": self.session_id,
            "type": "tool_use",
            "toolName": tool_name,
            "inputHash": input_hash,
            "cached": cached,
        })

    def summary(self) -> dict:
        """Generate a cache efficiency summary.

        Returns:
            Dict with total_messages, cached_count, cache_ratio,
            and estimated_tokens_saved.
        """
        ratio = self._cached / self._total if self._total > 0 else 0.0
        return {
            "total_messages": self._total,
            "cached_count": self._cached,
            "cache_ratio": round(ratio, 3),
            "estimated_tokens_saved": self._total_tokens_saved,
        }

    def emit_summary_event(self) -> None:
        """Emit cache trace summary to events.jsonl."""
        if not self.enabled or self._total == 0:
            return
        try:
            from core.task_event import append_event_log

            event = {
                "type": "CACHE_TRACE_SUMMARY",
                "sessionId": self.session_id,
                **self.summary(),
            }
            append_event_log(self.spec_dir, event)
        except Exception:
            pass  # Non-critical

    def _write(self, entry: dict) -> None:
        """Append a trace entry to cache_trace.jsonl."""
        try:
            log_dir = self.spec_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "cache_trace.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except (OSError, UnicodeEncodeError):
            pass  # Non-critical
