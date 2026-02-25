"""
Enhanced Retry with Jitter (OpenClaw P3.1)
==========================================

Intelligent retry logic with Retry-After header parsing, jitter for
thundering herd prevention, and provider-specific error detection.

Replaces/extends the basic retry_backoff() in spec/phases/models.py.

Usage:
    from core.retry import calculate_backoff, parse_retry_after, RetryConfig

    config = RetryConfig(max_retries=3, base_delay=2.0)
    for attempt in range(config.max_retries):
        try:
            result = await api_call()
            break
        except ApiError as e:
            delay = calculate_backoff(attempt, config, parse_retry_after(e.headers))
            await asyncio.sleep(delay)
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

# Reuse retryable detection from models.py (single source of truth)
from spec.phases.models import is_retryable_error  # noqa: F401 — re-export


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 2.0
    max_delay: float = 300.0
    jitter: bool = True
    retryable_statuses: set[int] = field(
        default_factory=lambda: {429, 500, 502, 503, 529}
    )


# Default singleton config
DEFAULT_CONFIG = RetryConfig()


def parse_retry_after(headers: dict | None) -> float | None:
    """Parse Retry-After header value (seconds or HTTP-date).

    Handles both formats per RFC 7231:
    - Seconds: "Retry-After: 120" → 120.0
    - HTTP-date: "Retry-After: Fri, 07 Feb 2026 12:00:00 GMT" → delta seconds

    Args:
        headers: Response headers dict (case-insensitive lookup)

    Returns:
        Delay in seconds, or None if no valid Retry-After found.
    """
    if not headers:
        return None

    # Case-insensitive header lookup
    value = None
    for k, v in headers.items():
        if k.lower() == "retry-after":
            value = str(v).strip()
            break

    if not value:
        return None

    # Try numeric seconds first
    if re.match(r"^\d+(\.\d+)?$", value):
        delay = float(value)
        return max(0.0, min(delay, 600.0))  # Cap at 10 minutes

    # Try HTTP-date format
    try:
        retry_dt = parsedate_to_datetime(value)
        now = datetime.now(timezone.utc)
        delta = (retry_dt - now).total_seconds()
        return max(0.0, min(delta, 600.0))
    except (ValueError, TypeError, OverflowError):
        return None


def calculate_backoff(
    attempt: int,
    config: RetryConfig | None = None,
    retry_after: float | None = None,
) -> float:
    """Calculate backoff delay for a retry attempt.

    Uses exponential backoff with optional jitter and Retry-After override.

    Args:
        attempt: Zero-based attempt index (0 = first retry)
        config: Retry configuration (uses DEFAULT_CONFIG if None)
        retry_after: Parsed Retry-After value (overrides calculated delay)

    Returns:
        Delay in seconds before next retry.
    """
    if config is None:
        config = DEFAULT_CONFIG

    if retry_after is not None and retry_after > 0:
        # Retry-After takes precedence, but add small jitter to avoid thundering herd
        delay = retry_after
        if config.jitter:
            delay *= 0.9 + random.random() * 0.2  # +/- 10%
        return min(delay, config.max_delay)

    # Exponential backoff: base_delay * 2^attempt
    delay = config.base_delay * (2**attempt)

    # Apply jitter: delay * [0.5, 1.0)
    if config.jitter:
        delay *= 0.5 + random.random() * 0.5

    return min(delay, config.max_delay)


def log_retry_event(
    spec_dir: Path,
    attempt: int,
    delay: float,
    error_text: str,
    provider: str = "",
) -> None:
    """Log a retry attempt to events.jsonl.

    Args:
        spec_dir: Spec directory for events.jsonl
        attempt: Current attempt number (0-based)
        delay: Backoff delay in seconds
        error_text: Error message that triggered the retry
        provider: Provider name (anthropic, openai, etc.)
    """
    try:
        from core.task_event import append_event_log

        append_event_log(spec_dir, {
            "type": "RETRY_ATTEMPT",
            "attempt": attempt,
            "delaySeconds": round(delay, 2),
            "errorPreview": error_text[:200] if error_text else "",
            "provider": provider,
        })
    except Exception:
        pass  # Non-critical: best-effort logging
