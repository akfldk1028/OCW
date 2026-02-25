"""
Per-Agent Exec Policy (OpenClaw P0 Absorption)
================================================

Agent-aware bash command evaluation. Each agent type gets a SecurityLevel
that controls what commands it can execute, layered ON TOP of the existing
security/ module (zero modifications to security/*).

Node pattern:
    Input:  command, agent_type, project_dir, spec_dir
    Output: ExecDecision(allowed, reason, segments, policy_source)

Usage:
    from core.exec_policy import evaluate_exec_policy, SecurityLevel

    decision = evaluate_exec_policy(command, "spec_gatherer", project_dir, spec_dir)
    if not decision.allowed:
        # block with decision.reason
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SecurityLevel(Enum):
    """Exec permission level for an agent type.

    DENY:      All bash execution blocked.
    READONLY:  Only safe read-only binaries allowed (cat, ls, grep, etc.).
    ALLOWLIST: Defer to existing SecurityProfile allowlist (current behavior).
    FULL:      Defer to existing SecurityProfile allowlist (current behavior).
                Semantically identical to ALLOWLIST but signals "power agent".
    """

    DENY = "deny"
    READONLY = "readonly"
    ALLOWLIST = "allowlist"
    FULL = "full"


# ── Safe read-only binaries (always allowed for READONLY+ agents) ────────────

SAFE_BINS: frozenset[str] = frozenset({
    # File reading
    "cat", "head", "tail", "less", "more", "file", "stat",
    # Directory listing
    "ls", "dir", "tree", "pwd", "realpath", "basename", "dirname",
    # Search / filter
    "grep", "rg", "awk", "sed", "find", "which", "whereis", "locate",
    "wc", "sort", "uniq", "diff", "comm", "cut", "tr", "tee",
    # JSON / text
    "jq", "yq", "column", "fmt", "fold", "expand", "unexpand",
    # System info (read-only)
    "echo", "printf", "date", "uname", "whoami", "hostname", "env", "printenv",
    "id", "df", "du", "free", "uptime", "nproc", "arch",
    # VCS (read-only — destructive git ops blocked by security/git_validators.py)
    "git",
    # NOTE: python/node/npm/npx are intentionally EXCLUDED.
    # extract_commands() only sees the binary name, so allowing "npm" here
    # would let READONLY agents run "npm install" or "npx malicious-pkg".
    # Use extra_allow in AgentExecPolicy for specific agents that need them.
})


@dataclass
class CommandSegment:
    """Result of evaluating a single command within a compound statement."""

    raw: str
    command: str
    allowed: bool
    reason: str
    is_safe_bin: bool = False


@dataclass
class ExecDecision:
    """Final decision for a bash command execution request.

    Attributes:
        allowed: Whether the command should proceed.
        reason: Human-readable explanation.
        agent_type: Agent that requested execution.
        segments: Per-command breakdown (for compound commands).
        policy_source: Which policy layer made the decision.
    """

    allowed: bool
    reason: str
    agent_type: str
    segments: list[CommandSegment] = field(default_factory=list)
    policy_source: str = "agent_policy"


@dataclass
class AgentExecPolicy:
    """Per-agent execution policy configuration.

    Attributes:
        security_level: Base security level for the agent.
        extra_allow: Additional commands to allow beyond the level default.
        extra_deny: Commands to deny even if the level would allow them.
    """

    security_level: SecurityLevel
    extra_allow: frozenset[str] = field(default_factory=frozenset)
    extra_deny: frozenset[str] = field(default_factory=frozenset)


# ── Agent → Policy mapping (backed by AgentRegistry) ──────────────────────


def _build_exec_policies() -> dict[str, AgentExecPolicy]:
    """Build AGENT_EXEC_POLICIES from the unified AgentRegistry.

    Reads security_level, extra_allow, extra_deny from each AgentDefinition
    and creates the corresponding AgentExecPolicy objects.

    Returns a dict identical in shape to the old hardcoded AGENT_EXEC_POLICIES.
    """
    from core.agent_registry import AgentRegistry

    registry = AgentRegistry.instance()
    policies: dict[str, AgentExecPolicy] = {}

    for agent_id, defn in registry.all_agents().items():
        try:
            level = SecurityLevel(defn.security_level)
        except ValueError:
            level = SecurityLevel.ALLOWLIST

        policies[agent_id] = AgentExecPolicy(
            security_level=level,
            extra_allow=frozenset(defn.extra_allow),
            extra_deny=frozenset(defn.extra_deny),
        )

    return policies


# Build once at module load — same behavior as before
AGENT_EXEC_POLICIES: dict[str, AgentExecPolicy] = _build_exec_policies()

# Default policy for unknown/custom agents — preserves existing behavior
DEFAULT_EXEC_POLICY = AgentExecPolicy(SecurityLevel.ALLOWLIST)


def _get_agent_policy(agent_type: str) -> AgentExecPolicy:
    """Get the exec policy for an agent, with spec-level override support.

    Lookup order:
    1. AGENT_EXEC_POLICIES (built-in)
    2. DEFAULT_EXEC_POLICY (unknown/custom agents)

    Args:
        agent_type: Agent type identifier.

    Returns:
        AgentExecPolicy for the agent.
    """
    return AGENT_EXEC_POLICIES.get(agent_type, DEFAULT_EXEC_POLICY)


def _load_spec_override(
    spec_dir: Path | None, agent_type: str
) -> AgentExecPolicy | None:
    """Load optional per-spec exec policy override from task_metadata.json.

    Looks for:
        {"execPolicy": {"<agent_type>": {"securityLevel": "readonly", ...}}}

    Args:
        spec_dir: Spec directory (may be None).
        agent_type: Agent type to look up.

    Returns:
        AgentExecPolicy if override found, None otherwise.
    """
    if not spec_dir:
        return None

    metadata_path = spec_dir / "task_metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None

    exec_policies = metadata.get("execPolicy", {})
    agent_override = exec_policies.get(agent_type)
    if not agent_override or not isinstance(agent_override, dict):
        return None

    level_str = agent_override.get("securityLevel", "allowlist")
    try:
        level = SecurityLevel(level_str)
    except ValueError:
        return None

    extra_allow = frozenset(agent_override.get("extraAllow", []))
    extra_deny = frozenset(agent_override.get("extraDeny", []))

    return AgentExecPolicy(level, extra_allow, extra_deny)


def evaluate_exec_policy(
    command: str,
    agent_type: str,
    project_dir: Path | None = None,
    spec_dir: Path | None = None,
) -> ExecDecision:
    """Evaluate whether a bash command is allowed for the given agent.

    Cascade:
    1. Extract command names from the shell string.
    2. Check spec-level override (task_metadata.json execPolicy).
    3. Apply agent-level policy (DENY/READONLY/ALLOWLIST/FULL).
    4. For READONLY: only SAFE_BINS pass. Others are blocked.
    5. For ALLOWLIST/FULL: all commands pass (security/ module handles the rest).

    This function does NOT replace the security/ module — it's an additional
    layer that runs BEFORE the existing bash_security_hook.

    Args:
        command: The bash command string.
        agent_type: Agent type identifier.
        project_dir: Project directory (for future profile lookup).
        spec_dir: Spec directory (for override lookup).

    Returns:
        ExecDecision with allowed/reason and per-segment breakdown.
    """
    # Import parser lazily to avoid circular imports
    from security.parser import extract_commands, split_command_segments

    # Resolve policy: spec override > built-in
    policy = _load_spec_override(spec_dir, agent_type) or _get_agent_policy(agent_type)

    # DENY: Block everything immediately
    if policy.security_level == SecurityLevel.DENY:
        return ExecDecision(
            allowed=False,
            reason=f"Agent '{agent_type}' has DENY exec policy — all bash commands blocked.",
            agent_type=agent_type,
            policy_source="agent_policy:deny",
        )

    # Parse command segments
    raw_segments = split_command_segments(command)
    segments: list[CommandSegment] = []

    for raw_seg in raw_segments:
        seg_commands = extract_commands(raw_seg)
        for cmd in seg_commands:
            is_safe = cmd in SAFE_BINS
            is_extra_allowed = cmd in policy.extra_allow
            is_extra_denied = cmd in policy.extra_deny

            # Extra deny always wins
            if is_extra_denied:
                segments.append(CommandSegment(
                    raw=raw_seg, command=cmd, allowed=False,
                    reason=f"Command '{cmd}' explicitly denied for agent '{agent_type}'.",
                ))
                continue

            if policy.security_level == SecurityLevel.READONLY:
                # READONLY: only safe bins + extra_allow pass
                allowed = is_safe or is_extra_allowed
                reason = (
                    f"Safe binary" if is_safe
                    else f"Extra-allowed for {agent_type}" if is_extra_allowed
                    else f"Command '{cmd}' blocked — agent '{agent_type}' is READONLY."
                )
            else:
                # ALLOWLIST / FULL: allow everything at this layer
                # The security/ module will do its own validation
                allowed = True
                reason = "Allowed by agent policy (defers to security profile)."

            segments.append(CommandSegment(
                raw=raw_seg, command=cmd, allowed=allowed,
                reason=reason, is_safe_bin=is_safe,
            ))

    # Final decision: blocked if ANY segment is blocked
    blocked_segments = [s for s in segments if not s.allowed]
    if blocked_segments:
        reasons = "; ".join(s.reason for s in blocked_segments)
        return ExecDecision(
            allowed=False,
            reason=reasons,
            agent_type=agent_type,
            segments=segments,
            policy_source=f"agent_policy:{policy.security_level.value}",
        )

    return ExecDecision(
        allowed=True,
        reason=f"All commands allowed for agent '{agent_type}' ({policy.security_level.value}).",
        agent_type=agent_type,
        segments=segments,
        policy_source=f"agent_policy:{policy.security_level.value}",
    )
