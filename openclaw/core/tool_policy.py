"""
Tool Allow/Deny Policy (OpenClaw P3.3 + P0 Absorption)
========================================================

Per-agent tool filtering with glob patterns, group expansion,
deny-first evaluation, standard tool groups, and tool profiles.

Usage:
    from core.tool_policy import load_tool_policy, filter_tools
    from core.tool_policy import ToolProfile, get_profile_tools

    policy = load_tool_policy(spec_dir, "coder")
    allowed_tools = filter_tools(policy, all_tools)

    # Get tools for a standard profile
    coding_tools = get_profile_tools(ToolProfile.CODING)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path


@dataclass
class ToolPolicy:
    """Tool access policy with allow/deny glob patterns.

    Evaluation order:
    1. Check deny patterns — if any match, tool is blocked
    2. Check allow patterns — if any match, tool is permitted
    3. If no allow patterns match, tool is blocked
    """

    allow: list[str] = field(default_factory=lambda: ["*"])
    deny: list[str] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)


# Default permissive policy (existing behavior)
DEFAULT_POLICY = ToolPolicy(allow=["*"], deny=[])


# ── Standard tool groups (built-in, always available) ─────────────────────────

STANDARD_GROUPS: dict[str, list[str]] = {
    "fs_read": ["Read", "Glob", "Grep"],
    "fs_write": ["Write", "Edit"],
    "runtime": ["Bash"],
    "web": ["WebFetch", "WebSearch"],
    "memory": ["mcp__graphiti-memory__*", "mcp__auto-claude__record_*"],
    "docs": ["mcp__context7__*"],
    "browser": ["mcp__playwright__*", "mcp__electron__*", "mcp__marionette__*"],
    "progress": [
        "mcp__auto-claude__get_build_progress",
        "mcp__auto-claude__update_subtask_status",
    ],
    "qa": ["mcp__auto-claude__update_qa_status"],
    "design": [
        "mcp__auto-claude__create_child_spec",
        "mcp__auto-claude__create_batch_child_specs",
    ],
}


# ── Tool profiles (preset tool sets for common agent roles) ───────────────────

class ToolProfile(Enum):
    """Preset tool configurations for common agent roles.

    Each profile maps to a set of standard groups. Use get_profile_tools()
    to resolve a profile into a flat list of tool patterns.
    """

    MINIMAL = "minimal"    # Read-only filesystem + web
    READONLY = "readonly"  # Minimal + docs
    CODING = "coding"      # Full dev: fs + runtime + web + docs + memory + progress
    QA = "qa"              # Coding + browser + QA tools
    FULL = "full"          # Everything (wildcard)


_PROFILE_GROUPS: dict[ToolProfile, list[str]] = {
    ToolProfile.MINIMAL: ["@fs_read", "@web"],
    ToolProfile.READONLY: ["@fs_read", "@docs", "@web"],
    ToolProfile.CODING: [
        "@fs_read", "@fs_write", "@runtime", "@web",
        "@docs", "@memory", "@progress",
    ],
    ToolProfile.QA: [
        "@fs_read", "@fs_write", "@runtime", "@web",
        "@docs", "@memory", "@progress", "@browser", "@qa",
    ],
    ToolProfile.FULL: ["*"],
}


def get_profile_tools(profile: ToolProfile) -> list[str]:
    """Resolve a ToolProfile into a flat list of tool patterns.

    Args:
        profile: The ToolProfile to resolve.

    Returns:
        List of tool name patterns (may include globs like "mcp__*__*").
    """
    patterns = _PROFILE_GROUPS.get(profile, ["*"])
    return _expand_groups(patterns, STANDARD_GROUPS)


def _expand_groups(patterns: list[str], groups: dict[str, list[str]]) -> list[str]:
    """Expand @group references in pattern lists.

    Lookup order for group resolution:
    1. Provided groups dict (spec-level toolGroups from task_metadata.json)
    2. STANDARD_GROUPS (built-in fallback)

    Args:
        patterns: List of patterns, may contain "@group_name" references.
        groups: Group name → tool list mapping (spec-level overrides).

    Returns:
        Expanded list with group references replaced by their contents.
    """
    expanded = []
    for pattern in patterns:
        if pattern.startswith("@"):
            group_name = pattern[1:]
            if group_name in groups:
                expanded.extend(groups[group_name])
            elif group_name in STANDARD_GROUPS:
                expanded.extend(STANDARD_GROUPS[group_name])
            else:
                expanded.append(pattern)
        else:
            expanded.append(pattern)
    return expanded


def load_tool_policy(spec_dir: Path, agent_type: str) -> ToolPolicy:
    """Load tool policy from task_metadata.json for an agent type.

    Reads the "toolPolicy" and "toolGroups" keys from task_metadata.json.
    Returns DEFAULT_POLICY if no policy is defined (opt-in activation).

    Args:
        spec_dir: Spec directory containing task_metadata.json.
        agent_type: Agent type (e.g., "coder", "planner").

    Returns:
        ToolPolicy for the specified agent type.
    """
    metadata_path = spec_dir / "task_metadata.json"
    if not metadata_path.exists():
        return DEFAULT_POLICY

    try:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return DEFAULT_POLICY

    # Load global tool groups
    groups = metadata.get("toolGroups", {})

    # Load agent-specific policy
    policies = metadata.get("toolPolicy", {})
    agent_policy = policies.get(agent_type)
    if not agent_policy:
        return DEFAULT_POLICY

    allow = agent_policy.get("allow", ["*"])
    deny = agent_policy.get("deny", [])

    # Expand group references
    allow = _expand_groups(allow, groups)
    deny = _expand_groups(deny, groups)

    return ToolPolicy(allow=allow, deny=deny, groups=groups)


def is_tool_allowed(policy: ToolPolicy, tool_name: str) -> bool:
    """Check if a tool is allowed by the policy.

    Evaluation: deny first, then allow.

    Args:
        policy: The ToolPolicy to evaluate against.
        tool_name: Full tool name (e.g., "Read", "bash:git push*").

    Returns:
        True if the tool is allowed, False if denied.
    """
    # Deny takes precedence
    for pattern in policy.deny:
        if fnmatch(tool_name, pattern):
            return False

    # Check allow patterns
    for pattern in policy.allow:
        if fnmatch(tool_name, pattern):
            return True

    # No allow match → blocked
    return False


def filter_tools(policy: ToolPolicy, tools: list[str]) -> list[str]:
    """Filter a list of tool names through the policy.

    Args:
        policy: The ToolPolicy to apply.
        tools: List of tool name strings.

    Returns:
        Filtered list containing only allowed tools.
    """
    return [t for t in tools if is_tool_allowed(policy, t)]


def log_tool_blocked(spec_dir: Path, tool_name: str, agent_type: str) -> None:
    """Log a tool block event to events.jsonl.

    Args:
        spec_dir: Spec directory for events.jsonl.
        tool_name: The blocked tool name.
        agent_type: Agent that attempted to use the tool.
    """
    try:
        from core.task_event import append_event_log

        append_event_log(spec_dir, {
            "type": "TOOL_BLOCKED",
            "toolName": tool_name,
            "agentType": agent_type,
        })
    except Exception:
        pass  # Non-critical: best-effort logging
