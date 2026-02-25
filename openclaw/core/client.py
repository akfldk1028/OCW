"""
Claude SDK Client Configuration
===============================

Functions for creating and configuring the Claude Agent SDK client.

All AI interactions should use `create_client()` to ensure consistent OAuth authentication
and proper tool/MCP configuration. For simple message calls without full agent sessions,
use `create_simple_client()` from `core.simple_client`.

The client factory now uses AGENT_CONFIGS from agents/tools_pkg/models.py as the
single source of truth for phase-aware tool and MCP server configuration.
"""

import copy
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from core.platform import (
    is_windows,
    validate_cli_path,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Project Index Cache
# =============================================================================
# Caches project index and capabilities to avoid reloading on every create_client() call.
# This significantly reduces the time to create new agent sessions.

_PROJECT_INDEX_CACHE: dict[str, tuple[dict[str, Any], dict[str, bool], float]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minute TTL
_CACHE_LOCK = threading.Lock()  # Protects _PROJECT_INDEX_CACHE access


def _get_cached_project_data(
    project_dir: Path,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """
    Get project index and capabilities with caching.

    Args:
        project_dir: Path to the project directory

    Returns:
        Tuple of (project_index, project_capabilities)
    """

    key = str(project_dir.resolve())
    now = time.time()
    debug = os.environ.get("DEBUG", "").lower() in ("true", "1")

    # Check cache with lock
    with _CACHE_LOCK:
        if key in _PROJECT_INDEX_CACHE:
            cached_index, cached_capabilities, cached_time = _PROJECT_INDEX_CACHE[key]
            cache_age = now - cached_time
            if cache_age < _CACHE_TTL_SECONDS:
                if debug:
                    print(
                        f"[ClientCache] Cache HIT for project index (age: {cache_age:.1f}s / TTL: {_CACHE_TTL_SECONDS}s)"
                    )
                logger.debug(f"Using cached project index for {project_dir}")
                # Return deep copies to prevent callers from corrupting the cache
                return copy.deepcopy(cached_index), copy.deepcopy(cached_capabilities)
            elif debug:
                print(
                    f"[ClientCache] Cache EXPIRED for project index (age: {cache_age:.1f}s > TTL: {_CACHE_TTL_SECONDS}s)"
                )

    # Cache miss or expired - load fresh data (outside lock to avoid blocking)
    load_start = time.time()
    logger.debug(f"Loading project index for {project_dir}")
    project_index = load_project_index(project_dir)
    project_capabilities = detect_project_capabilities(project_index)

    if debug:
        load_duration = (time.time() - load_start) * 1000
        print(
            f"[ClientCache] Cache MISS - loaded project index in {load_duration:.1f}ms"
        )

    # Store in cache with lock - use double-checked locking pattern
    # Re-check if another thread populated the cache while we were loading
    with _CACHE_LOCK:
        if key in _PROJECT_INDEX_CACHE:
            cached_index, cached_capabilities, cached_time = _PROJECT_INDEX_CACHE[key]
            cache_age = time.time() - cached_time
            if cache_age < _CACHE_TTL_SECONDS:
                # Another thread already cached valid data while we were loading
                if debug:
                    print(
                        "[ClientCache] Cache was populated by another thread, using cached data"
                    )
                # Return deep copies to prevent callers from corrupting the cache
                return copy.deepcopy(cached_index), copy.deepcopy(cached_capabilities)
        # Either no cache entry or it's expired - store our fresh data
        _PROJECT_INDEX_CACHE[key] = (project_index, project_capabilities, time.time())

    # Return the freshly loaded data (no need to copy since it's not from cache)
    return project_index, project_capabilities


def invalidate_project_cache(project_dir: Path | None = None) -> None:
    """
    Invalidate the project index cache.

    Args:
        project_dir: Specific project to invalidate, or None to clear all
    """
    with _CACHE_LOCK:
        if project_dir is None:
            _PROJECT_INDEX_CACHE.clear()
            logger.debug("Cleared all project index cache entries")
        else:
            key = str(project_dir.resolve())
            if key in _PROJECT_INDEX_CACHE:
                del _PROJECT_INDEX_CACHE[key]
                logger.debug(f"Invalidated project index cache for {project_dir}")


from agents.tools_pkg import (
    CONTEXT7_TOOLS,
    ELECTRON_TOOLS,
    GRAPHITI_MCP_TOOLS,
    LINEAR_TOOLS,
    PLAYWRIGHT_TOOLS,
    create_auto_claude_mcp_server,
    get_allowed_tools,
    get_required_mcp_servers,
    is_tools_available,
)
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import HookMatcher
from core.auth import (
    configure_sdk_authentication,
    get_sdk_env_vars,
)
from linear_updater import is_linear_enabled
from prompts_pkg.project_context import detect_project_capabilities, load_project_index
from core.exec_policy import evaluate_exec_policy
from security import bash_security_hook


def _validate_custom_mcp_server(server: dict) -> bool:
    """
    Validate a custom MCP server configuration for security.

    Supports standard MCP JSON format (mcpServers style):
    - stdio: {"command": "npx", "args": [...], "env": {...}}
    - http/sse: {"type": "http"|"sse", "url": "...", "headers": {...}}

    Auto-Claude extensions (optional):
    - "id": Server identifier (defaults to server name key)
    - "name": Display name
    - "tools": Tool whitelist ("*" or ["tool1", "tool2"])
    - "agents": Agent whitelist ("*" or ["planner", "coder"])

    Args:
        server: Dict representing a custom MCP server configuration

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(server, dict):
        return False

    # Must have either "command" (stdio) or "url" (http/sse)
    has_command = "command" in server
    has_url = "url" in server

    if not has_command and not has_url:
        logger.warning(
            "Custom MCP server must have either 'command' (stdio) or 'url' (http/sse)"
        )
        return False

    # Determine server type
    # - If "command" present without "type" → stdio (MCP standard: type is optional for stdio)
    # - If "type" is "http" or "sse" → remote server
    server_type = server.get("type")
    if has_command and server_type is None:
        server_type = "stdio"  # MCP standard default
    elif has_url and server_type not in ("http", "sse"):
        logger.warning(f"URL-based server must have type 'http' or 'sse', got: {server_type}")
        return False

    # Allowlist of safe executable commands for MCP servers
    # Only allow known package managers and interpreters - NO shell commands
    SAFE_COMMANDS = {
        "npx",
        "npm",
        "node",
        "python",
        "python3",
        "uv",
        "uvx",
    }

    # Blocklist of dangerous shell commands that should never be allowed
    DANGEROUS_COMMANDS = {
        "bash",
        "sh",
        "cmd",
        "powershell",
        "pwsh",  # PowerShell Core
        "/bin/bash",
        "/bin/sh",
        "/bin/zsh",
        "/usr/bin/bash",
        "/usr/bin/sh",
        "zsh",
        "fish",
    }

    # Dangerous interpreter flags that allow arbitrary code execution
    # Covers Python (-e, -c, -m, -p), Node.js (--eval, --print, loaders), and general
    DANGEROUS_FLAGS = {
        "--eval",
        "-e",
        "-c",
        "--exec",
        "-m",  # Python module execution
        "-p",  # Python eval+print
        "--print",  # Node.js print
        "--input-type=module",  # Node.js ES module mode
        "--experimental-loader",  # Node.js custom loaders
        "--require",  # Node.js require injection
        "-r",  # Node.js require shorthand
    }

    # Type-specific validation
    if server_type == "stdio" or has_command:
        # stdio server (local process via command)
        if not isinstance(server.get("command"), str) or not server["command"]:
            logger.warning("stdio MCP server missing 'command' field")
            return False

        # SECURITY FIX: Validate command is in safe list and not in dangerous list
        command = server.get("command", "")

        # Reject paths - commands must be bare names only (no / or \)
        # This prevents path traversal like '/custom/malicious' or './evil'
        if "/" in command or "\\" in command:
            logger.warning(
                f"Rejected command with path in MCP server: {command}. "
                f"Commands must be bare names without path separators."
            )
            return False

        if command in DANGEROUS_COMMANDS:
            logger.warning(
                f"Rejected dangerous command in MCP server: {command}. "
                f"Shell commands are not allowed for security reasons."
            )
            return False

        if command not in SAFE_COMMANDS:
            logger.warning(
                f"Rejected unknown command in MCP server: {command}. "
                f"Only allowed commands: {', '.join(sorted(SAFE_COMMANDS))}"
            )
            return False

        # Validate args is a list of strings if present
        if "args" in server:
            if not isinstance(server["args"], list):
                return False
            if not all(isinstance(arg, str) for arg in server["args"]):
                return False
            # Check for dangerous interpreter flags that allow code execution
            for arg in server["args"]:
                if arg in DANGEROUS_FLAGS:
                    logger.warning(
                        f"Rejected dangerous flag '{arg}' in MCP server args. "
                        f"Interpreter code execution flags are not allowed."
                    )
                    return False

        # Validate env is a dict of strings if present (MCP standard field)
        if "env" in server:
            if not isinstance(server["env"], dict):
                return False
            if not all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in server["env"].items()
            ):
                return False

    elif server_type in ("http", "sse"):
        # HTTP or SSE server (remote)
        if not isinstance(server.get("url"), str) or not server["url"]:
            logger.warning(f"{server_type.upper()} MCP server missing 'url' field")
            return False
        # Validate headers is a dict of strings if present
        if "headers" in server:
            if not isinstance(server["headers"], dict):
                return False
            if not all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in server["headers"].items()
            ):
                return False

    # Optional description must be string if present
    if "description" in server and not isinstance(server.get("description"), str):
        return False

    # Validate tools field if present
    # Can be "*" (wildcard) or list of tool names
    if "tools" in server:
        tools = server.get("tools")
        if tools != "*" and not isinstance(tools, list):
            logger.warning("Custom MCP server 'tools' must be '*' or list of tool names")
            return False
        if isinstance(tools, list):
            if not all(isinstance(t, str) for t in tools):
                logger.warning("Custom MCP server 'tools' list must contain strings only")
                return False

    # Validate agents field if present (list of agent types to enable for)
    if "agents" in server:
        agents = server.get("agents")
        if agents != "*" and not isinstance(agents, list):
            logger.warning("Custom MCP server 'agents' must be '*' or list of agent types")
            return False
        if isinstance(agents, list):
            if not all(isinstance(a, str) for a in agents):
                logger.warning("Custom MCP server 'agents' list must contain strings only")
                return False

    # Reject any unexpected fields that could be exploited
    # Standard MCP fields + Auto-Claude extensions
    allowed_fields = {
        # MCP standard fields
        "type",       # "stdio" (optional), "http", "sse"
        "command",    # stdio: executable command
        "args",       # stdio: command arguments
        "env",        # stdio: environment variables
        "url",        # http/sse: server URL
        "headers",    # http/sse: HTTP headers
        # Auto-Claude extension fields
        "id",         # Server identifier (used as key in mcp_servers dict)
        "name",       # Display name
        "description",  # Human-readable description
        "tools",      # Tool whitelist: "*" or ["tool1", "tool2"]
        "agents",     # Agent whitelist: "*" or ["planner", "coder"]
    }
    unexpected_fields = set(server.keys()) - allowed_fields
    if unexpected_fields:
        logger.warning(f"Custom MCP server has unexpected fields: {unexpected_fields}")
        return False

    return True


def load_project_mcp_config(project_dir: Path) -> dict:
    """
    Load MCP configuration from project's .auto-claude/.env file.

    Returns a dict of MCP-related env vars:
    - CONTEXT7_ENABLED (default: true)
    - LINEAR_MCP_ENABLED (default: true)
    - ELECTRON_MCP_ENABLED (default: false)
    - BROWSER_MCP_DISABLED (default: false) — opt-out for Playwright browser automation
    - AGENT_MCP_<agent>_ADD (per-agent MCP additions)
    - AGENT_MCP_<agent>_REMOVE (per-agent MCP removals)
    - CUSTOM_MCP_SERVERS (JSON array of custom server configs)

    Args:
        project_dir: Path to the project directory

    Returns:
        Dict of MCP configuration values (string values, except CUSTOM_MCP_SERVERS which is parsed JSON)
    """
    env_path = project_dir / ".auto-claude" / ".env"
    if not env_path.exists():
        return {}

    config = {}
    mcp_keys = {
        "CONTEXT7_ENABLED",
        "LINEAR_MCP_ENABLED",
        "ELECTRON_MCP_ENABLED",
        "BROWSER_MCP_DISABLED",
    }

    try:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    # Include global MCP toggles
                    if key in mcp_keys:
                        config[key] = value
                    # Include per-agent MCP overrides (AGENT_MCP_<agent>_ADD/REMOVE)
                    elif key.startswith("AGENT_MCP_"):
                        config[key] = value
                    # Include custom MCP servers (parse JSON with schema validation)
                    elif key == "CUSTOM_MCP_SERVERS":
                        try:
                            parsed = json.loads(value)
                            if not isinstance(parsed, list):
                                logger.warning(
                                    "CUSTOM_MCP_SERVERS must be a JSON array"
                                )
                                config["CUSTOM_MCP_SERVERS"] = []
                            else:
                                # Validate each server and filter out invalid ones
                                valid_servers = []
                                for i, server in enumerate(parsed):
                                    if _validate_custom_mcp_server(server):
                                        valid_servers.append(server)
                                    else:
                                        logger.warning(
                                            f"Skipping invalid custom MCP server at index {i}"
                                        )
                                config["CUSTOM_MCP_SERVERS"] = valid_servers
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse CUSTOM_MCP_SERVERS JSON: {value}"
                            )
                            config["CUSTOM_MCP_SERVERS"] = []
    except Exception as e:
        logger.debug(f"Failed to load project MCP config from {env_path}: {e}")

    return config


def is_graphiti_mcp_enabled() -> bool:
    """
    Check if Graphiti MCP server integration is enabled.

    Requires GRAPHITI_MCP_URL to be set (e.g., http://localhost:8000/mcp/)
    This is separate from GRAPHITI_ENABLED which controls the Python library integration.
    """
    return bool(os.environ.get("GRAPHITI_MCP_URL"))


def get_graphiti_mcp_url() -> str:
    """Get the Graphiti MCP server URL."""
    return os.environ.get("GRAPHITI_MCP_URL", "http://localhost:8000/mcp/")


def is_electron_mcp_enabled() -> bool:
    """
    Check if Electron MCP server integration is enabled.

    Requires ELECTRON_MCP_ENABLED to be set to 'true'.
    When enabled, QA agents can use Electron MCP tools to connect to Electron apps
    via Chrome DevTools Protocol on the configured debug port.
    """
    return os.environ.get("ELECTRON_MCP_ENABLED", "").lower() == "true"


def get_electron_debug_port() -> int:
    """Get the Electron remote debugging port (default: 9222)."""
    return int(os.environ.get("ELECTRON_DEBUG_PORT", "9222"))


def should_use_claude_md() -> bool:
    """Check if CLAUDE.md instructions should be included in system prompt."""
    return os.environ.get("USE_CLAUDE_MD", "").lower() == "true"


def _get_custom_mcp_tool_permissions(
    mcp_config: dict | None, required_servers: list[str]
) -> list[str]:
    """
    Get permission strings for custom MCP server tools.

    Args:
        mcp_config: Per-project MCP config containing CUSTOM_MCP_SERVERS
        required_servers: List of servers that will be started

    Returns:
        List of permission strings like "mcp__server__tool(*)"
    """
    if not mcp_config:
        return []

    permissions = []
    for custom in mcp_config.get("CUSTOM_MCP_SERVERS", []):
        server_id = custom.get("id")
        if not server_id or server_id not in required_servers:
            continue

        custom_tools = custom.get("tools", [])
        if custom_tools == "*":
            # Wildcard: allow all tools from this server
            permissions.append(f"mcp__{server_id}__*(*)")
        elif isinstance(custom_tools, list):
            # Explicit tool list
            for tool in custom_tools:
                if isinstance(tool, str):
                    if tool.startswith("mcp__"):
                        permissions.append(f"{tool}(*)")
                    else:
                        permissions.append(f"mcp__{server_id}__{tool}(*)")

    return permissions


def load_claude_md(project_dir: Path) -> str | None:
    """
    Load CLAUDE.md content with fallback chain.

    Priority:
        1. Project-level: project_dir/CLAUDE.md
        2. Global fallback: $CLAUDE_CONFIG_DIR/CLAUDE.md (or ~/.claude/CLAUDE.md)

    Args:
        project_dir: Root directory of the project

    Returns:
        Content of CLAUDE.md if found, None otherwise
    """
    # 1. Project-level (highest priority)
    claude_md_path = project_dir / "CLAUDE.md"
    if claude_md_path.exists():
        try:
            return claude_md_path.read_text(encoding="utf-8")
        except Exception:
            pass

    # 2. Global fallback ($CLAUDE_CONFIG_DIR/CLAUDE.md)
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR", "")
    if not config_dir:
        config_dir = str(Path.home() / ".claude")
    global_md = Path(config_dir) / "CLAUDE.md"
    if global_md.exists():
        try:
            return global_md.read_text(encoding="utf-8")
        except Exception:
            pass

    return None


def _create_exec_policy_hook(
    agent_type: str, project_dir: Path, spec_dir: Path
):
    """Create an agent-aware bash security hook (closure factory).

    Returns an async hook function that:
    1. Evaluates exec_policy for the agent (DENY/READONLY/ALLOWLIST/FULL).
    2. If DENY or READONLY blocks → immediately deny with reason.
    3. If ALLOWLIST/FULL allows → delegates to the existing bash_security_hook
       for defense-in-depth (SecurityProfile validation, command validators).
    4. Logs EXEC_BLOCKED events to events.jsonl on denial.

    Args:
        agent_type: Agent type identifier (e.g., "coder", "spec_gatherer").
        project_dir: Project directory.
        spec_dir: Spec directory for override lookup.

    Returns:
        Async hook function compatible with SDK PreToolUse hooks.
    """

    # Pre-compute worktree detection for the hook closure
    _is_worktree = (project_dir / ".git").is_file()  # Worktrees have .git as file, not dir

    async def exec_policy_hook(
        input_data: dict, tool_use_id: str | None = None, context=None
    ) -> dict:
        # Only intercept Bash tool calls
        tool_name = input_data.get("tool_name", "")
        if tool_name != "Bash":
            return {}

        tool_input = input_data.get("tool_input")
        if not tool_input or not isinstance(tool_input, dict):
            return {}

        command = tool_input.get("command", "")
        if not command:
            return {}

        # Block dangerous git commands in worktree mode (Bug #18: agent bypasses merge flow)
        if _is_worktree:
            import re
            cmd_lower = command.lower().strip()
            # Block: git merge, git checkout <base-branch>, git push, git rebase
            worktree_forbidden_patterns = [
                r'\bgit\s+merge\b',
                r'\bgit\s+checkout\s+(main|master|develop)\b',
                r'\bgit\s+push\b',
                r'\bgit\s+rebase\b',
            ]
            for pattern in worktree_forbidden_patterns:
                if re.search(pattern, cmd_lower):
                    return {
                        "hookSpecificOutput": {
                            "permissionDecision": "deny",
                            "permissionDecisionReason": (
                                f"BLOCKED: '{command[:80]}' is forbidden in worktree mode. "
                                "Merging/pushing is handled automatically after QA passes. "
                                "Only use git add/commit/status/diff in your worktree."
                            ),
                        }
                    }

        # Evaluate agent-level exec policy
        decision = evaluate_exec_policy(
            command, agent_type, project_dir, spec_dir
        )

        if not decision.allowed:
            # Log blocked execution to events.jsonl
            try:
                from core.task_event import append_event_log

                append_event_log(spec_dir, {
                    "type": "EXEC_BLOCKED",
                    "agentType": agent_type,
                    "command": command[:200],
                    "reason": decision.reason,
                    "policySource": decision.policy_source,
                })
            except Exception:
                pass  # Non-critical: best-effort logging

            return {
                "hookSpecificOutput": {
                    "permissionDecision": "deny",
                    "permissionDecisionReason": decision.reason,
                }
            }

        # Agent-level policy allows → delegate to existing security hook
        # This provides defense-in-depth: even FULL agents go through
        # SecurityProfile validation, command validators, secret scanning, etc.
        return await bash_security_hook(input_data, tool_use_id, context)

    return exec_policy_hook


def create_client(
    project_dir: Path,
    spec_dir: Path,
    model: str,
    agent_type: str = "coder",
    max_thinking_tokens: int | None = None,
    output_format: dict | None = None,
    agents: dict | None = None,
) -> ClaudeSDKClient:
    """
    Create a Claude Agent SDK client with multi-layered security.

    Uses AGENT_CONFIGS for phase-aware tool and MCP server configuration.
    Only starts MCP servers that the agent actually needs, reducing context
    window bloat and startup latency.

    Args:
        project_dir: Root directory for the project (working directory)
        spec_dir: Directory containing the spec (for settings file)
        model: Claude model to use
        agent_type: Agent type identifier from AGENT_CONFIGS
                   (e.g., 'coder', 'planner', 'qa_reviewer', 'spec_gatherer')
        max_thinking_tokens: Token budget for extended thinking (None = disabled)
                            - ultrathink: 16000 (spec creation)
                            - high: 10000 (QA review)
                            - medium: 5000 (planning, validation)
                            - None: disabled (coding)
        output_format: Optional structured output format for validated JSON responses.
                      Use {"type": "json_schema", "schema": Model.model_json_schema()}
                      See: https://platform.claude.com/docs/en/agent-sdk/structured-outputs
        agents: Optional dict of subagent definitions for SDK parallel execution.
               Format: {"agent-name": {"description": "...", "prompt": "...",
                        "tools": [...], "model": "inherit"}}
               See: https://platform.claude.com/docs/en/agent-sdk/subagents

    Returns:
        Configured ClaudeSDKClient

    Raises:
        ValueError: If agent_type is not found in AGENT_CONFIGS

    Security layers (defense in depth):
    1. Sandbox - OS-level bash command isolation prevents filesystem escape
    2. Permissions - File operations restricted to project_dir only
    3. Security hooks - Bash commands validated against an allowlist
       (see security.py for ALLOWED_COMMANDS)
    4. Tool filtering - Each agent type only sees relevant tools (prevents misuse)
    """
    # Collect env vars to pass to SDK (ANTHROPIC_BASE_URL, CLAUDE_CONFIG_DIR, etc.)
    sdk_env = get_sdk_env_vars()

    # Get the config dir for profile-specific credential lookup
    # CLAUDE_CONFIG_DIR enables per-profile Keychain entries with SHA256-hashed service names
    config_dir = sdk_env.get("CLAUDE_CONFIG_DIR")

    # Configure SDK authentication (OAuth or API profile mode)
    configure_sdk_authentication(config_dir)

    if config_dir:
        logger.info(f"Using CLAUDE_CONFIG_DIR for profile: {config_dir}")

    # Debug: Log git-bash path detection on Windows
    if "CLAUDE_CODE_GIT_BASH_PATH" in sdk_env:
        logger.info(f"Git Bash path found: {sdk_env['CLAUDE_CODE_GIT_BASH_PATH']}")
    elif is_windows():
        logger.warning("Git Bash path not detected on Windows!")

    # Check if Linear integration is enabled
    linear_enabled = is_linear_enabled()
    linear_api_key = os.environ.get("LINEAR_API_KEY", "")

    # Check if custom auto-claude tools are available
    auto_claude_tools_enabled = is_tools_available()

    # Load project capabilities for dynamic MCP tool selection
    # This enables context-aware tool injection based on project type
    # Uses caching to avoid reloading on every create_client() call
    project_index, project_capabilities = _get_cached_project_data(project_dir)

    # Load per-project MCP configuration from .auto-claude/.env
    mcp_config = load_project_mcp_config(project_dir)

    # Get allowed tools using phase-aware configuration
    # This respects AGENT_CONFIGS and only includes tools the agent needs
    # Also respects per-project MCP configuration
    allowed_tools_list = get_allowed_tools(
        agent_type,
        project_capabilities,
        linear_enabled,
        mcp_config,
    )

    # Get required MCP servers for this agent type
    # This is the key optimization - only start servers the agent needs
    # Now also respects per-project MCP configuration
    required_servers = get_required_mcp_servers(
        agent_type,
        project_capabilities,
        linear_enabled,
        mcp_config,
    )

    # Check if Graphiti MCP is enabled (already filtered by get_required_mcp_servers)
    graphiti_mcp_enabled = "graphiti" in required_servers

    # Determine browser tools for permissions (already in allowed_tools_list)
    browser_tools_permissions = []
    if "electron" in required_servers:
        browser_tools_permissions = ELECTRON_TOOLS
    elif "playwright" in required_servers:
        browser_tools_permissions = PLAYWRIGHT_TOOLS

    # Create comprehensive security settings
    # Note: Using both relative paths ("./**") and absolute paths to handle
    # cases where Claude uses absolute paths for file operations
    project_path_str = str(project_dir.resolve())
    spec_path_str = str(spec_dir.resolve())

    # Detect if we're running in a worktree and get the original project directory
    # Worktrees are located in either:
    # - .auto-claude/worktrees/tasks/{spec-name}/ (new location)
    # - .worktrees/{spec-name}/ (legacy location)
    # When running in a worktree, we need to allow access to both the worktree
    # and the original project's .auto-claude/ directory for spec files
    original_project_permissions = []
    resolved_project_path = project_dir.resolve()

    # Check for worktree paths and extract original project directory
    # This handles spec worktrees, PR review worktrees, and legacy worktrees
    # Note: Windows paths are normalized to forward slashes before comparison
    worktree_markers = [
        "/.auto-claude/worktrees/tasks/",  # Spec/task worktrees
        "/.auto-claude/github/pr/worktrees/",  # PR review worktrees
        "/.worktrees/",  # Legacy worktree location
    ]
    project_path_posix = str(resolved_project_path).replace("\\", "/")

    for marker in worktree_markers:
        if marker in project_path_posix:
            # Extract the original project directory (parent of worktree location)
            # Use rsplit to get the rightmost occurrence (handles nested projects)
            original_project_str = project_path_posix.rsplit(marker, 1)[0]
            original_project_dir = Path(original_project_str)

            # Grant permissions for relevant directories in the original project
            permission_ops = ["Read", "Write", "Edit", "Glob", "Grep"]
            dirs_to_permit = [
                original_project_dir / ".auto-claude",
                original_project_dir / ".worktrees",  # Legacy support
            ]

            for dir_path in dirs_to_permit:
                if dir_path.exists():
                    path_str = str(dir_path.resolve())
                    original_project_permissions.extend(
                        [f"{op}({path_str}/**)" for op in permission_ops]
                    )
            break

    security_settings = {
        "sandbox": {"enabled": True, "autoAllowBashIfSandboxed": True},
        "permissions": {
            "defaultMode": "acceptEdits",  # Auto-approve edits within allowed directories
            "allow": [
                # Allow all file operations within the project directory
                # Include both relative (./**) and absolute paths for compatibility
                "Read(./**)",
                "Write(./**)",
                "Edit(./**)",
                "Glob(./**)",
                "Grep(./**)",
                # Also allow absolute paths (Claude sometimes uses full paths)
                f"Read({project_path_str}/**)",
                f"Write({project_path_str}/**)",
                f"Edit({project_path_str}/**)",
                f"Glob({project_path_str}/**)",
                f"Grep({project_path_str}/**)",
                # Allow spec directory explicitly (needed when spec is in worktree)
                f"Read({spec_path_str}/**)",
                f"Write({spec_path_str}/**)",
                f"Edit({spec_path_str}/**)",
                # Allow original project's .auto-claude/ and .worktrees/ directories
                # when running in a worktree (fixes issue #385 - permission errors)
                *original_project_permissions,
                # Bash permission granted here, but actual commands are validated
                # by the bash_security_hook (see security.py for allowed commands)
                "Bash(*)",
                # Allow web tools for documentation and research
                "WebFetch(*)",
                "WebSearch(*)",
                # Allow MCP tools based on required servers
                # Format: tool_name(*) allows all arguments
                *(
                    [f"{tool}(*)" for tool in CONTEXT7_TOOLS]
                    if "context7" in required_servers
                    else []
                ),
                *(
                    [f"{tool}(*)" for tool in LINEAR_TOOLS]
                    if "linear" in required_servers
                    else []
                ),
                *(
                    [f"{tool}(*)" for tool in GRAPHITI_MCP_TOOLS]
                    if graphiti_mcp_enabled
                    else []
                ),
                *[f"{tool}(*)" for tool in browser_tools_permissions],
                # Add permissions for custom MCP server tools
                *_get_custom_mcp_tool_permissions(mcp_config, required_servers),
            ],
        },
    }

    # Write settings to a file in the project directory
    settings_file = project_dir / ".claude_settings.json"
    with open(settings_file, "w", encoding="utf-8") as f:
        json.dump(security_settings, f, indent=2)

    print(f"Security settings: {settings_file}")
    print("   - Sandbox enabled (OS-level bash isolation)")
    print(f"   - Filesystem restricted to: {project_dir.resolve()}")
    if original_project_permissions:
        print("   - Worktree permissions: granted for original project directories")
    print("   - Bash commands restricted to allowlist")
    if max_thinking_tokens:
        print(f"   - Extended thinking: {max_thinking_tokens:,} tokens")
    else:
        print("   - Extended thinking: disabled")

    # Build list of MCP servers for display based on required_servers
    mcp_servers_list = []
    if "context7" in required_servers:
        mcp_servers_list.append("context7 (documentation)")
    if "electron" in required_servers:
        mcp_servers_list.append(
            f"electron (desktop automation, port {get_electron_debug_port()})"
        )
    if "playwright" in required_servers:
        mcp_servers_list.append("playwright (browser automation)")
    if "linear" in required_servers:
        mcp_servers_list.append("linear (project management)")
    if graphiti_mcp_enabled:
        mcp_servers_list.append("graphiti-memory (knowledge graph)")
    if "auto-claude" in required_servers and auto_claude_tools_enabled:
        mcp_servers_list.append(f"auto-claude ({agent_type} tools)")

    # Add custom MCP servers to display
    custom_servers = mcp_config.get("CUSTOM_MCP_SERVERS", [])
    for custom in custom_servers:
        server_id = custom.get("id")
        if server_id and server_id in required_servers:
            server_name = custom.get("name", server_id)
            server_desc = custom.get("description", "custom")
            mcp_servers_list.append(f"{server_name} ({server_desc})")

    if mcp_servers_list:
        print(f"   - MCP servers: {', '.join(mcp_servers_list)}")
    else:
        print("   - MCP servers: none (minimal configuration)")

    # Show detected project capabilities for QA agents
    if agent_type in ("qa_reviewer", "qa_fixer") and any(project_capabilities.values()):
        caps = [
            k.replace("is_", "").replace("has_", "")
            for k, v in project_capabilities.items()
            if v
        ]
        print(f"   - Project capabilities: {', '.join(caps)}")
    print()

    # Configure MCP servers - ONLY start servers that are required
    # This is the key optimization to reduce context bloat and startup latency
    mcp_servers = {}

    if "context7" in required_servers:
        mcp_servers["context7"] = {
            "command": "npx",
            "args": ["-y", "@upstash/context7-mcp"],
        }

    if "electron" in required_servers:
        # Electron MCP for desktop apps
        # Electron app must be started with --remote-debugging-port=<port>
        mcp_servers["electron"] = {
            "command": "npm",
            "args": ["exec", "electron-mcp-server"],
        }

    if "playwright" in required_servers:
        # Playwright MCP for web frontends (not Electron)
        import platform as _platform

        if _platform.system() == "Windows":
            mcp_servers["playwright"] = {
                "command": "cmd",
                "args": ["/c", "npx", "@playwright/mcp@latest", "--headless"],
            }
        else:
            mcp_servers["playwright"] = {
                "command": "npx",
                "args": ["@playwright/mcp@latest", "--headless"],
            }

    if "marionette" in required_servers:
        # Marionette MCP for Flutter widget-level interaction (leancodepl/marionette_mcp)
        # Requires: dart pub global activate marionette_mcp
        # Connects to running Flutter app via VM Service protocol
        import platform as _platform2
        import shutil

        # Find marionette_mcp in PATH or Dart pub global cache
        marionette_cmd = shutil.which("marionette_mcp")
        if not marionette_cmd and _platform2.system() == "Windows":
            # Dart pub global installs .bat files on Windows
            pub_cache_bin = Path.home() / "AppData" / "Local" / "Pub" / "Cache" / "bin"
            bat_path = pub_cache_bin / "marionette_mcp.bat"
            if bat_path.exists():
                marionette_cmd = str(bat_path)

        if marionette_cmd:
            if _platform2.system() == "Windows":
                mcp_servers["marionette"] = {
                    "command": "cmd",
                    "args": ["/c", marionette_cmd],
                }
            else:
                mcp_servers["marionette"] = {
                    "command": marionette_cmd,
                    "args": [],
                }
        else:
            logger.warning(
                "marionette_mcp not found — install with: dart pub global activate marionette_mcp"
            )

    if "linear" in required_servers:
        mcp_servers["linear"] = {
            "type": "http",
            "url": "https://mcp.linear.app/mcp",
            "headers": {"Authorization": f"Bearer {linear_api_key}"},
        }

    # Graphiti MCP server for knowledge graph memory
    if graphiti_mcp_enabled:
        mcp_servers["graphiti-memory"] = {
            "type": "http",
            "url": get_graphiti_mcp_url(),
        }

    # Add custom auto-claude MCP server if required and available
    if "auto-claude" in required_servers and auto_claude_tools_enabled:
        auto_claude_mcp_server = create_auto_claude_mcp_server(spec_dir, project_dir)
        if auto_claude_mcp_server:
            mcp_servers["auto-claude"] = auto_claude_mcp_server

    # Add custom MCP servers from project config (MCP standard format)
    custom_servers = mcp_config.get("CUSTOM_MCP_SERVERS", [])
    for custom in custom_servers:
        server_id = custom.get("id")
        if not server_id:
            continue
        # Only include if agent has it in their effective server list
        if server_id not in required_servers:
            continue

        # Determine type: stdio (command-based) or http/sse (URL-based)
        has_command = "command" in custom
        has_url = "url" in custom
        server_type = custom.get("type")

        if has_command:
            # stdio server (MCP standard: type is optional for command-based)
            server_config = {
                "command": custom.get("command", "npx"),
                "args": custom.get("args", []),
            }
            # Add env if present (MCP standard field)
            if custom.get("env"):
                server_config["env"] = custom["env"]
            mcp_servers[server_id] = server_config

        elif has_url and server_type in ("http", "sse"):
            # HTTP or SSE server
            server_config = {
                "type": server_type,
                "url": custom.get("url", ""),
            }
            if custom.get("headers"):
                server_config["headers"] = custom["headers"]
            mcp_servers[server_id] = server_config

    # Build system prompt
    base_prompt = (
        f"You are an expert full-stack developer building production-quality software. "
        f"Your working directory is: {project_dir.resolve()}\n"
        f"Your filesystem access is RESTRICTED to this directory only. "
        f"Use relative paths (starting with ./) for all file operations. "
        f"Never use absolute paths or try to access files outside your working directory.\n\n"
        f"You follow existing code patterns, write clean maintainable code, and verify "
        f"your work through thorough testing. You communicate progress through Git commits "
        f"and build-progress.txt updates."
    )

    # Include CLAUDE.md if enabled and present
    if should_use_claude_md():
        claude_md_content = load_claude_md(project_dir)
        if claude_md_content:
            base_prompt = f"{base_prompt}\n\n# Project Instructions (from CLAUDE.md)\n\n{claude_md_content}"
            print("   - CLAUDE.md: included in system prompt")
        else:
            print("   - CLAUDE.md: not found in project root")
    else:
        print("   - CLAUDE.md: disabled by project settings")
    print()

    # Build options dict, conditionally including output_format
    resolved_cwd = str(project_dir.resolve())
    print(f"[SDK] cwd={resolved_cwd} (agent_type={agent_type})")

    options_kwargs: dict[str, Any] = {
        "model": model,
        "system_prompt": base_prompt,
        "allowed_tools": allowed_tools_list,
        "mcp_servers": mcp_servers,
        "hooks": {
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[
                    _create_exec_policy_hook(agent_type, project_dir, spec_dir)
                ]),
            ],
        },
        "max_turns": 1000,
        "cwd": resolved_cwd,
        "settings": str(settings_file.resolve()),
        "env": sdk_env,  # Pass ANTHROPIC_BASE_URL etc. to subprocess
        "max_thinking_tokens": max_thinking_tokens,  # Extended thinking budget
        "max_buffer_size": 10
        * 1024
        * 1024,  # 10MB buffer (default: 1MB) - fixes large tool results
        # Enable file checkpointing to track file read/write state across tool calls
        # This prevents "File has not been read yet" errors in recovery sessions
        "enable_file_checkpointing": True,
    }

    # Optional: Allow CLI path override via environment variable
    # The SDK bundles its own CLI, but users can override if needed
    env_cli_path = os.environ.get("CLAUDE_CLI_PATH")
    if env_cli_path and validate_cli_path(env_cli_path):
        options_kwargs["cli_path"] = env_cli_path
        logger.info(f"Using CLAUDE_CLI_PATH override: {env_cli_path}")

    # Add structured output format if specified
    # See: https://platform.claude.com/docs/en/agent-sdk/structured-outputs
    if output_format:
        options_kwargs["output_format"] = output_format

    # Add subagent definitions if specified
    # See: https://platform.claude.com/docs/en/agent-sdk/subagents
    if agents:
        options_kwargs["agents"] = agents

    return ClaudeSDKClient(options=ClaudeAgentOptions(**options_kwargs))
