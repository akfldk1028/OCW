"""
Unified Agent Registry (n8n-Pattern Architecture)
===================================================

Single source of truth for all agent definitions. Each agent is one
self-contained node with clear inputs/outputs — exactly like an n8n node.

Previously, agent configuration was scattered across 4 registries:
1. agents/tools_pkg/models.py → AGENT_CONFIGS (tools, mcp_servers, thinking)
2. core/exec_policy.py → AGENT_EXEC_POLICIES (SecurityLevel per agent)
3. services/task_daemon/executor.py → AGENT_REGISTRY (script, prompt, execution_mode)
4. core/tool_policy.py → ToolProfile presets (MINIMAL, READONLY, CODING, QA, FULL)

Now: one AgentDefinition = one agent. One registry holds them all.

Usage:
    from core.agent_registry import AgentRegistry

    reg = AgentRegistry.instance()
    coder = reg.get("coder")
    assert coder.security_level == "full"
    assert coder.tool_profile == "CODING"

    # List by category
    qa_agents = reg.list_by_category("qa")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Constants (canonical definitions — imported by shims)
# =============================================================================

BASE_READ_TOOLS = ["Read", "Glob", "Grep"]
BASE_WRITE_TOOLS = ["Write", "Edit", "Bash"]
WEB_TOOLS = ["WebFetch", "WebSearch"]

# Auto-Claude MCP tool names
TOOL_UPDATE_SUBTASK_STATUS = "mcp__auto-claude__update_subtask_status"
TOOL_GET_BUILD_PROGRESS = "mcp__auto-claude__get_build_progress"
TOOL_RECORD_DISCOVERY = "mcp__auto-claude__record_discovery"
TOOL_RECORD_GOTCHA = "mcp__auto-claude__record_gotcha"
TOOL_GET_SESSION_CONTEXT = "mcp__auto-claude__get_session_context"
TOOL_UPDATE_QA_STATUS = "mcp__auto-claude__update_qa_status"
TOOL_CREATE_CHILD_SPEC = "mcp__auto-claude__create_child_spec"
TOOL_CREATE_BATCH_CHILD_SPECS = "mcp__auto-claude__create_batch_child_specs"


# =============================================================================
# AgentDefinition Dataclass
# =============================================================================


@dataclass
class AgentDefinition:
    """Single source of truth for one agent. n8n node = 1 AgentDefinition.

    All fields that were previously scattered across AGENT_CONFIGS,
    AGENT_EXEC_POLICIES, AGENT_REGISTRY, and ToolProfile are unified here.

    Attributes:
        id: Unique agent identifier (e.g., "coder", "qa_reviewer")
        description: Human-readable purpose
        category: Functional category for grouping

        # Tools (from AGENT_CONFIGS)
        tools: Claude SDK tools the agent can use
        mcp_servers: Required MCP servers
        mcp_servers_optional: Conditional MCP servers (e.g., "linear")
        auto_claude_tools: Custom auto-claude MCP tools
        thinking_default: Default thinking level

        # Security (from AGENT_EXEC_POLICIES)
        security_level: Bash execution policy level
        extra_allow: Additional allowed commands for READONLY
        extra_deny: Commands denied even if level allows

        # Execution (from AGENT_REGISTRY in executor.py)
        script: Custom script path (None = use run.py)
        use_claude_cli: True = Claude CLI mode, False = run.py
        prompt_template: For CLI mode prompt generation
        system_prompt: Prompt file name (e.g., "coder.md")
        execution_mode: "plan" | None
        extra_args: CLI extra args

        # Tool Profile (from tool_policy.py)
        tool_profile: Standard profile name or None
    """

    id: str
    description: str = ""
    category: str = "utility"  # spec | build | qa | design | verification | utility | analysis

    # ── Tools ──
    tools: list[str] = field(default_factory=list)
    mcp_servers: list[str] = field(default_factory=list)
    mcp_servers_optional: list[str] = field(default_factory=list)
    auto_claude_tools: list[str] = field(default_factory=list)
    thinking_default: str = "medium"

    # ── Security ──
    security_level: str = "allowlist"  # deny | readonly | allowlist | full
    extra_allow: list[str] = field(default_factory=list)
    extra_deny: list[str] = field(default_factory=list)

    # ── Execution ──
    script: str | None = None
    use_claude_cli: bool = False
    prompt_template: str | None = None
    system_prompt: str | None = None
    execution_mode: str | None = None
    extra_args: list[str] = field(default_factory=list)

    # ── Tool Profile ──
    tool_profile: str | None = None  # MINIMAL | READONLY | CODING | QA | FULL

    # ── Metadata ──
    is_custom: bool = False
    custom_prompt_file: str | None = None

    def to_agent_config_dict(self) -> dict:
        """Convert to AGENT_CONFIGS-compatible dict (backward compat shim)."""
        d = {
            "tools": list(self.tools),
            "mcp_servers": list(self.mcp_servers),
            "auto_claude_tools": list(self.auto_claude_tools),
            "thinking_default": self.thinking_default,
        }
        if self.mcp_servers_optional:
            d["mcp_servers_optional"] = list(self.mcp_servers_optional)
        if self.is_custom:
            d["_custom"] = True
            d["_description"] = self.description
            if self.custom_prompt_file:
                d["_prompt_file"] = self.custom_prompt_file
            d["_use_claude_cli"] = self.use_claude_cli
            d["_script"] = self.script
            d["_extra_args"] = list(self.extra_args)
            d["_prompt_template"] = self.prompt_template
            d["_execution_mode"] = self.execution_mode
        return d

    def to_exec_policy_tuple(self) -> tuple[str, list[str], list[str]]:
        """Return (security_level, extra_allow, extra_deny) for exec_policy shim."""
        return (self.security_level, list(self.extra_allow), list(self.extra_deny))

    def to_executor_config_dict(self) -> dict:
        """Convert to executor.py AgentConfig-compatible fields."""
        return {
            "script": self.script,
            "use_claude_cli": self.use_claude_cli,
            "prompt_template": self.prompt_template,
            "system_prompt": self.system_prompt,
            "execution_mode": self.execution_mode,
            "extra_args": list(self.extra_args),
            "mcp_servers": list(self.mcp_servers),
        }


# =============================================================================
# AgentRegistry Singleton
# =============================================================================


class AgentRegistry:
    """Singleton registry. All agent lookups go through here.

    Usage:
        reg = AgentRegistry.instance()
        coder = reg.get("coder")
        qa_agents = reg.list_by_category("qa")
    """

    _instance: ClassVar[AgentRegistry | None] = None
    _agents: dict[str, AgentDefinition]

    def __init__(self) -> None:
        self._agents = {}
        self._load_builtins()
        self._load_custom_agents()

    @classmethod
    def instance(cls) -> AgentRegistry:
        """Get or create the singleton registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def _reset(cls) -> None:
        """Reset singleton (for testing only)."""
        cls._instance = None

    def get(self, agent_id: str) -> AgentDefinition | None:
        """Look up an agent by ID. Returns None if not found."""
        return self._agents.get(agent_id)

    def get_or_default(self, agent_id: str) -> AgentDefinition:
        """Look up an agent by ID, returning default if not found."""
        agent = self._agents.get(agent_id)
        if agent is not None:
            return agent
        default = self._agents.get("default")
        if default is not None:
            return default
        # Absolute fallback
        return AgentDefinition(id=agent_id)

    def register(self, definition: AgentDefinition) -> None:
        """Register or override an agent definition."""
        self._agents[definition.id] = definition

    def list_by_category(self, category: str) -> list[AgentDefinition]:
        """List all agents in a category."""
        return [a for a in self._agents.values() if a.category == category]

    def all_ids(self) -> list[str]:
        """Return all registered agent IDs."""
        return list(self._agents.keys())

    def all_agents(self) -> dict[str, AgentDefinition]:
        """Return all agents as a dict."""
        return dict(self._agents)

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def _load_builtins(self) -> None:
        """Load BUILTIN_AGENTS dict (defined at bottom of this file)."""
        for agent_id, defn in BUILTIN_AGENTS.items():
            self._agents[agent_id] = defn

    def _load_custom_agents(self) -> None:
        """Load from custom_agents/config.json — backward compatible."""
        custom_agents_dir = Path(__file__).parent.parent / "custom_agents"
        config_path = custom_agents_dir / "config.json"

        if not config_path.exists():
            return

        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load custom agents config: {e}")
            return

        agents_data = config.get("agents", {})
        for agent_type, agent_config in agents_data.items():
            # Check for conflicts with built-in agents
            if agent_type in self._agents and not self._agents[agent_type].is_custom:
                logger.warning(
                    f"Custom agent '{agent_type}' conflicts with built-in agent. Skipped."
                )
                continue

            # Validate prompt file
            prompt_file = agent_config.get("prompt_file", "")
            prompt_path = custom_agents_dir / "prompts" / prompt_file
            if not prompt_path.exists():
                logger.warning(
                    f"Custom agent '{agent_type}' prompt file not found: {prompt_path}"
                )
                continue

            defn = AgentDefinition(
                id=agent_type,
                description=agent_config.get("description", ""),
                category="custom",
                tools=agent_config.get("tools", BASE_READ_TOOLS + BASE_WRITE_TOOLS),
                mcp_servers=agent_config.get("mcp_servers", []),
                auto_claude_tools=agent_config.get("auto_claude_tools", []),
                thinking_default=agent_config.get("thinking_default", "medium"),
                security_level="allowlist",  # Custom agents get default security
                script=agent_config.get("script"),
                use_claude_cli=agent_config.get("use_claude_cli", False),
                prompt_template=agent_config.get("prompt_template"),
                execution_mode=agent_config.get("execution_mode"),
                extra_args=agent_config.get("extra_args", []),
                is_custom=True,
                custom_prompt_file=str(prompt_path),
            )
            self._agents[agent_type] = defn
            logger.info(f"Registered custom agent: {agent_type}")

    def load_project_agents(self, project_dir: str | Path) -> int:
        """Load project-specific agents from {project_dir}/.auto-claude/agents/.

        Project agents override global custom_agents (same ID = project wins).
        Also loads project-level .env if present.

        Args:
            project_dir: Path to the project root (e.g., D:/Data/MB_N2N/MB_N2N)

        Returns:
            Number of agents loaded from the project.
        """
        project_path = Path(project_dir)
        agents_dir = project_path / ".auto-claude" / "agents"
        config_path = agents_dir / "config.json"

        if not config_path.exists():
            logger.debug(f"No project agents at {config_path}")
            return 0

        # Load project-level .env (override global)
        project_env = project_path / ".auto-claude" / ".env"
        if project_env.exists():
            try:
                self._load_env_file(project_env)
                logger.info(f"Loaded project .env: {project_env}")
            except Exception as e:
                logger.warning(f"Failed to load project .env: {e}")

        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load project agents config: {e}")
            return 0

        count = 0
        agents_data = config.get("agents", {})
        for agent_type, agent_config in agents_data.items():
            # Project agents can override global custom agents
            if agent_type in self._agents and not self._agents[agent_type].is_custom:
                logger.warning(
                    f"Project agent '{agent_type}' conflicts with built-in agent. Skipped."
                )
                continue

            prompt_file = agent_config.get("prompt_file", "")
            prompt_path = agents_dir / "prompts" / prompt_file
            if not prompt_path.exists():
                logger.warning(
                    f"Project agent '{agent_type}' prompt not found: {prompt_path}"
                )
                continue

            defn = AgentDefinition(
                id=agent_type,
                description=agent_config.get("description", ""),
                category="project",
                tools=agent_config.get("tools", BASE_READ_TOOLS + BASE_WRITE_TOOLS),
                mcp_servers=agent_config.get("mcp_servers", []),
                auto_claude_tools=agent_config.get("auto_claude_tools", []),
                thinking_default=agent_config.get("thinking_default", "medium"),
                security_level="allowlist",
                script=agent_config.get("script"),
                use_claude_cli=agent_config.get("use_claude_cli", False),
                prompt_template=agent_config.get("prompt_template"),
                execution_mode=agent_config.get("execution_mode"),
                extra_args=agent_config.get("extra_args", []),
                is_custom=True,
                custom_prompt_file=str(prompt_path),
            )
            self._agents[agent_type] = defn
            count += 1
            logger.info(f"Registered project agent: {agent_type} (from {project_path.name})")

        return count

    @staticmethod
    def _load_env_file(env_path: Path) -> None:
        """Load a .env file into os.environ (simple key=value, skip comments)."""
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if value and not value.startswith("#"):
                        os.environ[key] = value


# =============================================================================
# BUILTIN_AGENTS — All ~30 built-in agents in one place
# =============================================================================

BUILTIN_AGENTS: dict[str, AgentDefinition] = {
    # ═══════════════════════════════════════════════════════════════════════
    # SPEC CREATION AGENTS
    # ═══════════════════════════════════════════════════════════════════════
    "spec_gatherer": AgentDefinition(
        id="spec_gatherer",
        description="Gathers project context for spec creation",
        category="spec",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="readonly",
        tool_profile="MINIMAL",
    ),
    "spec_researcher": AgentDefinition(
        id="spec_researcher",
        description="Researches documentation for spec creation",
        category="spec",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=["context7"],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="readonly",
        tool_profile="READONLY",
    ),
    "spec_writer": AgentDefinition(
        id="spec_writer",
        description="Writes spec.md from gathered context",
        category="spec",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS,
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="high",
        security_level="readonly",
        tool_profile="CODING",
    ),
    "spec_critic": AgentDefinition(
        id="spec_critic",
        description="Self-critiques spec quality",
        category="spec",
        tools=BASE_READ_TOOLS,
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="ultrathink",
        security_level="deny",
        tool_profile="MINIMAL",
    ),
    "spec_discovery": AgentDefinition(
        id="spec_discovery",
        description="Discovers project structure for spec",
        category="spec",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="readonly",
        tool_profile="MINIMAL",
    ),
    "spec_context": AgentDefinition(
        id="spec_context",
        description="Builds context for spec creation",
        category="spec",
        tools=BASE_READ_TOOLS,
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="readonly",
        tool_profile="MINIMAL",
    ),
    "spec_validation": AgentDefinition(
        id="spec_validation",
        description="Validates spec completeness",
        category="spec",
        tools=BASE_READ_TOOLS,
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="high",
        security_level="readonly",
        tool_profile="MINIMAL",
    ),
    "spec_compaction": AgentDefinition(
        id="spec_compaction",
        description="Compacts and optimizes spec content",
        category="spec",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS,
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="readonly",
        tool_profile="CODING",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # BUILD AGENTS
    # ═══════════════════════════════════════════════════════════════════════
    "planner": AgentDefinition(
        id="planner",
        description="Creates implementation plans with subtasks",
        category="build",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7", "graphiti", "auto-claude"],
        mcp_servers_optional=["linear"],
        auto_claude_tools=[
            TOOL_GET_BUILD_PROGRESS,
            TOOL_GET_SESSION_CONTEXT,
            TOOL_RECORD_DISCOVERY,
            TOOL_CREATE_CHILD_SPEC,
            TOOL_CREATE_BATCH_CHILD_SPECS,
        ],
        thinking_default="high",
        security_level="full",
        tool_profile="CODING",
    ),
    "coder": AgentDefinition(
        id="coder",
        description="Implements subtasks from the plan",
        category="build",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7", "graphiti", "auto-claude"],
        mcp_servers_optional=["linear"],
        auto_claude_tools=[
            TOOL_UPDATE_SUBTASK_STATUS,
            TOOL_GET_BUILD_PROGRESS,
            TOOL_RECORD_DISCOVERY,
            TOOL_RECORD_GOTCHA,
            TOOL_GET_SESSION_CONTEXT,
        ],
        thinking_default="none",
        security_level="full",
        tool_profile="CODING",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # QA AGENTS
    # ═══════════════════════════════════════════════════════════════════════
    "qa_reviewer": AgentDefinition(
        id="qa_reviewer",
        description="Validates implementation against acceptance criteria",
        category="qa",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7", "graphiti", "auto-claude", "browser"],
        mcp_servers_optional=["linear"],
        auto_claude_tools=[
            TOOL_GET_BUILD_PROGRESS,
            TOOL_UPDATE_QA_STATUS,
            TOOL_GET_SESSION_CONTEXT,
        ],
        thinking_default="high",
        security_level="full",
        tool_profile="QA",
    ),
    "qa_fixer": AgentDefinition(
        id="qa_fixer",
        description="Fixes issues found during QA review",
        category="qa",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7", "graphiti", "auto-claude", "browser"],
        mcp_servers_optional=["linear"],
        auto_claude_tools=[
            TOOL_UPDATE_SUBTASK_STATUS,
            TOOL_GET_BUILD_PROGRESS,
            TOOL_UPDATE_QA_STATUS,
            TOOL_RECORD_GOTCHA,
        ],
        thinking_default="medium",
        security_level="full",
        tool_profile="QA",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # VERIFICATION & ERROR-CHECK AGENTS
    # ═══════════════════════════════════════════════════════════════════════
    "verify": AgentDefinition(
        id="verify",
        description="Verifies implementation (tests/builds/runtime)",
        category="verification",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7", "auto-claude", "browser"],
        auto_claude_tools=[TOOL_GET_BUILD_PROGRESS, TOOL_GET_SESSION_CONTEXT],
        thinking_default="high",
        security_level="full",
        system_prompt="verify_agent.md",
        tool_profile="QA",
    ),
    "error_check": AgentDefinition(
        id="error_check",
        description="Fixes errors in implementation",
        category="verification",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7", "graphiti", "auto-claude"],
        auto_claude_tools=[
            TOOL_UPDATE_SUBTASK_STATUS,
            TOOL_GET_BUILD_PROGRESS,
            TOOL_RECORD_GOTCHA,
        ],
        thinking_default="medium",
        security_level="full",
        system_prompt="error_check_agent.md",
        tool_profile="CODING",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # DESIGN AGENTS (executor.py AGENT_REGISTRY entries)
    # ═══════════════════════════════════════════════════════════════════════
    "design": AgentDefinition(
        id="design",
        description="Design architect — decomposes into child tasks",
        category="design",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7", "graphiti", "auto-claude"],
        auto_claude_tools=[
            TOOL_GET_BUILD_PROGRESS,
            TOOL_GET_SESSION_CONTEXT,
            TOOL_RECORD_DISCOVERY,
            TOOL_CREATE_CHILD_SPEC,
            TOOL_CREATE_BATCH_CHILD_SPECS,
        ],
        thinking_default="high",
        security_level="full",
        use_claude_cli=False,
        system_prompt="design_architect.md",
        prompt_template=(
            "You are a Design Architect Agent.\n\n"
            "Task: {task}\n\n"
            "Analyze the project structure and create implementation tasks using "
            "the create_batch_child_specs tool.\n\n"
            "Spec Content:\n{spec_content}"
        ),
        tool_profile="CODING",
    ),
    "architecture": AgentDefinition(
        id="architecture",
        description="Architecture analysis agent",
        category="design",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7", "graphiti", "auto-claude"],
        auto_claude_tools=[
            TOOL_GET_BUILD_PROGRESS,
            TOOL_GET_SESSION_CONTEXT,
            TOOL_RECORD_DISCOVERY,
            TOOL_CREATE_CHILD_SPEC,
            TOOL_CREATE_BATCH_CHILD_SPECS,
        ],
        thinking_default="high",
        security_level="full",
        use_claude_cli=False,
        prompt_template=(
            "You are an Architecture Analyst Agent.\n\n"
            "Task: {task}\n\n"
            "Analyze the codebase architecture and provide recommendations.\n\n"
            "Spec Content:\n{spec_content}"
        ),
        tool_profile="CODING",
    ),
    "research": AgentDefinition(
        id="research",
        description="Research agent for codebase investigation",
        category="design",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=["context7"],
        auto_claude_tools=[],
        thinking_default="high",
        security_level="readonly",
        use_claude_cli=True,
        execution_mode="plan",
        prompt_template=(
            "You are a Research Agent.\n\n"
            "Task: {task}\n\n"
            "Investigate the codebase and gather information.\n\n"
            "Spec Content:\n{spec_content}"
        ),
        tool_profile="READONLY",
    ),
    "review": AgentDefinition(
        id="review",
        description="Code review agent",
        category="design",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=["context7"],
        auto_claude_tools=[],
        thinking_default="high",
        security_level="readonly",
        use_claude_cli=True,
        execution_mode="plan",
        prompt_template=(
            "You are a Code Review Agent.\n\n"
            "Task: {task}\n\n"
            "Review the code and provide feedback.\n\n"
            "Spec Content:\n{spec_content}"
        ),
        tool_profile="READONLY",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # IMPLEMENTATION AGENTS (executor.py)
    # ═══════════════════════════════════════════════════════════════════════
    "impl": AgentDefinition(
        id="impl",
        description="Default implementation agent",
        category="build",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7", "graphiti", "auto-claude"],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="full",
        tool_profile="CODING",
    ),
    "frontend": AgentDefinition(
        id="frontend",
        description="Frontend implementation agent",
        category="build",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7", "graphiti", "auto-claude", "browser"],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="full",
        tool_profile="CODING",
    ),
    "backend": AgentDefinition(
        id="backend",
        description="Backend implementation agent",
        category="build",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7", "graphiti", "auto-claude"],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="full",
        tool_profile="CODING",
    ),
    "database": AgentDefinition(
        id="database",
        description="Database implementation agent",
        category="build",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7", "graphiti", "auto-claude"],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="full",
        tool_profile="CODING",
    ),
    "api": AgentDefinition(
        id="api",
        description="API implementation agent",
        category="build",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7", "graphiti", "auto-claude"],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="full",
        tool_profile="CODING",
    ),
    "test": AgentDefinition(
        id="test",
        description="Test implementation agent",
        category="build",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7", "graphiti", "auto-claude"],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="full",
        tool_profile="CODING",
    ),
    "integration": AgentDefinition(
        id="integration",
        description="Integration implementation agent",
        category="build",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7", "graphiti", "auto-claude"],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="full",
        tool_profile="CODING",
    ),
    "docs": AgentDefinition(
        id="docs",
        description="Documentation agent",
        category="utility",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7"],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="full",
        tool_profile="CODING",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # UTILITY AGENTS
    # ═══════════════════════════════════════════════════════════════════════
    "insights": AgentDefinition(
        id="insights",
        description="Extracts codebase insights",
        category="utility",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="none",
        security_level="readonly",
        tool_profile="MINIMAL",
    ),
    "merge_resolver": AgentDefinition(
        id="merge_resolver",
        description="Resolves merge conflicts",
        category="utility",
        tools=[],
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="low",
        security_level="deny",
        tool_profile="MINIMAL",
    ),
    "commit_message": AgentDefinition(
        id="commit_message",
        description="Generates commit messages",
        category="utility",
        tools=[],
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="low",
        security_level="deny",
        tool_profile="MINIMAL",
    ),
    "pr_template_filler": AgentDefinition(
        id="pr_template_filler",
        description="Fills PR templates",
        category="utility",
        tools=BASE_READ_TOOLS,
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="low",
        security_level="deny",
        tool_profile="MINIMAL",
    ),
    "pr_reviewer": AgentDefinition(
        id="pr_reviewer",
        description="Reviews pull requests",
        category="utility",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=["context7"],
        auto_claude_tools=[],
        thinking_default="high",
        security_level="readonly",
        tool_profile="READONLY",
    ),
    "pr_orchestrator_parallel": AgentDefinition(
        id="pr_orchestrator_parallel",
        description="Orchestrates parallel PR reviews",
        category="utility",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=["context7"],
        auto_claude_tools=[],
        thinking_default="high",
        security_level="readonly",
        tool_profile="READONLY",
    ),
    "pr_followup_parallel": AgentDefinition(
        id="pr_followup_parallel",
        description="Parallel followup PR reviewer",
        category="utility",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=["context7"],
        auto_claude_tools=[],
        thinking_default="high",
        security_level="readonly",
        tool_profile="READONLY",
    ),
    "pr_finding_validator": AgentDefinition(
        id="pr_finding_validator",
        description="Validates PR findings against actual code",
        category="utility",
        tools=BASE_READ_TOOLS,
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="readonly",
        tool_profile="MINIMAL",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # BATCH AGENTS
    # ═══════════════════════════════════════════════════════════════════════
    "batch_planning": AgentDefinition(
        id="batch_planning",
        description="Batch planning agent",
        category="analysis",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=["context7"],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="readonly",
        tool_profile="READONLY",
    ),
    "batch_execution": AgentDefinition(
        id="batch_execution",
        description="Batch execution agent",
        category="build",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7", "graphiti", "auto-claude"],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="full",
        tool_profile="CODING",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # PR AGENTS (non-parallel variants)
    # ═══════════════════════════════════════════════════════════════════════
    "pr_orchestrator": AgentDefinition(
        id="pr_orchestrator",
        description="Orchestrates PR reviews",
        category="utility",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=["context7"],
        auto_claude_tools=[],
        thinking_default="high",
        security_level="readonly",
        tool_profile="READONLY",
    ),
    "pr_followup": AgentDefinition(
        id="pr_followup",
        description="Followup PR reviewer",
        category="utility",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=["context7"],
        auto_claude_tools=[],
        thinking_default="high",
        security_level="readonly",
        tool_profile="READONLY",
    ),
    "pr_finding": AgentDefinition(
        id="pr_finding",
        description="Validates PR findings",
        category="utility",
        tools=BASE_READ_TOOLS,
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="readonly",
        tool_profile="MINIMAL",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # ANALYSIS AGENTS
    # ═══════════════════════════════════════════════════════════════════════
    "analysis": AgentDefinition(
        id="analysis",
        description="Project analysis agent",
        category="analysis",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=["context7"],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="readonly",
        tool_profile="READONLY",
    ),
    "batch_analysis": AgentDefinition(
        id="batch_analysis",
        description="Batch analysis agent",
        category="analysis",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="low",
        security_level="readonly",
        tool_profile="MINIMAL",
    ),
    "batch_validation": AgentDefinition(
        id="batch_validation",
        description="Batch validation agent",
        category="analysis",
        tools=BASE_READ_TOOLS,
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="low",
        security_level="readonly",
        tool_profile="MINIMAL",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # ROADMAP & IDEATION AGENTS
    # ═══════════════════════════════════════════════════════════════════════
    "roadmap_discovery": AgentDefinition(
        id="roadmap_discovery",
        description="Discovers roadmap opportunities",
        category="analysis",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=["context7"],
        auto_claude_tools=[],
        thinking_default="high",
        security_level="readonly",
        tool_profile="READONLY",
    ),
    "competitor_analysis": AgentDefinition(
        id="competitor_analysis",
        description="Analyzes competitors",
        category="analysis",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=["context7"],
        auto_claude_tools=[],
        thinking_default="high",
        security_level="readonly",
        tool_profile="READONLY",
    ),
    "ideation": AgentDefinition(
        id="ideation",
        description="Generates improvement ideas",
        category="analysis",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="high",
        security_level="readonly",
        tool_profile="READONLY",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # MCTS (Multi-Path Search) AGENTS
    # ═══════════════════════════════════════════════════════════════════════
    "mcts_idea_generator": AgentDefinition(
        id="mcts_idea_generator",
        description="Generates N diverse solution approaches for a given task",
        category="design",
        tools=BASE_READ_TOOLS + WEB_TOOLS,
        mcp_servers=["context7"],
        auto_claude_tools=[],
        thinking_default="high",
        security_level="readonly",
        system_prompt="mcts_idea_generator.md",
        tool_profile="READONLY",
    ),
    "mcts_improver": AgentDefinition(
        id="mcts_improver",
        description="Proposes targeted improvements to an existing solution based on lessons",
        category="design",
        tools=BASE_READ_TOOLS,
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="high",
        security_level="readonly",
        system_prompt="mcts_improver.md",
        tool_profile="READONLY",
    ),
    "mcts_debugger": AgentDefinition(
        id="mcts_debugger",
        description="Analyzes failed branch root cause and proposes fix direction",
        category="design",
        tools=BASE_READ_TOOLS,
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="readonly",
        system_prompt="mcts_debugger.md",
        tool_profile="READONLY",
    ),
    "mcts_lesson_extractor": AgentDefinition(
        id="mcts_lesson_extractor",
        description="Compares completed branches and extracts structured lessons",
        category="analysis",
        tools=BASE_READ_TOOLS,
        mcp_servers=[],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="readonly",
        system_prompt="mcts_lesson_extractor.md",
        tool_profile="READONLY",
    ),

    # ═══════════════════════════════════════════════════════════════════════
    # DEFAULT FALLBACK
    # ═══════════════════════════════════════════════════════════════════════
    "default": AgentDefinition(
        id="default",
        description="Default fallback agent",
        category="utility",
        tools=BASE_READ_TOOLS + BASE_WRITE_TOOLS + WEB_TOOLS,
        mcp_servers=["context7", "graphiti", "auto-claude"],
        auto_claude_tools=[],
        thinking_default="medium",
        security_level="allowlist",
        tool_profile="CODING",
    ),
}
