"""
Built-in Pipeline Definitions
===============================

Declarative pipeline definitions for common execution flows.
Each pipeline is a DAG of PipelineStages that the PipelineEngine
executes in topological order.

These pipelines wrap the existing flow (build_commands.py) in
a declarative structure. The handlers reference existing functions.

Available pipelines:
    DEFAULT_PIPELINE  — Standard build: plan → code → QA → merge
    DESIGN_PIPELINE   — Design decomposition: plan only
    QA_ONLY_PIPELINE  — QA validation only (resume)

Usage:
    from core.pipelines import DEFAULT_PIPELINE, get_pipeline
    from core.pipeline import PipelineEngine

    pipeline = get_pipeline("default")
    engine = PipelineEngine(pipeline, context)
    await engine.run()
"""

from __future__ import annotations

from .pipeline import PipelineDefinition, PipelineStage, StageType


# =============================================================================
# Handler functions for pipeline stages
# =============================================================================
# These are thin wrappers that adapt existing functions to the
# pipeline context dict interface. They live here to avoid circular imports.
# Each handler receives a context dict and returns a result.


async def _run_planner(ctx: dict) -> bool:
    """Run the planner agent (first_run=True branch of coder.py)."""
    from agent import run_autonomous_agent

    await run_autonomous_agent(
        project_dir=ctx["working_dir"],
        spec_dir=ctx["spec_dir"],
        model=ctx["model"],
        max_iterations=ctx.get("max_iterations"),
        verbose=ctx.get("verbose", False),
        source_spec_dir=ctx.get("source_spec_dir"),
        original_project_dir=ctx.get("original_project_dir"),
    )
    return True


async def _run_qa(ctx: dict) -> bool:
    """Run the QA validation loop.

    Pre-condition: checks should_run_qa() before running.
    If build is not complete, QA is skipped (returns True).
    """
    from qa_loop import run_qa_validation_loop, should_run_qa

    spec_dir = ctx["spec_dir"]
    if not should_run_qa(spec_dir):
        return True  # Nothing to QA — skip

    return await run_qa_validation_loop(
        project_dir=ctx["working_dir"],
        spec_dir=spec_dir,
        model=ctx["model"],
        verbose=ctx.get("verbose", False),
        source_spec_dir=ctx.get("source_spec_dir"),
    )


async def _run_merge(ctx: dict) -> bool:
    """Finalize workspace (merge worktree)."""
    from workspace import finalize_workspace, handle_workspace_choice

    worktree_manager = ctx.get("worktree_manager")
    if worktree_manager:
        qa_approved = ctx.get("stage_qa_result", True)
        choice = finalize_workspace(
            ctx["project_dir"],
            ctx["spec_dir"].name,
            worktree_manager,
            auto_continue=ctx.get("auto_continue", False),
            auto_merge=ctx.get("auto_merge", False) and qa_approved,
        )
        handle_workspace_choice(
            choice, ctx["project_dir"], ctx["spec_dir"].name, worktree_manager
        )
    elif ctx.get("stage_qa_result", True):
        from core.workspace.finalization import mark_plan_done
        mark_plan_done(ctx["project_dir"], ctx["spec_dir"].name)

    return True


# =============================================================================
# Pipeline Definitions
# =============================================================================


DEFAULT_PIPELINE = PipelineDefinition(
    id="default",
    name="Standard Build Pipeline",
    stages=[
        PipelineStage(
            id="build",
            stage_type=StageType.AGENT,
            agent_id="coder",
            handler="core.pipelines:_run_planner",
            depends_on=[],
            description="Run planner + coder agents",
        ),
        PipelineStage(
            id="qa",
            stage_type=StageType.AGENT,
            agent_id="qa_reviewer",
            handler="core.pipelines:_run_qa",
            depends_on=["build"],
            condition="not skip_qa",
            description="QA validation loop (reviewer + fixer)",
        ),
        PipelineStage(
            id="merge",
            stage_type=StageType.HANDLER,
            handler="core.pipelines:_run_merge",
            depends_on=["qa"],
            description="Finalize workspace and merge",
        ),
    ],
)


DESIGN_PIPELINE = PipelineDefinition(
    id="design",
    name="Design Decomposition Pipeline",
    stages=[
        PipelineStage(
            id="decompose",
            stage_type=StageType.AGENT,
            agent_id="planner",
            handler="core.pipelines:_run_planner",
            depends_on=[],
            description="Run design architect to create child specs",
        ),
    ],
)


QA_ONLY_PIPELINE = PipelineDefinition(
    id="qa_only",
    name="QA Validation Only",
    stages=[
        PipelineStage(
            id="qa",
            stage_type=StageType.AGENT,
            agent_id="qa_reviewer",
            handler="core.pipelines:_run_qa",
            depends_on=[],
            description="Run QA validation loop only",
        ),
    ],
)


MCTS_PIPELINE = PipelineDefinition(
    id="mcts",
    name="MCTS Multi-Path Search",
    stages=[
        PipelineStage(
            id="mcts_search",
            stage_type=StageType.HANDLER,
            handler="core.pipelines:_run_mcts",
            depends_on=[],
            description="Run MCTS multi-path search with parallel branch exploration",
        ),
        PipelineStage(
            id="merge_best",
            stage_type=StageType.HANDLER,
            handler="core.pipelines:_run_merge",
            depends_on=["mcts_search"],
            condition="stage_mcts_search_result",
            description="Merge the best MCTS branch to project",
        ),
    ],
)


async def _run_mcts(ctx: dict) -> bool:
    """Run MCTS multi-path search."""
    from mcts import run_mcts_search

    result = await run_mcts_search(
        project_dir=ctx["project_dir"],
        spec_dir=ctx["spec_dir"],
        model=ctx["model"],
        max_iterations=ctx.get("mcts_max_iterations", 10),
        max_branches=ctx.get("mcts_max_branches", 20),
        budget_seconds=ctx.get("mcts_budget_seconds", 3600.0),
    )

    ctx["mcts_result"] = result
    ctx["stage_mcts_search_result"] = result.success
    # For the merge stage, set which spec to merge
    if result.best_spec_id:
        ctx["best_spec_id"] = result.best_spec_id

    return result.success


# =============================================================================
# Pipeline Registry
# =============================================================================

_PIPELINES: dict[str, PipelineDefinition] = {
    "default": DEFAULT_PIPELINE,
    "design": DESIGN_PIPELINE,
    "qa_only": QA_ONLY_PIPELINE,
    "mcts": MCTS_PIPELINE,
}


def get_pipeline(pipeline_id: str) -> PipelineDefinition:
    """Get a pipeline definition by ID.

    Args:
        pipeline_id: Pipeline identifier

    Returns:
        PipelineDefinition

    Raises:
        KeyError: If pipeline not found
    """
    if pipeline_id not in _PIPELINES:
        raise KeyError(
            f"Unknown pipeline: '{pipeline_id}'. "
            f"Available: {sorted(_PIPELINES.keys())}"
        )
    return _PIPELINES[pipeline_id]


def register_pipeline(pipeline: PipelineDefinition) -> None:
    """Register a custom pipeline definition."""
    _PIPELINES[pipeline.id] = pipeline


def list_pipelines() -> list[str]:
    """List all available pipeline IDs."""
    return sorted(_PIPELINES.keys())
