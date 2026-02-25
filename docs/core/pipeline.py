"""
Declarative Pipeline Engine (n8n-Pattern Architecture)
=======================================================

Pipeline stages defined as data, not hardcoded function calls.
Enables reordering, conditional stages, parallel execution.

Each PipelineStage is an n8n node: clear inputs (context dict),
clear outputs (result added to context), well-defined dependencies.

Usage:
    from core.pipeline import PipelineEngine
    from core.pipelines import DEFAULT_PIPELINE

    ctx = {"project_dir": project_dir, "spec_dir": spec_dir, ...}
    engine = PipelineEngine(DEFAULT_PIPELINE, ctx)
    result = await engine.run()

Node pattern:
    Input:  context dict (project_dir, spec_dir, model, ...)
    Output: updated context dict with stage results
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


class StageType(Enum):
    """Type of pipeline stage."""
    AGENT = "agent"              # Run a single agent
    PARALLEL = "parallel"        # Run child stages in parallel
    CONDITIONAL = "conditional"  # Run if condition met
    HANDLER = "handler"          # Run a plain handler function


@dataclass
class PipelineStage:
    """One node in the pipeline graph.

    Attributes:
        id: Unique stage identifier (e.g., "plan", "code", "qa")
        stage_type: How to execute this stage
        agent_id: AgentDefinition.id (for AGENT type, informational)
        depends_on: Stage IDs that must complete first
        condition: Capability check expression (e.g., "not skip_qa")
        children: Sub-stages (for PARALLEL type)
        handler: Python dotted path to async handler function
                 e.g., "qa.loop:run_qa_validation_loop"
        max_retries: Number of retries on failure
        description: Human-readable stage description
    """

    id: str
    stage_type: StageType = StageType.HANDLER
    agent_id: str | None = None
    depends_on: list[str] = field(default_factory=list)
    condition: str | None = None
    children: list[PipelineStage] = field(default_factory=list)
    handler: str = ""
    max_retries: int = 0
    description: str = ""


@dataclass
class PipelineDefinition:
    """Complete pipeline definition — a directed acyclic graph of stages.

    Attributes:
        id: Pipeline identifier (e.g., "default", "design")
        name: Human-readable pipeline name
        stages: Ordered list of stages (topological order preferred)
    """

    id: str
    name: str
    stages: list[PipelineStage] = field(default_factory=list)


@dataclass
class StageResult:
    """Result of executing one pipeline stage.

    Attributes:
        stage_id: Which stage produced this result
        success: Whether the stage completed successfully
        result: Return value from the handler
        duration_seconds: Execution time
        error: Error message if failed
        skipped: Whether the stage was skipped (condition not met)
    """

    stage_id: str
    success: bool = True
    result: Any = None
    duration_seconds: float = 0.0
    error: str | None = None
    skipped: bool = False


# =============================================================================
# Pipeline Engine
# =============================================================================


class PipelineEngine:
    """Executes a PipelineDefinition stage by stage.

    Respects depends_on ordering, evaluates conditions, supports
    parallel execution of child stages.

    Usage:
        engine = PipelineEngine(pipeline, context)
        final_context = await engine.run()
    """

    def __init__(
        self,
        pipeline: PipelineDefinition,
        context: dict[str, Any],
    ) -> None:
        self.pipeline = pipeline
        self.context = dict(context)  # Copy to avoid mutation
        self.results: dict[str, StageResult] = {}
        self._completed: set[str] = set()

    async def run(self) -> dict[str, Any]:
        """Execute all stages respecting depends_on ordering.

        Returns the final context dict with all stage results.
        """
        logger.info(f"Pipeline '{self.pipeline.id}' starting ({len(self.pipeline.stages)} stages)")

        for stage in self._topological_sort():
            # Check dependencies
            unmet = [d for d in stage.depends_on if d not in self._completed]
            if unmet:
                logger.warning(
                    f"Stage '{stage.id}' has unmet dependencies: {unmet}. Skipping."
                )
                self.results[stage.id] = StageResult(
                    stage_id=stage.id, success=False, skipped=True,
                    error=f"Unmet dependencies: {unmet}",
                )
                continue

            # Check failed dependencies (skip if any dependency failed)
            failed_deps = [
                d for d in stage.depends_on
                if d in self.results and not self.results[d].success and not self.results[d].skipped
            ]
            if failed_deps:
                logger.warning(
                    f"Stage '{stage.id}' skipped — dependency failed: {failed_deps}"
                )
                self.results[stage.id] = StageResult(
                    stage_id=stage.id, success=False, skipped=True,
                    error=f"Dependency failed: {failed_deps}",
                )
                continue

            # Evaluate condition
            if stage.condition and not self._evaluate_condition(stage.condition):
                logger.info(f"Stage '{stage.id}' skipped — condition not met: {stage.condition}")
                self.results[stage.id] = StageResult(
                    stage_id=stage.id, success=True, skipped=True,
                )
                self._completed.add(stage.id)
                continue

            # Execute stage
            result = await self._run_stage(stage)
            self.results[stage.id] = result
            self.context[f"stage_{stage.id}_result"] = result.result

            if result.success or result.skipped:
                self._completed.add(stage.id)
            else:
                logger.error(f"Stage '{stage.id}' failed: {result.error}")
                # Don't abort — let dependent stages handle the failure

        logger.info(
            f"Pipeline '{self.pipeline.id}' completed. "
            f"{len(self._completed)}/{len(self.pipeline.stages)} stages succeeded."
        )
        self.context["_pipeline_results"] = self.results
        return self.context

    async def _run_stage(self, stage: PipelineStage) -> StageResult:
        """Execute a single stage with retry support."""
        start = time.monotonic()

        for attempt in range(stage.max_retries + 1):
            try:
                if stage.stage_type == StageType.PARALLEL:
                    result = await self._run_parallel(stage)
                else:
                    result = await self._run_handler(stage)

                duration = time.monotonic() - start
                return StageResult(
                    stage_id=stage.id,
                    success=True,
                    result=result,
                    duration_seconds=duration,
                )
            except Exception as e:
                if attempt < stage.max_retries:
                    logger.warning(
                        f"Stage '{stage.id}' attempt {attempt + 1} failed: {e}. Retrying..."
                    )
                    continue

                duration = time.monotonic() - start
                return StageResult(
                    stage_id=stage.id,
                    success=False,
                    error=str(e),
                    duration_seconds=duration,
                )

        # Should never reach here, but just in case
        return StageResult(stage_id=stage.id, success=False, error="Unknown error")

    async def _run_handler(self, stage: PipelineStage) -> Any:
        """Import and call the handler function with context."""
        if not stage.handler:
            logger.warning(f"Stage '{stage.id}' has no handler, skipping execution")
            return None

        handler = self._resolve_handler(stage.handler)
        result = handler(self.context)

        # Support both sync and async handlers
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _run_parallel(self, stage: PipelineStage) -> list[Any]:
        """Run child stages in parallel."""
        if not stage.children:
            return []

        tasks = []
        submitted_children: list[PipelineStage] = []
        for child in stage.children:
            # Evaluate child condition
            if child.condition and not self._evaluate_condition(child.condition):
                self.results[child.id] = StageResult(
                    stage_id=child.id, success=True, skipped=True,
                )
                self._completed.add(child.id)
                continue
            tasks.append(self._run_stage(child))
            submitted_children.append(child)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results — index aligns with submitted_children, not stage.children
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                child = submitted_children[i]
                self.results[child.id] = StageResult(
                    stage_id=child.id, success=False, error=str(result),
                )
            elif isinstance(result, StageResult):
                self.results[result.stage_id] = result
                if result.success:
                    self._completed.add(result.stage_id)
                processed.append(result.result)

        return processed

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a condition string against the current context.

        Supports simple expressions:
        - "skip_qa" → context.get("skip_qa", False)
        - "not skip_qa" → not context.get("skip_qa", False)
        - "qa_approved" → context.get("qa_approved", False)
        - "qa_approved and auto_merge" → both truthy
        """
        condition = condition.strip()

        # Handle "not X"
        if condition.startswith("not "):
            key = condition[4:].strip()
            return not self.context.get(key, False)

        # Handle "X and Y"
        if " and " in condition:
            parts = [p.strip() for p in condition.split(" and ")]
            return all(self.context.get(p, False) for p in parts)

        # Handle "X or Y"
        if " or " in condition:
            parts = [p.strip() for p in condition.split(" or ")]
            return any(self.context.get(p, False) for p in parts)

        # Simple key lookup
        return bool(self.context.get(condition, False))

    def _resolve_handler(self, handler_path: str) -> Callable:
        """Resolve a dotted path like 'qa.loop:run_qa_validation_loop' to a callable.

        Format: "module.path:function_name"
        """
        if ":" not in handler_path:
            raise ValueError(
                f"Handler '{handler_path}' must use format 'module.path:function_name'"
            )

        module_path, func_name = handler_path.rsplit(":", 1)
        module = importlib.import_module(module_path)
        handler = getattr(module, func_name, None)
        if handler is None:
            raise AttributeError(
                f"Module '{module_path}' has no function '{func_name}'"
            )
        return handler

    def _topological_sort(self) -> list[PipelineStage]:
        """Sort stages topologically based on depends_on.

        Falls back to original order if no dependencies.
        """
        stages_by_id = {s.id: s for s in self.pipeline.stages}
        visited: set[str] = set()
        result: list[PipelineStage] = []

        def visit(stage_id: str) -> None:
            if stage_id in visited:
                return
            visited.add(stage_id)
            stage = stages_by_id.get(stage_id)
            if stage is None:
                return
            for dep in stage.depends_on:
                if dep in stages_by_id:
                    visit(dep)
            result.append(stage)

        for stage in self.pipeline.stages:
            visit(stage.id)

        return result
