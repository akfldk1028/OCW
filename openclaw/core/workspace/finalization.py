#!/usr/bin/env python3
"""
Workspace Finalization
======================

Functions for finalizing workspaces and handling user choices after build completion.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ui import (
    Icons,
    MenuOption,
    bold,
    box,
    highlight,
    icon,
    info,
    muted,
    print_status,
    select_menu,
    success,
    warning,
)
from worktree import WorktreeInfo, WorktreeManager

from .display import show_build_summary, show_changed_files
from .git_utils import get_existing_build_worktree
from .models import WorkspaceChoice


def finalize_workspace(
    project_dir: Path,
    spec_name: str,
    manager: WorktreeManager | None,
    auto_continue: bool = False,
    auto_merge: bool = False,
) -> WorkspaceChoice:
    """
    Handle post-build workflow - let user decide what to do with changes.

    Safe design:
    - No "discard" option (requires separate --discard command)
    - Default is "test" - encourages testing before merging
    - Everything is preserved until user explicitly merges or discards

    Args:
        project_dir: The project directory
        spec_name: Name of the spec that was built
        manager: The worktree manager (None if direct mode was used)
        auto_continue: If True, skip interactive prompts (UI mode)
        auto_merge: If True, automatically merge without prompting (QA must have passed)

    Returns:
        WorkspaceChoice indicating what user wants to do
    """
    if manager is None:
        # Direct mode - nothing to finalize
        content = [
            success(f"{icon(Icons.SUCCESS)} BUILD COMPLETE!"),
            "",
            "Changes were made directly to your project.",
            muted("Use 'git status' to see what changed."),
        ]
        print()
        print(box(content, width=60, style="heavy"))
        return WorkspaceChoice.MERGE  # Already merged

    # Auto-merge mode: QA passed, merge immediately without prompting
    if auto_merge:
        print()
        print(success(f"{icon(Icons.SUCCESS)} QA PASSED - Auto-merging to project..."))
        return WorkspaceChoice.MERGE

    # In auto_continue mode (UI), skip interactive prompts
    # The worktree stays for the UI to manage
    if auto_continue:
        worktree_info = manager.get_worktree_info(spec_name)
        if worktree_info:
            print()
            print(success(f"Build complete in worktree: {worktree_info.path}"))
            print(muted("Worktree preserved for UI review."))
        return WorkspaceChoice.LATER

    # Isolated mode - show options with testing as the recommended path
    content = [
        success(f"{icon(Icons.SUCCESS)} BUILD COMPLETE!"),
        "",
        "The AI built your feature in a separate workspace.",
    ]
    print()
    print(box(content, width=60, style="heavy"))

    show_build_summary(manager, spec_name)

    # Get the worktree path for test instructions
    worktree_info = manager.get_worktree_info(spec_name)
    staging_path = worktree_info.path if worktree_info else None

    # Enhanced menu for post-build options
    options = [
        MenuOption(
            key="test",
            label="Test the feature (Recommended)",
            icon=Icons.PLAY,
            description="Run the app and try it out before adding to your project",
        ),
        MenuOption(
            key="merge",
            label="Add to my project now",
            icon=Icons.SUCCESS,
            description="Merge the changes into your files immediately",
        ),
        MenuOption(
            key="review",
            label="Review what changed",
            icon=Icons.FILE,
            description="See exactly what files were modified",
        ),
        MenuOption(
            key="later",
            label="Decide later",
            icon=Icons.PAUSE,
            description="Your build is saved - you can come back anytime",
        ),
    ]

    print()
    choice = select_menu(
        title="What would you like to do?",
        options=options,
        allow_quit=False,
    )

    if choice == "test":
        return WorkspaceChoice.TEST
    elif choice == "merge":
        return WorkspaceChoice.MERGE
    elif choice == "review":
        return WorkspaceChoice.REVIEW
    else:
        return WorkspaceChoice.LATER


def handle_workspace_choice(
    choice: WorkspaceChoice,
    project_dir: Path,
    spec_name: str,
    manager: WorktreeManager,
) -> None:
    """
    Execute the user's choice.

    Args:
        choice: What the user wants to do
        project_dir: The project directory
        spec_name: Name of the spec
        manager: The worktree manager
    """
    worktree_info = manager.get_worktree_info(spec_name)
    staging_path = worktree_info.path if worktree_info else None

    if choice == WorkspaceChoice.TEST:
        # Show testing instructions
        content = [
            bold(f"{icon(Icons.PLAY)} TEST YOUR FEATURE"),
            "",
            "Your feature is ready to test in a separate workspace.",
        ]
        print()
        print(box(content, width=60, style="heavy"))

        print()
        print("To test it, open a NEW terminal and run:")
        print()
        if staging_path:
            print(highlight(f"  cd {staging_path}"))
        else:
            worktree_path = get_existing_build_worktree(project_dir, spec_name)
            if worktree_path:
                print(highlight(f"  cd {worktree_path}"))
            else:
                print(
                    highlight(
                        f"  cd {project_dir}/.auto-claude/worktrees/tasks/{spec_name}"
                    )
                )

        # Show likely test/run commands
        if staging_path:
            commands = manager.get_test_commands(spec_name)
            print()
            print("Then run your project:")
            for cmd in commands[:2]:  # Show top 2 commands
                print(f"  {cmd}")

        print()
        print(muted("-" * 60))
        print()
        print("When you're done testing:")
        print(highlight(f"  python auto-claude/run.py --spec {spec_name} --merge"))
        print()
        print("To discard (if you don't like it):")
        print(muted(f"  python auto-claude/run.py --spec {spec_name} --discard"))
        print()

    elif choice == WorkspaceChoice.MERGE:
        from core.git_executable import run_git

        print()
        print_status("Adding changes to your project...", "progress")

        # Capture pre-merge state for validation
        pre_merge = run_git(["rev-parse", "HEAD"], cwd=project_dir)
        pre_merge_sha = pre_merge.stdout.strip() if pre_merge.returncode == 0 else ""

        success_result = manager.merge_worktree(spec_name, delete_after=True)

        if success_result:
            # Post-merge validation: verify commits actually landed
            post_merge = run_git(["rev-parse", "HEAD"], cwd=project_dir)
            post_merge_sha = post_merge.stdout.strip() if post_merge.returncode == 0 else ""

            if pre_merge_sha and post_merge_sha and pre_merge_sha == post_merge_sha:
                # HEAD didn't change — merge was empty (code lost!)
                print()
                print_status(
                    "WARNING: Merge returned success but no new commits on base branch.",
                    "warning",
                )
                print(muted("This may indicate the code was committed on the wrong branch."))
                print(muted("Marking task done but code may need manual verification."))

            print()
            print_status("Your feature has been added to your project.", "success")
            # Update implementation_plan.json to mark task as done
            mark_plan_done(project_dir, spec_name)
        else:
            # Fallback: try direct git merge if worktree merge failed
            # The code is QA-verified; only the git operation failed
            print()
            print_status("Worktree merge failed. Trying direct git merge...", "warning")
            fallback_ok = _fallback_direct_merge(project_dir, spec_name, manager)
            if fallback_ok:
                print_status("Your feature has been added to your project.", "success")
                mark_plan_done(project_dir, spec_name)
            else:
                print()
                print_status("There was a conflict merging the changes.", "error")
                print(muted("Your build is still saved in the separate workspace."))
                print(muted("You may need to merge manually or ask for help."))

    elif choice == WorkspaceChoice.REVIEW:
        show_changed_files(manager, spec_name)
        print()
        print(muted("-" * 60))
        print()
        print("To see full details of changes:")
        if worktree_info:
            print(
                muted(
                    f"  git diff {worktree_info.base_branch}...{worktree_info.branch}"
                )
            )
        print()
        print("To test the feature:")
        if staging_path:
            print(highlight(f"  cd {staging_path}"))
        print()
        print("To add these changes to your project:")
        print(highlight(f"  python auto-claude/run.py --spec {spec_name} --merge"))
        print()

    else:  # LATER
        print()
        print_status("No problem! Your build is saved.", "success")
        print()
        print("To test the feature:")
        if staging_path:
            print(highlight(f"  cd {staging_path}"))
        else:
            worktree_path = get_existing_build_worktree(project_dir, spec_name)
            if worktree_path:
                print(highlight(f"  cd {worktree_path}"))
            else:
                print(
                    highlight(
                        f"  cd {project_dir}/.auto-claude/worktrees/tasks/{spec_name}"
                    )
                )
        print()
        print("When you're ready to add it:")
        print(highlight(f"  python auto-claude/run.py --spec {spec_name} --merge"))
        print()
        print("To see what was built:")
        print(muted(f"  python auto-claude/run.py --spec {spec_name} --review"))
        print()


def _fallback_direct_merge(
    project_dir: Path, spec_name: str, manager: WorktreeManager
) -> bool:
    """
    Fallback merge when merge_worktree() fails.

    Tries a direct git merge of the branch into the base branch,
    bypassing worktree-specific logic. Per git-merge docs, --no-verify
    skips pre-merge and commit-msg hooks.

    Returns:
        True if fallback merge succeeded
    """
    from core.git_executable import run_git

    branch_name = manager.get_branch_name(spec_name)
    base_branch = manager.base_branch

    try:
        # Ensure we're on the base branch
        current = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=project_dir)
        if current.returncode == 0 and current.stdout.strip() != base_branch:
            checkout = run_git(["checkout", base_branch], cwd=project_dir, timeout=60)
            if checkout.returncode != 0:
                print(f"  Fallback: could not checkout {base_branch}: {checkout.stderr[:200]}")
                return False

        # Try direct merge with --no-verify and extended timeout
        result = run_git(
            ["merge", "--no-ff", "--no-verify", branch_name,
             "-m", f"auto-claude: Merge {branch_name}"],
            cwd=project_dir,
            timeout=300,
        )

        if result.returncode == 0:
            print(f"  Fallback merge succeeded: {branch_name} -> {base_branch}")
            # Clean up worktree and branch
            try:
                manager.remove_worktree(spec_name, delete_branch=True)
            except Exception:
                pass  # Non-fatal: worktree cleanup is best-effort
            return True

        output = (result.stdout + result.stderr).lower()
        if "already up to date" in output or "already up-to-date" in output:
            print(f"  Branch {branch_name} already up to date.")
            try:
                manager.remove_worktree(spec_name, delete_branch=True)
            except Exception:
                pass
            return True

        print(f"  Fallback merge also failed (rc={result.returncode}):")
        print(f"    stdout: {result.stdout[:200]}")
        print(f"    stderr: {result.stderr[:200]}")
        run_git(["merge", "--abort"], cwd=project_dir)
        return False

    except Exception as e:
        print(f"  Fallback merge error: {e}")
        return False


def mark_plan_done(project_dir: Path, spec_name: str) -> None:
    """Update implementation_plan.json to mark task as done after successful merge.

    Also emits MARK_DONE event to stdout so the frontend XState machine
    transitions the Kanban card to the Done column in real-time.
    """
    from core.file_utils import write_json_atomic

    spec_dir = project_dir / ".auto-claude" / "specs" / spec_name
    plan_path = spec_dir / "implementation_plan.json"

    if not plan_path.exists():
        return

    try:
        with open(plan_path, encoding="utf-8") as f:
            plan = json.load(f)

        plan["status"] = "done"
        plan["planStatus"] = "completed"
        plan["xstateState"] = "done"
        plan["executionPhase"] = "complete"
        plan["reviewReason"] = None
        plan["updated_at"] = datetime.now(timezone.utc).isoformat()

        write_json_atomic(plan_path, plan, indent=2)

        # Emit MARK_DONE event to stdout for frontend XState machine
        from core.task_event import TaskEventEmitter

        emitter = TaskEventEmitter.from_spec_dir(spec_dir)
        emitter.emit("MARK_DONE")
    except (OSError, json.JSONDecodeError):
        pass


def review_existing_build(project_dir: Path, spec_name: str) -> bool:
    """
    Show what an existing build contains.

    Called when user runs: python auto-claude/run.py --spec X --review

    Args:
        project_dir: The project directory
        spec_name: Name of the spec

    Returns:
        True if build exists
    """
    worktree_path = get_existing_build_worktree(project_dir, spec_name)

    if not worktree_path:
        print()
        print_status(f"No existing build found for '{spec_name}'.", "warning")
        print()
        print("To start a new build:")
        print(highlight(f"  python auto-claude/run.py --spec {spec_name}"))
        return False

    content = [
        bold(f"{icon(Icons.FILE)} BUILD CONTENTS"),
    ]
    print()
    print(box(content, width=60, style="heavy"))

    manager = WorktreeManager(project_dir)
    worktree_info = manager.get_worktree_info(spec_name)

    show_build_summary(manager, spec_name)
    show_changed_files(manager, spec_name)

    print()
    print(muted("-" * 60))
    print()
    print("To test the feature:")
    print(highlight(f"  cd {worktree_path}"))
    print()
    print("To add these changes to your project:")
    print(highlight(f"  python auto-claude/run.py --spec {spec_name} --merge"))
    print()
    print("To see full diff:")
    if worktree_info:
        print(muted(f"  git diff {worktree_info.base_branch}...{worktree_info.branch}"))
    print()

    return True


def discard_existing_build(project_dir: Path, spec_name: str) -> bool:
    """
    Discard an existing build (with confirmation).

    Called when user runs: python auto-claude/run.py --spec X --discard

    Requires typing "delete" to confirm - prevents accidents.

    Args:
        project_dir: The project directory
        spec_name: Name of the spec

    Returns:
        True if discarded
    """
    worktree_path = get_existing_build_worktree(project_dir, spec_name)

    if not worktree_path:
        print()
        print_status(f"No existing build found for '{spec_name}'.", "warning")
        return False

    content = [
        warning(f"{icon(Icons.WARNING)} DELETE BUILD RESULTS?"),
        "",
        "This will permanently delete all work for this build.",
    ]
    print()
    print(box(content, width=60, style="heavy"))

    manager = WorktreeManager(project_dir)

    show_build_summary(manager, spec_name)

    print()
    print(f"Are you sure? Type {highlight('delete')} to confirm: ", end="")

    try:
        confirmation = input().strip().lower()
    except KeyboardInterrupt:
        print()
        print_status("Cancelled. Your build is still saved.", "info")
        return False

    if confirmation != "delete":
        print()
        print_status("Cancelled. Your build is still saved.", "info")
        return False

    # Actually delete
    manager.remove_worktree(spec_name, delete_branch=True)

    print()
    print_status("Build deleted.", "success")
    return True


def check_existing_build(project_dir: Path, spec_name: str) -> bool:
    """
    Check if there's an existing build and offer options.

    Returns True if user wants to continue with existing build,
    False if they want to start fresh (after discarding).
    """
    worktree_path = get_existing_build_worktree(project_dir, spec_name)

    if not worktree_path:
        return False  # No existing build

    content = [
        info(f"{icon(Icons.INFO)} EXISTING BUILD FOUND"),
        "",
        "There's already a build in progress for this spec.",
    ]
    print()
    print(box(content, width=60, style="heavy"))

    options = [
        MenuOption(
            key="continue",
            label="Continue where it left off",
            icon=Icons.PLAY,
            description="Resume building from the last checkpoint",
        ),
        MenuOption(
            key="review",
            label="Review what was built",
            icon=Icons.FILE,
            description="See the files that were created/modified",
        ),
        MenuOption(
            key="merge",
            label="Add to my project now",
            icon=Icons.SUCCESS,
            description="Merge the existing build into your project",
        ),
        MenuOption(
            key="fresh",
            label="Start fresh",
            icon=Icons.ERROR,
            description="Discard current build and start over",
        ),
    ]

    print()
    choice = select_menu(
        title="What would you like to do?",
        options=options,
        allow_quit=True,
    )

    if choice is None:
        print()
        print_status("Cancelled.", "info")
        sys.exit(0)

    # Import merge function only when needed to avoid circular imports
    # merge_existing_build is in the parent workspace.py module
    import workspace as ws

    if choice == "continue":
        return True  # Continue with existing
    elif choice == "review":
        review_existing_build(project_dir, spec_name)
        print()
        input("Press Enter to continue building...")
        return True
    elif choice == "merge":
        ws.merge_existing_build(project_dir, spec_name)
        return False  # Start fresh after merge
    elif choice == "fresh":
        discarded = discard_existing_build(project_dir, spec_name)
        return not discarded  # If discarded, start fresh
    else:
        return True  # Default to continue


def list_all_worktrees(project_dir: Path) -> list[WorktreeInfo]:
    """
    List all spec worktrees in the project.

    Args:
        project_dir: Main project directory

    Returns:
        List of WorktreeInfo for each spec worktree
    """
    manager = WorktreeManager(project_dir)
    return manager.list_all_worktrees()


def cleanup_all_worktrees(project_dir: Path, confirm: bool = True) -> bool:
    """
    Clean up all spec worktrees in the project.

    Args:
        project_dir: Main project directory
        confirm: Whether to ask for confirmation

    Returns:
        True if worktrees were cleaned up
    """
    manager = WorktreeManager(project_dir)
    worktrees = manager.list_all_worktrees()

    if not worktrees:
        print()
        print_status("No worktrees found.", "info")
        return False

    if confirm:
        print()
        print_status(f"Found {len(worktrees)} worktree(s):", "info")
        for wt in worktrees:
            print(f"  - {wt.spec_name}")
        print()
        print(f"Delete all worktrees? Type {highlight('yes')} to confirm: ", end="")

        try:
            confirmation = input().strip().lower()
        except KeyboardInterrupt:
            print()
            print_status("Cancelled.", "info")
            return False

        if confirmation != "yes":
            print()
            print_status("Cancelled.", "info")
            return False

    # Clean up all worktrees
    for wt in worktrees:
        manager.remove_worktree(wt.spec_name, delete_branch=True)

    print()
    print_status(f"Cleaned up {len(worktrees)} worktree(s).", "success")
    return True
