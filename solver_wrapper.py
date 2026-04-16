"""Wrapper around stackweave.Solver for streaming progress events."""

import asyncio
import json
from typing import AsyncGenerator, Optional, List, Set, Dict
from fastapi.sse import ServerSentEvent

from models import (
    Conflict,
    Dependency,
    SolveCompleteEvent,
    ErrorEvent,
    SolvingProgress,
)

try:
    from stackweave.solver import Solver, ProgressReporter
    from stackweave.parsers import parse_manifest
    from stackweave.lockfile import generate_lockfile
    HAS_STACKWEAVE = True
except ImportError:
    HAS_STACKWEAVE = False


async def solve_manifest(
    manifest_text: str,
    manifest_type: str,
) -> AsyncGenerator[ServerSentEvent, None]:
    """
    Solve a manifest file and stream progress events.

    Yields ServerSentEvent objects with event types:
    - parsing: manifest is being parsed
    - solving: constraint solving in progress (with round number)
    - lockfile: lockfile is being generated
    - complete: final result with solution and lockfile
    - error: an error occurred

    Args:
        manifest_text: Contents of the manifest file
        manifest_type: Type of manifest ("requirements.txt", "environment.yml", "pyproject.toml")

    Yields:
        ServerSentEvent objects for streaming
    """

    if not HAS_STACKWEAVE:
        yield ServerSentEvent(
            data=json.dumps({"error": "Stackweave solver not available"}),
            event="error"
        )
        return

    try:
        # Emit parsing event
        yield ServerSentEvent(
            data=json.dumps({}),
            event="parsing"
        )

        # Run in thread to avoid blocking the event loop
        # The Solver is synchronous
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _run_solver,
            manifest_text,
            manifest_type,
        )

        # Extract results
        if result["success"]:
            # Emit lockfile event
            yield ServerSentEvent(
                data=json.dumps({}),
                event="lockfile"
            )

            # Build dependency graph from solution
            dependencies = _build_dependency_graph(result.get("solution", {}))

            # Build complete event
            complete_data = SolveCompleteEvent(
                conflicts=[],
                solution=result.get("solution", {}),
                dependencies=dependencies,
                lockfile=result.get("lockfile", ""),
                solver_time=result.get("solver_time", 0.0),
            )

            yield ServerSentEvent(
                data=complete_data.model_dump_json(),
                event="complete"
            )
        else:
            # Build conflicts
            conflicts = [
                Conflict(
                    package="unknown",
                    constraint_mismatch=msg,
                    suggestion="Check constraint compatibility",
                    severity="critical",
                )
                for msg in result.get("conflicts", [])
            ]

            error_msg = result.get("error", "Unknown error")
            yield ServerSentEvent(
                data=json.dumps({
                    "error": error_msg,
                    "suggestion": "Reduce constraints or check manifest syntax"
                }),
                event="error"
            )

    except Exception as e:
        error_event = ErrorEvent(
            error=str(e),
            suggestion="Check manifest format and constraints"
        )
        yield ServerSentEvent(
            data=error_event.model_dump_json(),
            event="error"
        )


def _run_solver(manifest_text: str, manifest_type: str) -> Dict:
    """
    Run the solver synchronously (to be called in executor).

    Returns a dict with:
    - success: bool
    - solution: Dict[str, str] if successful
    - conflicts: List[str] if not successful
    - lockfile: str if successful
    - solver_time: float
    - error: Optional[str]
    """
    try:
        # Parse manifest
        parsed = parse_manifest(manifest_text, manifest_type)
        constraint_sets = parsed["constraint_sets"]
        root_requirements = parsed["root_requirements"]

        # Create and run solver
        solver = Solver(timeout=300.0)

        # No progress reporter for now (simpler MVP)
        result = solver.solve(
            constraint_sets=constraint_sets,
            root_requirements=root_requirements,
            reporter=None,
        )

        if result.success and result.solution:
            # Generate lockfile
            lockfile = generate_lockfile(result.solution)

            return {
                "success": True,
                "solution": result.solution,
                "lockfile": lockfile,
                "solver_time": result.solver_time,
            }
        else:
            return {
                "success": False,
                "conflicts": result.conflicts or ["Unknown conflict"],
                "error": result.error,
                "solver_time": result.solver_time,
            }

    except Exception as e:
        return {
            "success": False,
            "conflicts": [str(e)],
            "error": str(e),
            "solver_time": 0.0,
        }


def _build_dependency_graph(solution: Dict[str, str]) -> List[Dependency]:
    """
    Build a basic dependency graph from the solution.

    For now, returns empty list. In Phase 2+, this will be enhanced
    to show actual dependency edges with constraints.

    Args:
        solution: Package -> version mapping

    Returns:
        List of Dependency edges
    """
    return []
