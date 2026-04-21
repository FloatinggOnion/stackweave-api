"""Wrapper around stackweave.Solver for streaming progress events."""

import asyncio
import json
import tempfile
import os
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
    from stackweave.solver import Solver, ProgressReporter, Constraint, ConstraintSet
    from stackweave import VersionRange
    from stackweave.parsers import parse_manifest
    from stackweave.lockfile import LockfileGenerator
    from packaging.specifiers import SpecifierSet
    HAS_STACKWEAVE = True
    print("✓ Stackweave imported successfully in solver_wrapper")
except ImportError as e:
    HAS_STACKWEAVE = False
    print(f"✗ Failed to import stackweave: {e}")


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


def _normalize_version_spec(version_spec: str) -> str:
    """
    Normalize a version spec to be compatible with packaging.SpecifierSet.

    Handles:
    - Bare package names (e.g., "requests" → "")
    - Wildcards (e.g., "requests*" → "")
    - Wildcard suffixes (e.g., ">=2.0.*" → ">=2.0.0")

    Args:
        version_spec: The version specification string

    Returns:
        A normalized spec string compatible with SpecifierSet
    """
    spec = version_spec.strip()

    # Remove trailing wildcards (e.g., "requests*" → "")
    spec = spec.rstrip('*')

    # Replace wildcard suffixes (e.g., ">=2.0.*" → ">=2.0.0")
    if '.*' in spec:
        spec = spec.replace('.*', '.0')

    # If only an operator is left (e.g., ">=", "<"), it's invalid
    # Return empty string for any spec, which means "any version"
    if not spec or spec in ['>=', '<=', '==', '!=', '~=', '>', '<']:
        return ''

    return spec


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
        # Parse manifest: write text to temp file since parse_manifest expects a file path
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(manifest_text)
            temp_path = f.name

        try:
            parsed = parse_manifest(temp_path)

            # Build root_requirements as a set of package names
            root_requirements = {dep.name for dep in parsed.dependencies}

            # Build constraint_sets from dependencies
            constraint_sets = {}
            for dep in parsed.dependencies:
                # Normalize version spec to handle wildcards and invalid specs
                normalized_spec = _normalize_version_spec(dep.version_spec)

                # Create a VersionRange from the normalized spec
                try:
                    spec_set = SpecifierSet(normalized_spec) if normalized_spec else SpecifierSet("")
                except Exception as e:
                    print(f"Warning: Could not parse version spec '{dep.version_spec}': {e}")
                    # Fallback to any version
                    spec_set = SpecifierSet("")

                version_range = VersionRange(
                    specifier_set=spec_set,
                    original_spec=dep.version_spec
                )

                # Create a Constraint for this package
                constraint = Constraint(
                    package=dep.name,
                    version_range=version_range,
                    markers=None,
                    extras=dep.extras or []
                )

                # Create or update ConstraintSet for this package
                if dep.name not in constraint_sets:
                    constraint_sets[dep.name] = ConstraintSet(
                        package=dep.name,
                        constraints=[constraint]
                    )
                else:
                    constraint_sets[dep.name].constraints.append(constraint)

        finally:
            os.unlink(temp_path)

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
            lockfile_gen = LockfileGenerator()
            lockfile = lockfile_gen.generate(result.solution)

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
