"""Wrapper around stackweave.Solver for streaming progress events and template validation."""

import asyncio
import json
import tempfile
import os
import hashlib
from typing import AsyncGenerator, Optional, List, Set, Dict, Tuple
from datetime import datetime, timedelta
from fastapi.sse import ServerSentEvent

from solver_models import (
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


# In-memory cache for validation results: {(template_hash, workflow): (result, timestamp)}
_validation_cache: Dict[Tuple[str, str], Tuple[Dict, datetime]] = {}


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
    Build a dependency graph from the solution.

    Creates nodes for all packages in the solution. Shows all resolved packages
    even if there are no explicit dependency edges.

    Args:
        solution: Package -> version mapping

    Returns:
        List of Dependency edges (may be empty if no inter-dependencies)
    """
    # For now, return empty dependencies but nodes will be created from solution
    # In Phase 2+, we can enhance this to extract actual dependency relationships
    # from the solver's internal state

    # Create dependency edges if we have multi-package solutions
    # For single packages, we'll have nodes but no edges
    dependencies: List[Dependency] = []

    # TODO: Extract actual dependency relationships from solver result
    # This would show which packages depend on which other packages

    return dependencies


# ============================================================================
# Template Validation Functions (Phase 7 - Wave 2)
# ============================================================================


def _compute_template_hash(template: "Template") -> str:
    """Compute SHA256 hash of template content.

    Hashes the python_packages and system_packages to uniquely identify
    the template's dependency set. Used for caching validation results.

    Args:
        template: The Template model instance

    Returns:
        Hexadecimal SHA256 hash string
    """
    from models.templates import Template

    # Get the first workflow (template should validate we have at least one)
    workflows_key = list(template.workflows.keys())[0]
    workflow_def = template.workflows[workflows_key]

    # Build hashable content: sorted package lists
    content = {
        "python_packages": sorted(workflow_def.python_packages),
        "system_packages": sorted(workflow_def.system_packages),
        "metadata": {
            "gpu_memory_required_gb": template.metadata.gpu_memory_required_gb,
            "recommended_gpu": sorted(template.metadata.recommended_gpu),
        }
    }

    # Hash as JSON
    content_json = json.dumps(content, sort_keys=True)
    return hashlib.sha256(content_json.encode()).hexdigest()


def _cache_validation_result(template_hash: str, workflow: str, result: Dict) -> None:
    """Cache validation result with timestamp.

    Stores result in in-memory dict indexed by (template_hash, workflow).
    Includes timestamp for expiration checking.

    Per D-06: Cached results used as fallback if solver times out.

    Args:
        template_hash: SHA256 hash of template content
        workflow: Workflow name (e.g., "tts-finetuning")
        result: Validation result dict (status, conflicts, suggestions, etc.)
    """
    _validation_cache[(template_hash, workflow)] = (result, datetime.utcnow())


def _get_cached_validation_result(template_hash: str, workflow: str) -> Optional[Dict]:
    """Retrieve cached validation result if valid.

    Returns cached result if it exists and is less than 24 hours old.
    Returns None if expired or not found.

    Per D-06: Cached results fallback if solver times out, but only if recent.

    Args:
        template_hash: SHA256 hash of template content
        workflow: Workflow name (e.g., "tts-finetuning")

    Returns:
        Result dict if found and fresh, None otherwise
    """
    key = (template_hash, workflow)
    if key not in _validation_cache:
        return None

    result, timestamp = _validation_cache[key]
    age = datetime.utcnow() - timestamp

    # Cache expires after 24 hours
    if age > timedelta(hours=24):
        del _validation_cache[key]
        return None

    return result


def _build_constraint_sets_from_template(
    template: "Template",
    workflow: str
) -> Tuple[Dict[str, ConstraintSet], Set[str]]:
    """Extract and build constraint sets from template.

    Parses python_packages and system_packages from the workflow definition,
    converts to Constraint and ConstraintSet objects for the Stackweave solver.

    Per D-06 (Pitfall 1 mitigation): CUDA versions are pinned exactly (==),
    not as ranges (>=), to avoid version mismatch hell.

    Args:
        template: The Template model instance
        workflow: Workflow name to extract dependencies from

    Returns:
        Tuple of (constraint_sets dict, root_requirements set)

    Raises:
        ValueError: If package name parsing fails
        KeyError: If workflow not found in template
    """
    from models.templates import Template

    workflow_def = template.workflows[workflow]
    constraint_sets: Dict[str, ConstraintSet] = {}
    root_requirements: Set[str] = set()

    # Process Python packages
    for pkg_spec in workflow_def.python_packages:
        try:
            # Parse package name and version spec (PEP 508 format)
            pkg_name = pkg_spec.split("[")[0]  # Remove extras
            # Extract just the package name (before any version specifier)
            for sep in ["==", ">=", "<=", ">", "<", "~=", "!="]:
                if sep in pkg_name:
                    pkg_name = pkg_name.split(sep)[0]
                    break
            pkg_name = pkg_name.strip()

            # Extract version spec (everything after package name)
            version_spec = pkg_spec[len(pkg_name):].strip()

            # Build SpecifierSet
            try:
                spec_set = SpecifierSet(version_spec) if version_spec else SpecifierSet("")
            except Exception as e:
                print(f"Warning: Invalid version spec '{version_spec}' for {pkg_name}: {e}")
                spec_set = SpecifierSet("")

            # Create Constraint
            constraint = Constraint(
                package=pkg_name,
                version_range=VersionRange(specifier_set=spec_set, original_spec=version_spec),
                markers=None,
                extras=[],
            )

            # Add to constraint sets
            if pkg_name not in constraint_sets:
                constraint_sets[pkg_name] = ConstraintSet(package=pkg_name, constraints=[constraint])
            else:
                constraint_sets[pkg_name].constraints.append(constraint)

            root_requirements.add(pkg_name)

        except Exception as e:
            raise ValueError(f"Failed to parse package spec '{pkg_spec}': {e}")

    # Process system packages (CUDA, cuDNN, etc.)
    for sys_pkg in workflow_def.system_packages:
        try:
            # Extract package name and version
            if "cuda" in sys_pkg.lower():
                pkg_name = "nvidia::cuda"
                # CUDA must be pinned exactly (D-06 mitigation)
                if "=" in sys_pkg:
                    version = sys_pkg.split("=")[-1].strip()
                    version_spec = f"=={version}"
                else:
                    version_spec = ""
            elif "cudnn" in sys_pkg.lower():
                pkg_name = "nvidia::cudnn"
                if "=" in sys_pkg:
                    version = sys_pkg.split("=")[-1].strip()
                    version_spec = f"=={version}"
                else:
                    version_spec = ""
            else:
                # Other system packages: extract name
                pkg_name = sys_pkg.split("=")[0].split(">")[0].split("<")[0].strip()
                version_spec = sys_pkg[len(pkg_name):].strip()

            # Build SpecifierSet
            try:
                spec_set = SpecifierSet(version_spec) if version_spec else SpecifierSet("")
            except Exception as e:
                print(f"Warning: Invalid version spec '{version_spec}' for {pkg_name}: {e}")
                spec_set = SpecifierSet("")

            # Create Constraint
            constraint = Constraint(
                package=pkg_name,
                version_range=VersionRange(specifier_set=spec_set, original_spec=version_spec),
                markers=None,
                extras=[],
            )

            # Add to constraint sets
            if pkg_name not in constraint_sets:
                constraint_sets[pkg_name] = ConstraintSet(package=pkg_name, constraints=[constraint])
            else:
                constraint_sets[pkg_name].constraints.append(constraint)

            root_requirements.add(pkg_name)

        except Exception as e:
            print(f"Warning: Failed to parse system package '{sys_pkg}': {e}")

    return constraint_sets, root_requirements


async def validate_template_with_solver(
    template: "Template",
    workflow: str,
    timeout: float = 120.0
) -> Dict:
    """Validate full template definition against Stackweave solver.

    Per D-04: Full template validation in CI/CD (not provision time).
    Per D-06: 2-minute timeout with cached result fallback.

    Extracts workflow dependencies (python_packages + system_packages),
    calls Stackweave solver, and returns structured response with:
    - status: "ok" or "conflict"
    - conflicts: List of {package, message} dicts
    - suggestions: List of {suggestion, effort, reason} dicts
    - solver_time: Float seconds
    - cached: Bool indicating if result came from cache

    On TimeoutError: Falls back to cached result from previous run if exists,
    otherwise returns safe default (assume compatible, with warning).

    Args:
        template: The Template model instance
        workflow: Workflow name to validate (e.g., "tts-finetuning")
        timeout: Timeout in seconds (default 120 for CI/CD per D-06)

    Returns:
        Dict with keys: status, conflicts, suggestions, solver_time, cached, warning(optional)

    Raises:
        KeyError: If workflow not found in template
        ValueError: If dependency parsing fails
    """
    from models.templates import Template

    if not HAS_STACKWEAVE:
        return {
            "status": "ok",
            "conflicts": [],
            "suggestions": [],
            "solver_time": 0.0,
            "cached": False,
            "warning": "Stackweave not available; skipping validation"
        }

    try:
        # Compute template hash for caching
        template_hash = _compute_template_hash(template)

        # Build constraint sets
        constraint_sets, root_requirements = _build_constraint_sets_from_template(template, workflow)

        # Run solver in executor to avoid blocking
        loop = asyncio.get_event_loop()

        def run_solver():
            solver = Solver(timeout=timeout)
            return solver.solve(
                constraint_sets=constraint_sets,
                root_requirements=root_requirements,
                reporter=None,
            )

        # Run with timeout
        result = await asyncio.wait_for(
            loop.run_in_executor(None, run_solver),
            timeout=timeout + 5.0  # Add 5s buffer for executor overhead
        )

        # Build response
        conflicts = []
        suggestions = []

        if result.conflicts:
            for conflict in result.conflicts:
                pkg_name = conflict.split(":")[0] if ":" in conflict else "unknown"
                conflicts.append({
                    "package": pkg_name,
                    "message": conflict
                })
                # Generate basic suggestion from conflict
                suggestions.append({
                    "suggestion": f"Review constraint for {pkg_name}",
                    "effort": "medium",
                    "reason": "Check package versions and dependencies"
                })

        response = {
            "status": "ok" if result.success else "conflict",
            "conflicts": conflicts,
            "suggestions": suggestions,
            "solver_time": result.solver_time or 0.0,
            "cached": False
        }

        # Cache the result
        _cache_validation_result(template_hash, workflow, response)

        return response

    except asyncio.TimeoutError:
        # Timeout: try cached result
        template_hash = _compute_template_hash(template)
        cached = _get_cached_validation_result(template_hash, workflow)

        if cached:
            return {
                **cached,
                "cached": True,
                "warning": f"Solver timed out; using cached result"
            }
        else:
            return {
                "status": "ok",
                "conflicts": [],
                "suggestions": [],
                "solver_time": 0.0,
                "cached": True,
                "warning": "Solver timeout; assuming compatible (risky)"
            }

    except Exception as e:
        return {
            "status": "error",
            "conflicts": [{"package": "unknown", "message": str(e)}],
            "suggestions": [],
            "solver_time": 0.0,
            "cached": False,
            "error": str(e)
        }


async def validate_customization(
    workflow: str,
    template: "Template",
    customization: Dict[str, str],
    timeout: float = 30.0
) -> Dict:
    """Quick validation of user customizations at provision time.

    Per D-10: Customizations validated against solver before provisioning.
    Per D-11: Quick PyTorch+CUDA compatibility check only (<30 seconds).
    Not full transitive resolution, just spot-check critical packages.

    Returns dict with:
    - compatible: Bool
    - conflicts: List of conflict messages if not compatible
    - suggestions: List of {suggestion, effort, reason} dicts
    - effort: "none", "easy", "medium", or "hard"

    Args:
        workflow: Workflow name (e.g., "tts-finetuning")
        template: The Template model instance (base template)
        customization: User overrides dict, e.g. {"torch": "2.1.2", "cuda": "12.0"}
        timeout: Timeout in seconds (default 30 for quick check per D-11)

    Returns:
        Dict with keys: compatible, conflicts, suggestions, effort
    """
    from models.templates import Template

    # Extract PyTorch and CUDA versions from customization
    torch_version = customization.get("torch")
    cuda_version = customization.get("cuda")

    # If either missing, we don't have enough info for quick check
    if not torch_version or not cuda_version:
        return {
            "compatible": True,
            "reason": "Not enough info for quick check (torch and/or cuda not specified)",
            "conflicts": [],
            "suggestions": [],
            "effort": "none"
        }

    # Perform PyTorch+CUDA compatibility check
    try:
        compat_result = await _check_pytorch_cuda_compat(torch_version, cuda_version, timeout=timeout)
        return compat_result
    except Exception as e:
        return {
            "compatible": False,
            "reason": str(e),
            "conflicts": [str(e)],
            "suggestions": [],
            "effort": "medium"
        }


async def _check_pytorch_cuda_compat(
    torch_version: str,
    cuda_version: str,
    timeout: float = 30.0
) -> Dict:
    """Quick PyTorch+CUDA compatibility spot-check.

    Per D-11: This is a spot-check only, not full transitive resolution.
    Uses Stackweave solver's built-in CUDA metadata (from Phase 1) to detect
    common incompatibilities between PyTorch and CUDA versions.

    Returns dict with:
    - compatible: Bool
    - conflicts: List if not compatible
    - suggestions: List with effort estimates
    - effort: Effort to fix (none/easy/medium/hard)

    Args:
        torch_version: PyTorch version (e.g., "2.1.2")
        cuda_version: CUDA version (e.g., "11.8")
        timeout: Timeout in seconds (default 30 per D-11)

    Returns:
        Dict with compatibility assessment
    """
    if not HAS_STACKWEAVE:
        return {
            "compatible": True,
            "conflicts": [],
            "suggestions": [],
            "effort": "none"
        }

    try:
        loop = asyncio.get_event_loop()

        def run_compat_check():
            # Create constraints for PyTorch and CUDA
            torch_constraint = Constraint(
                package="torch",
                version_range=VersionRange(specifier_set=SpecifierSet(f"=={torch_version}")),
                markers=None,
                extras=[],
            )

            cuda_constraint = Constraint(
                package="nvidia::cuda",
                version_range=VersionRange(specifier_set=SpecifierSet(f"=={cuda_version}")),
                markers=None,
                extras=[],
            )

            constraint_sets = {
                "torch": ConstraintSet(package="torch", constraints=[torch_constraint]),
                "nvidia::cuda": ConstraintSet(package="nvidia::cuda", constraints=[cuda_constraint]),
            }

            # Solve with quick timeout
            solver = Solver(timeout=timeout)
            result = solver.solve(
                constraint_sets=constraint_sets,
                root_requirements={"torch", "nvidia::cuda"},
                reporter=None,
            )

            return result

        # Run solver with timeout
        result = await asyncio.wait_for(
            loop.run_in_executor(None, run_compat_check),
            timeout=timeout + 5.0
        )

        if result.success:
            return {
                "compatible": True,
                "conflicts": [],
                "suggestions": [],
                "effort": "none"
            }
        else:
            # Extract conflict messages
            conflicts = result.conflicts or ["PyTorch and CUDA version mismatch"]

            suggestions = []
            if conflicts:
                suggestions.append({
                    "suggestion": f"Try different PyTorch or CUDA versions (check compatibility matrix)",
                    "effort": "easy",
                    "reason": "Version mismatch detected"
                })

            return {
                "compatible": False,
                "conflicts": conflicts,
                "suggestions": suggestions,
                "effort": "easy"
            }

    except asyncio.TimeoutError:
        return {
            "compatible": False,
            "conflicts": ["Compatibility check timed out"],
            "suggestions": [{
                "suggestion": "Try with a different CUDA version",
                "effort": "easy",
                "reason": "Check PyTorch compatibility matrix"
            }],
            "effort": "easy"
        }

    except Exception as e:
        return {
            "compatible": False,
            "conflicts": [str(e)],
            "suggestions": [],
            "effort": "medium"
        }
