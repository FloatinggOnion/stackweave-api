"""FastAPI routes for template validation and customization endpoints.

Provides endpoints for:
- POST /templates/{workflow}/validate — Full template validation (CI/CD, 120s timeout)
- POST /templates/{workflow}/customize — Quick customization validation (provision time, 30s timeout)

Per D-04: Validation is primarily at CI/CD time (full solver check).
Per D-10: User customizations validated before provisioning.
Per D-11: Customization validation uses quick PyTorch+CUDA check (<30s).
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict
from models.templates import Template
from solver_wrapper import validate_template_with_solver, validate_customization

router = APIRouter(prefix="/templates", tags=["templates"])

# Valid workflow types
VALID_WORKFLOWS = {
    "tts-finetuning",
    "asr-finetuning",
    "slm-training",
    "text-classification"
}


@router.post("/{workflow}/validate", response_model=Dict)
async def validate_template_endpoint(workflow: str, template: Template) -> Dict:
    """Validate full template definition against Stackweave solver.

    Per D-04: Full template validation in CI/CD pipeline (not provision time).
    Validates all template dependencies against Stackweave solver with
    2-minute timeout per D-06. Returns detailed conflict information and
    suggestions per D-07.

    This endpoint is called by:
    - CI/CD pipeline before merging template changes (Wave 3)
    - Template author review workflows

    Args:
        workflow: Workflow name (e.g., "tts-finetuning")
        template: Complete Template model (FastAPI auto-validates via Pydantic)

    Returns:
        Dict with keys:
        - status: "ok" | "conflict"
        - conflicts: List of {package, message} dicts
        - suggestions: List of {suggestion, effort (easy/medium/hard), reason} dicts
        - solver_time: Float seconds
        - cached: Bool (True if result came from cache due to timeout)
        - warning: Optional string if timeout/cache used

    Raises:
        HTTPException(400): Invalid workflow type
        HTTPException(422): Malformed template (Pydantic validation error)
        HTTPException(500): Solver error
    """
    # Validate workflow type
    if workflow not in VALID_WORKFLOWS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid workflow '{workflow}'. Must be one of: {', '.join(sorted(VALID_WORKFLOWS))}"
        )

    try:
        # Call solver wrapper with 120-second timeout (D-06)
        result = await validate_template_with_solver(
            template=template,
            workflow=workflow,
            timeout=120.0
        )

        # Check for solver errors
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Solver error: {result.get('error', 'Unknown error')}"
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Solver error: {str(e)}"
        )


@router.post("/{workflow}/customize", response_model=Dict)
async def validate_customization_endpoint(
    workflow: str,
    template: Template,
    customization: Dict[str, str]
) -> Dict:
    """Quick validation of user customizations before provisioning.

    Per D-10: Customizations validated against solver before provisioning starts.
    Per D-11: Quick PyTorch+CUDA compatibility check only (<30s timeout),
    not full transitive resolution. Balances speed vs coverage for user experience.

    This endpoint is called by:
    - Provisioning UI (before docker-compose launch)
    - Customization preview endpoints

    Args:
        workflow: Workflow name (e.g., "tts-finetuning")
        template: Base Template model
        customization: User customization dict, e.g. {"torch": "2.1.2", "cuda": "11.8"}

    Returns:
        Dict with keys:
        - compatible: Bool (True if customization is compatible)
        - conflicts: List of conflict messages if not compatible
        - suggestions: List of {suggestion, effort, reason} dicts
        - effort: "none" | "easy" | "medium" | "hard" — effort to fix conflicts
        - reason: Optional string explaining why compatible=True (if missing params)

    Raises:
        HTTPException(400): Invalid workflow type
        HTTPException(422): Malformed template or customization dict
        HTTPException(500): Solver error
    """
    # Validate workflow type
    if workflow not in VALID_WORKFLOWS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid workflow '{workflow}'. Must be one of: {', '.join(sorted(VALID_WORKFLOWS))}"
        )

    try:
        # Call solver wrapper with 30-second timeout (D-11)
        result = await validate_customization(
            workflow=workflow,
            template=template,
            customization=customization,
            timeout=30.0
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Customization validation error: {str(e)}"
        )
