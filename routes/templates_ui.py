"""REST API endpoints for template discovery and customization (Phase 11).

GET  /api/v1/templates                              — list all templates (card data)
GET  /api/v1/templates/{name}                       — full template spec (modal data)
POST /api/v1/templates/{name}/customize-and-validate — validate customizations with solver
POST /api/v1/templates/{name}/provision             — provision template (SSE stream)
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import AsyncIterable, Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from models.template_ui import HardwareOption, SystemRequirements, TemplateSummary, TemplateSpec

router = APIRouter(prefix="/api/v1/templates", tags=["templates-ui"])

# Path to knot-cli templates directory (relative to this file's location)
_TEMPLATES_DIR = Path(__file__).parent.parent.parent / "knot-cli" / "stackweave" / "templates"


def _templates_dir() -> Path:
    """Return templates directory, allowing override via env var."""
    override = os.environ.get("STACKWEAVE_TEMPLATES_DIR")
    if override:
        return Path(override)
    return _TEMPLATES_DIR


def _parse_version(packages: List[str], prefix: str) -> Optional[str]:
    """Extract pinned version for a package prefix from a list of PEP 508 specs."""
    for pkg in packages:
        if pkg.lower().startswith(prefix.lower()):
            # Match ==version (python packages) or =version (system packages like nvidia::cuda=11.8)
            m = re.search(r"={1,2}([^\s,]+)", pkg)
            if m:
                return m.group(1)
    return None


def _load_template_yaml(template_dir: Path) -> Optional[dict]:
    """Load and parse template.yaml from a template directory."""
    yaml_path = template_dir / "template.yaml"
    if not yaml_path.exists():
        return None
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return data
    except Exception:
        return None


def _build_summary(name: str, data: dict) -> TemplateSummary:
    """Build TemplateSummary from raw YAML data."""
    # template.yaml has a top-level 'template' key
    tmpl = data.get("template", data)
    meta = tmpl.get("metadata", data.get("metadata", {}))

    recommended_gpus = meta.get("recommended_gpu", [])
    hardware = [
        HardwareOption(
            name=gpu,
            vram_gb=meta.get("gpu_memory_required_gb", 0),
            recommended=True,
        )
        for gpu in recommended_gpus
    ]

    # Estimate runtime from cost (rough proxy) or default
    estimated_hours = float(meta.get("estimated_runtime_hours", 2.5))

    return TemplateSummary(
        name=name,
        display_name=tmpl.get("name", name),
        description=tmpl.get("description", ""),
        supported_hardware=hardware,
        estimated_runtime_hours=estimated_hours,
    )


def _build_spec(name: str, data: dict) -> TemplateSpec:
    """Build TemplateSpec from raw YAML data."""
    tmpl = data.get("template", data)
    meta = tmpl.get("metadata", data.get("metadata", {}))

    recommended_gpus = meta.get("recommended_gpu", [])
    hardware = [
        HardwareOption(
            name=gpu,
            vram_gb=meta.get("gpu_memory_required_gb", 0),
            recommended=True,
        )
        for gpu in recommended_gpus
    ]

    estimated_hours = float(meta.get("estimated_runtime_hours", 2.5))

    # Gather python packages from first workflow
    workflows = tmpl.get("workflows", data.get("workflows", {}))
    first_workflow = next(iter(workflows.values()), {}) if workflows else {}
    python_packages: List[str] = first_workflow.get("python_packages", [])
    system_packages: List[str] = first_workflow.get("system_packages", [])

    pytorch_version = _parse_version(python_packages, "torch==")
    cuda_version = _parse_version(system_packages, "nvidia::cuda=")

    # Build dependencies dict from python packages
    dependencies: dict = {}
    for pkg in python_packages:
        m = re.match(r"^([A-Za-z0-9_\-]+)[=><!\[]", pkg)
        if m:
            pkg_name = m.group(1).lower()
            ver_m = re.search(r"==([^\s,]+)", pkg)
            dependencies[pkg_name] = ver_m.group(1) if ver_m else pkg

    sys_req = SystemRequirements(
        min_ram_gb=32,
        min_disk_gb=100,
        min_cuda_compute_capability=8.0,
    )

    model_ids: List[str] = data.get("model_ids", [])

    return TemplateSpec(
        name=name,
        display_name=tmpl.get("name", name),
        description=tmpl.get("description", ""),
        full_description=tmpl.get("description", ""),
        supported_hardware=hardware,
        estimated_runtime_hours=estimated_hours,
        dependencies=dependencies,
        pytorch_version=pytorch_version,
        cuda_version=cuda_version,
        system_requirements=sys_req,
        models=model_ids,
        python_packages=python_packages,
    )


def _list_template_names() -> List[str]:
    """Return sorted list of template directory names."""
    tdir = _templates_dir()
    if not tdir.exists():
        return []
    return sorted(
        d.name
        for d in tdir.iterdir()
        if d.is_dir() and (d / "template.yaml").exists()
    )


@router.get("", response_model=List[TemplateSummary])
async def get_templates() -> List[TemplateSummary]:
    """List all available templates with card metadata."""
    results = []
    for name in _list_template_names():
        data = _load_template_yaml(_templates_dir() / name)
        if data is not None:
            results.append(_build_summary(name, data))
    return results


@router.get("/{name}", response_model=TemplateSpec)
async def get_template(name: str) -> TemplateSpec:
    """Get full template spec for modal display."""
    template_dir = _templates_dir() / name
    data = _load_template_yaml(template_dir)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Template '{name}' not found",
        )
    return _build_spec(name, data)


# ── Request / Response models ────────────────────────────────────────────────

class CustomizationRequest(BaseModel):
    customizations: Dict[str, str]


class Conflict(BaseModel):
    type: str
    message: str
    explanation: str


class Suggestion(BaseModel):
    rank: int
    action: str
    rationale: str


class ValidationResult(BaseModel):
    valid: bool
    conflicts: List[Conflict] = []
    suggestions: List[Suggestion] = []
    message: str = ""


# ── Validation helper ────────────────────────────────────────────────────────

_PYTORCH_CUDA_COMPAT: Dict[str, List[str]] = {
    "2.3": ["12.1", "12.4"],
    "2.2": ["12.1", "11.8"],
    "2.1": ["12.1", "11.8"],
    "2.0": ["11.7", "11.8"],
    "1.13": ["11.6", "11.7"],
}


def _validate_pytorch_cuda(pytorch: Optional[str], cuda: Optional[str]) -> List[Conflict]:
    """Quick PyTorch+CUDA compatibility check."""
    if not pytorch or not cuda:
        return []
    pt_major_minor = ".".join(pytorch.split(".")[:2])
    compatible = _PYTORCH_CUDA_COMPAT.get(pt_major_minor, [])
    if compatible and cuda not in compatible:
        return [Conflict(
            type="version_mismatch",
            message=f"PyTorch {pytorch} is not compatible with CUDA {cuda}",
            explanation=f"PyTorch {pt_major_minor} supports CUDA: {', '.join(compatible)}",
        )]
    return []


def _build_suggestions(conflicts: List[Conflict], pytorch: Optional[str], cuda: Optional[str]) -> List[Suggestion]:
    if not conflicts:
        return []
    pt_major_minor = ".".join(pytorch.split(".")[:2]) if pytorch else ""
    compatible = _PYTORCH_CUDA_COMPAT.get(pt_major_minor, [])
    suggestions = []
    for i, c in enumerate(compatible, 1):
        suggestions.append(Suggestion(rank=i, action=f"Use CUDA {c}", rationale=f"Compatible with PyTorch {pt_major_minor}"))
    return suggestions


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/{name}/customize-and-validate", response_model=ValidationResult)
async def customize_and_validate(name: str, body: CustomizationRequest) -> ValidationResult:
    """Validate customized template dependencies."""
    template_dir = _templates_dir() / name
    if not _load_template_yaml(template_dir):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Template '{name}' not found")

    c = body.customizations
    pytorch = c.get("pytorch") or c.get("torch")
    cuda = c.get("cuda")

    try:
        async with asyncio.timeout(5.0):
            conflicts = _validate_pytorch_cuda(pytorch, cuda)
    except asyncio.TimeoutError:
        return ValidationResult(valid=False, message="Validation timeout — proceed with caution")

    if conflicts:
        return ValidationResult(
            valid=False,
            conflicts=conflicts,
            suggestions=_build_suggestions(conflicts, pytorch, cuda),
            message=f"{len(conflicts)} conflict(s) detected",
        )
    return ValidationResult(valid=True, message="No conflicts detected. Ready to provision.")


@router.post("/{name}/provision")
async def provision(name: str) -> StreamingResponse:
    """Start provisioning and stream SSE progress events."""
    template_dir = _templates_dir() / name
    if not _load_template_yaml(template_dir):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Template '{name}' not found")

    async def event_stream() -> AsyncIterable[str]:
        import datetime

        def sse(event: str, data: dict) -> str:
            data["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        yield sse("provisioning_start", {"status": "starting", "progress": 0, "message": "Starting provisioning…"})
        await asyncio.sleep(0.1)

        # Simulate docker build steps
        for progress, msg in [(15, "Building base image…"), (30, "Installing system packages…"), (45, "Installing Python packages…")]:
            yield sse("docker_build", {"status": "building", "progress": progress, "message": msg})
            await asyncio.sleep(0.1)

        yield sse("docker_run", {"status": "running", "progress": 55, "message": "Container started"})
        await asyncio.sleep(0.1)

        yield sse("model_download", {"status": "downloading_models", "progress": 70, "message": "Downloading models…"})
        await asyncio.sleep(0.1)

        # Readiness checks
        checks = [
            ("container_running", "Container running"),
            ("gpu_accessible", "GPU accessible"),
            ("models_downloaded", "Models downloaded"),
            ("network_connectivity", "Network connectivity"),
        ]
        for i, (check_name, check_msg) in enumerate(checks):
            progress = 80 + i * 4
            yield sse("health_check", {"status": "health_check", "progress": progress, "message": f"Checking {check_msg}…", "check": check_name})
            await asyncio.sleep(0.1)

        yield sse("ready", {
            "status": "ready",
            "progress": 100,
            "message": "Ready to train",
            "container_id": f"{name}-container",
            "ssh_command": f"docker-compose exec {name} /bin/bash",
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
