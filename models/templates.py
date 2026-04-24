"""Pydantic models for template definitions and validation."""

from typing import List, Optional, Dict
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import re


class WorkflowType(str, Enum):
    """Supported workflow types."""
    TTS_FINETUNING = "tts-finetuning"
    ASR_FINETUNING = "asr-finetuning"
    SLM_TRAINING = "slm-training"
    TEXT_CLASSIFICATION = "text-classification"


class TemplateMetadata(BaseModel):
    """Template metadata including hardware requirements and cost estimates.

    This metadata helps users understand hardware requirements before selecting
    a template and enables cost estimation for future cloud provisioning.
    """
    gpu_memory_required_gb: int = Field(
        ...,
        ge=1,
        le=512,
        description="GPU memory required in GB (1-512)"
    )
    recommended_gpu: List[str] = Field(
        ...,
        description="List of recommended GPU models (e.g., 'A100', 'A10', 'RTX 4090')"
    )
    default_batch_size: int = Field(
        ...,
        ge=1,
        description="Default batch size for this workflow"
    )
    estimated_cost_per_hour_usd: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated cost per hour in USD (optional, for future cloud provisioning)"
    )


class WorkflowDefinition(BaseModel):
    """A single workflow definition within a template.

    Defines the Python version, system packages, Python packages, and
    environment variables needed for a specific workflow type.
    """
    description: str = Field(
        ...,
        min_length=10,
        description="Description of the workflow (minimum 10 characters)"
    )
    python_version: str = Field(
        default=">=3.10,<3.13",
        description="Python version specifier (e.g., '>=3.10,<3.13')"
    )
    system_packages: List[str] = Field(
        default_factory=list,
        description="List of system packages (e.g., CUDA, cuDNN)"
    )
    python_packages: List[str] = Field(
        ...,
        description="List of Python packages with version specs (PEP 508 format)"
    )
    environment_variables: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of environment variables as dicts with 'name' and 'value' keys"
    )

    @field_validator('python_packages')
    @classmethod
    def validate_python_packages_non_empty(cls, v: List[str]) -> List[str]:
        """Ensure at least one Python package is specified."""
        if not v or len(v) == 0:
            raise ValueError("python_packages must contain at least one package")
        return v


class Template(BaseModel):
    """Complete template definition for an ML workflow.

    Defines a complete, validated template including metadata, workflows,
    versioning, and lock file refresh policy. Templates are the foundation
    for reproducible ML environments.
    """
    name: str = Field(
        ...,
        min_length=5,
        max_length=100,
        description="Template name (5-100 characters)"
    )
    workflow_type: WorkflowType = Field(
        ...,
        description="Type of workflow (enum: tts-finetuning, asr-finetuning, slm-training, text-classification)"
    )
    version: str = Field(
        ...,
        description="Semantic version (e.g., '1.0.0', pattern: MAJOR.MINOR.PATCH)"
    )
    description: str = Field(
        ...,
        min_length=20,
        description="Template description (minimum 20 characters)"
    )
    stability_status: str = Field(
        default="stable",
        description="Stability status: 'stable' or 'deprecated'"
    )
    locked_at: str = Field(
        ...,
        description="ISO 8601 date when template was locked (YYYY-MM-DD format)"
    )
    metadata: TemplateMetadata = Field(
        ...,
        description="Template metadata including hardware requirements"
    )
    workflows: Dict[str, WorkflowDefinition] = Field(
        ...,
        description="Mapping of workflow names to their definitions"
    )
    refresh_by: str = Field(
        ...,
        description="ISO 8601 date when lock files should be refreshed (YYYY-MM-DD format)"
    )

    @field_validator('version')
    @classmethod
    def validate_version_semantic(cls, v: str) -> str:
        """Validate semantic versioning format (MAJOR.MINOR.PATCH)."""
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError(
                f"version must follow semantic versioning (MAJOR.MINOR.PATCH), got '{v}'"
            )
        return v

    @field_validator('locked_at', 'refresh_by')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate ISO 8601 date format (YYYY-MM-DD)."""
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
            raise ValueError(
                f"date must be ISO 8601 format (YYYY-MM-DD), got '{v}'"
            )
        return v

    @field_validator('stability_status')
    @classmethod
    def validate_stability_status(cls, v: str) -> str:
        """Validate stability status is either 'stable' or 'deprecated'."""
        if v not in ('stable', 'deprecated'):
            raise ValueError(
                f"stability_status must be 'stable' or 'deprecated', got '{v}'"
            )
        return v

    @field_validator('workflows')
    @classmethod
    def validate_workflows_non_empty(cls, v: Dict[str, WorkflowDefinition]) -> Dict[str, WorkflowDefinition]:
        """Ensure at least one workflow is defined."""
        if not v or len(v) == 0:
            raise ValueError("workflows must contain at least one workflow definition")
        return v
