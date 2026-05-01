"""Pydantic models for Phase 11 template UI endpoints.

TemplateSummary — card display (list endpoint)
TemplateSpec    — full spec display (detail endpoint)
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class HardwareOption(BaseModel):
    name: str
    vram_gb: int
    recommended: bool = False


class SystemRequirements(BaseModel):
    min_ram_gb: int = 0
    min_disk_gb: int = 0
    min_cuda_compute_capability: float = 0.0


class TemplateSummary(BaseModel):
    """Template metadata for card display."""
    name: str = Field(..., description="Template slug (e.g. 'tts-finetuning')")
    display_name: str = Field(..., description="Human-readable name")
    description: str
    supported_hardware: List[HardwareOption] = Field(default_factory=list)
    estimated_runtime_hours: float = Field(default=0.0)


class TemplateSpec(TemplateSummary):
    """Full template spec for modal display."""
    full_description: str = ""
    dependencies: Dict[str, str] = Field(default_factory=dict)
    pytorch_version: Optional[str] = None
    cuda_version: Optional[str] = None
    system_requirements: SystemRequirements = Field(default_factory=SystemRequirements)
    models: List[str] = Field(default_factory=list)
    python_packages: List[str] = Field(default_factory=list)
