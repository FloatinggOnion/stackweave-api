"""Pydantic models for Stackweave API."""

from typing import List, Optional, Dict
from enum import Enum
from pydantic import BaseModel, Field


class ManifestType(str, Enum):
    """Supported manifest file types."""
    REQUIREMENTS_TXT = "requirements.txt"
    ENVIRONMENT_YML = "environment.yml"
    PYPROJECT_TOML = "pyproject.toml"


class SolveRequest(BaseModel):
    """Request to solve dependencies."""
    manifest_text: str = Field(..., min_length=1, description="Contents of the manifest file")
    manifest_type: ManifestType = Field(..., description="Type of manifest file")


class Conflict(BaseModel):
    """A conflict in dependency resolution."""
    package: str = Field(..., description="Package name involved in conflict")
    constraint_mismatch: str = Field(..., description="Description of conflicting constraints")
    suggestion: str = Field(..., description="Suggested fix")
    severity: str = Field(..., pattern="^(critical|warning)$", description="Severity level")


# Solution is just a Dict[str, str] type alias
# No need for a separate class in Pydantic v2


class Dependency(BaseModel):
    """An edge in the dependency graph."""
    source: str = Field(..., description="Source package name")
    target: str = Field(..., description="Target package name")
    constraint: str = Field(..., description="Version constraint")
    resolved: bool = Field(..., description="Whether this dependency was successfully resolved")


class SolvingProgress(BaseModel):
    """Progress update during solving."""
    round: int = Field(..., ge=0, description="Current resolution round")
    total_rounds: int = Field(..., ge=1, description="Total rounds expected")
    pinned_packages: List[str] = Field(default_factory=list, description="Packages pinned so far")


class SolveCompleteEvent(BaseModel):
    """Final result of solving."""
    conflicts: List[Conflict] = Field(default_factory=list, description="Conflicts found, if any")
    solution: Dict[str, str] = Field(default_factory=dict, description="Package -> version mapping")
    dependencies: List[Dependency] = Field(default_factory=list, description="Dependency graph edges")
    lockfile: str = Field(..., description="Generated lockfile content")
    solver_time: float = Field(..., ge=0, description="Time spent solving in seconds")


class ErrorEvent(BaseModel):
    """An error event from the solver."""
    error: str = Field(..., description="Error message")
    suggestion: Optional[str] = Field(None, description="Suggested fix or next steps")
