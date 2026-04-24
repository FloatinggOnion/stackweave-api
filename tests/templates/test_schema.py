"""Unit tests for template schema validation."""

import pytest
import yaml
from pydantic import ValidationError
from models.templates import Template, TemplateMetadata, WorkflowDefinition, WorkflowType


# =============================================================================
# Test Loading Valid Templates
# =============================================================================

class TestLoadValidTemplate:
    """Test loading and validating valid template dictionaries."""

    @pytest.mark.unit
    def test_load_valid_template_dict(self, valid_tts_template_dict):
        """Test loading a valid template dict and converting to Template model."""
        template = Template.model_validate(valid_tts_template_dict)

        assert template.name == "TTS Fine-tuning"
        assert template.workflow_type == WorkflowType.TTS_FINETUNING
        assert template.version == "1.0.0"
        assert len(template.workflows) > 0

    @pytest.mark.unit
    def test_load_valid_template_yaml(self, valid_template_yaml):
        """Test loading a valid template YAML string with safe_load and validation."""
        data = yaml.safe_load(valid_template_yaml)
        template = Template.model_validate(data)

        assert template.name == "TTS Fine-tuning"
        assert template.stability_status == "stable"


# =============================================================================
# Test Required Fields
# =============================================================================

class TestValidateRequiredFields:
    """Test that all required fields are enforced."""

    @pytest.mark.unit
    def test_template_requires_name(self, invalid_template_missing_name):
        """Test that Template requires 'name' field."""
        with pytest.raises(ValidationError) as exc_info:
            Template.model_validate(invalid_template_missing_name)

        error = exc_info.value
        assert "name" in str(error)

    @pytest.mark.unit
    def test_template_requires_version(self, valid_tts_template_dict):
        """Test that Template requires 'version' field."""
        template = valid_tts_template_dict.copy()
        del template["version"]

        with pytest.raises(ValidationError) as exc_info:
            Template.model_validate(template)

        error = exc_info.value
        assert "version" in str(error)

    @pytest.mark.unit
    def test_template_requires_workflows(self, valid_tts_template_dict):
        """Test that Template requires 'workflows' field."""
        template = valid_tts_template_dict.copy()
        del template["workflows"]

        with pytest.raises(ValidationError) as exc_info:
            Template.model_validate(template)

        error = exc_info.value
        assert "workflows" in str(error)

    @pytest.mark.unit
    def test_template_requires_metadata(self, valid_tts_template_dict):
        """Test that Template requires 'metadata' field."""
        template = valid_tts_template_dict.copy()
        del template["metadata"]

        with pytest.raises(ValidationError) as exc_info:
            Template.model_validate(template)

        error = exc_info.value
        assert "metadata" in str(error)


# =============================================================================
# Test WorkflowDefinition Validation
# =============================================================================

class TestValidateWorkflowDefinition:
    """Test WorkflowDefinition field constraints."""

    @pytest.mark.unit
    def test_workflow_requires_python_packages(self, invalid_template_empty_packages):
        """Test that WorkflowDefinition rejects empty python_packages list."""
        with pytest.raises(ValidationError) as exc_info:
            Template.model_validate(invalid_template_empty_packages)

        error = exc_info.value
        assert "python_packages" in str(error)

    @pytest.mark.unit
    def test_workflow_python_version_has_default(self, valid_tts_template_dict):
        """Test that python_version defaults to '>=3.10,<3.13' if not specified."""
        template_dict = valid_tts_template_dict.copy()
        del template_dict["workflows"]["tts-finetuning"]["python_version"]

        template = Template.model_validate(template_dict)
        workflow = template.workflows["tts-finetuning"]

        assert workflow.python_version == ">=3.10,<3.13"

    @pytest.mark.unit
    def test_workflow_system_packages_optional(self, valid_tts_template_dict):
        """Test that system_packages can be empty list."""
        template_dict = valid_tts_template_dict.copy()
        template_dict["workflows"]["tts-finetuning"]["system_packages"] = []

        template = Template.model_validate(template_dict)
        assert template.workflows["tts-finetuning"].system_packages == []


# =============================================================================
# Test Metadata Validation
# =============================================================================

class TestValidateMetadata:
    """Test TemplateMetadata field constraints."""

    @pytest.mark.unit
    def test_metadata_gpu_memory_valid_range(self, valid_tts_template_dict):
        """Test that GPU memory in valid range (1-512 GB) is accepted."""
        for gb in [1, 24, 80, 512]:
            template_dict = valid_tts_template_dict.copy()
            template_dict["metadata"]["gpu_memory_required_gb"] = gb

            template = Template.model_validate(template_dict)
            assert template.metadata.gpu_memory_required_gb == gb

    @pytest.mark.unit
    def test_metadata_gpu_memory_invalid_below_1(self, valid_tts_template_dict):
        """Test that GPU memory below 1 GB is rejected."""
        template_dict = valid_tts_template_dict.copy()
        template_dict["metadata"]["gpu_memory_required_gb"] = 0

        with pytest.raises(ValidationError) as exc_info:
            Template.model_validate(template_dict)

        error = exc_info.value
        assert "gpu_memory_required_gb" in str(error)

    @pytest.mark.unit
    def test_metadata_gpu_memory_invalid_above_512(self, valid_tts_template_dict):
        """Test that GPU memory above 512 GB is rejected."""
        template_dict = valid_tts_template_dict.copy()
        template_dict["metadata"]["gpu_memory_required_gb"] = 1000

        with pytest.raises(ValidationError) as exc_info:
            Template.model_validate(template_dict)

        error = exc_info.value
        assert "gpu_memory_required_gb" in str(error)

    @pytest.mark.unit
    def test_metadata_requires_recommended_gpu_list(self, valid_tts_template_dict):
        """Test that recommended_gpu is required field."""
        template_dict = valid_tts_template_dict.copy()
        del template_dict["metadata"]["recommended_gpu"]

        with pytest.raises(ValidationError) as exc_info:
            Template.model_validate(template_dict)

        error = exc_info.value
        assert "recommended_gpu" in str(error)

    @pytest.mark.unit
    def test_metadata_cost_per_hour_optional(self, valid_tts_template_dict):
        """Test that estimated_cost_per_hour_usd is optional."""
        template_dict = valid_tts_template_dict.copy()
        del template_dict["metadata"]["estimated_cost_per_hour_usd"]

        template = Template.model_validate(template_dict)
        assert template.metadata.estimated_cost_per_hour_usd is None

    @pytest.mark.unit
    def test_metadata_cost_per_hour_negative_rejected(self, valid_tts_template_dict):
        """Test that negative cost is rejected."""
        template_dict = valid_tts_template_dict.copy()
        template_dict["metadata"]["estimated_cost_per_hour_usd"] = -1.0

        with pytest.raises(ValidationError) as exc_info:
            Template.model_validate(template_dict)

        error = exc_info.value
        assert "estimated_cost_per_hour_usd" in str(error)


# =============================================================================
# Test Versioning and Date Validation
# =============================================================================

class TestValidateVersioning:
    """Test version and date format validation."""

    @pytest.mark.unit
    def test_version_semantic_versioning_valid(self, valid_tts_template_dict):
        """Test that valid semantic versions are accepted."""
        for version in ["1.0.0", "2.5.3", "0.1.0", "10.20.30"]:
            template_dict = valid_tts_template_dict.copy()
            template_dict["version"] = version

            template = Template.model_validate(template_dict)
            assert template.version == version

    @pytest.mark.unit
    def test_version_semantic_versioning_invalid(self, invalid_template_bad_version):
        """Test that invalid semantic versions are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Template.model_validate(invalid_template_bad_version)

        error = exc_info.value
        assert "semantic" in str(error).lower() or "version" in str(error).lower()

    @pytest.mark.unit
    def test_locked_at_iso8601_valid(self, valid_tts_template_dict):
        """Test that valid ISO 8601 dates are accepted for locked_at."""
        template_dict = valid_tts_template_dict.copy()
        template_dict["locked_at"] = "2026-04-24"

        template = Template.model_validate(template_dict)
        assert template.locked_at == "2026-04-24"

    @pytest.mark.unit
    def test_locked_at_iso8601_invalid(self, valid_tts_template_dict):
        """Test that invalid date formats are rejected for locked_at."""
        template_dict = valid_tts_template_dict.copy()
        template_dict["locked_at"] = "2026/04/24"

        with pytest.raises(ValidationError) as exc_info:
            Template.model_validate(template_dict)

        error = exc_info.value
        assert "locked_at" in str(error)

    @pytest.mark.unit
    def test_refresh_by_iso8601_valid(self, valid_tts_template_dict):
        """Test that valid ISO 8601 dates are accepted for refresh_by."""
        template_dict = valid_tts_template_dict.copy()
        template_dict["refresh_by"] = "2026-10-24"

        template = Template.model_validate(template_dict)
        assert template.refresh_by == "2026-10-24"


# =============================================================================
# Test Stability Status Validation
# =============================================================================

class TestValidateStabilityStatus:
    """Test stability_status enum validation."""

    @pytest.mark.unit
    def test_stability_status_default_stable(self, valid_tts_template_dict):
        """Test that stability_status defaults to 'stable'."""
        template_dict = valid_tts_template_dict.copy()
        del template_dict["stability_status"]

        template = Template.model_validate(template_dict)
        assert template.stability_status == "stable"

    @pytest.mark.unit
    def test_stability_status_stable_valid(self, valid_tts_template_dict):
        """Test that 'stable' is a valid stability_status."""
        template_dict = valid_tts_template_dict.copy()
        template_dict["stability_status"] = "stable"

        template = Template.model_validate(template_dict)
        assert template.stability_status == "stable"

    @pytest.mark.unit
    def test_stability_status_deprecated_valid(self, valid_tts_template_dict):
        """Test that 'deprecated' is a valid stability_status."""
        template_dict = valid_tts_template_dict.copy()
        template_dict["stability_status"] = "deprecated"

        template = Template.model_validate(template_dict)
        assert template.stability_status == "deprecated"

    @pytest.mark.unit
    def test_stability_status_beta_invalid(self, valid_tts_template_dict):
        """Test that 'beta' is rejected (only stable/deprecated allowed)."""
        template_dict = valid_tts_template_dict.copy()
        template_dict["stability_status"] = "beta"

        with pytest.raises(ValidationError) as exc_info:
            Template.model_validate(template_dict)

        error = exc_info.value
        assert "stable" in str(error) and "deprecated" in str(error)

    @pytest.mark.unit
    def test_stability_status_case_sensitive(self, valid_tts_template_dict):
        """Test that stability_status is case-sensitive (STABLE is invalid)."""
        template_dict = valid_tts_template_dict.copy()
        template_dict["stability_status"] = "STABLE"

        with pytest.raises(ValidationError) as exc_info:
            Template.model_validate(template_dict)

        error = exc_info.value
        assert "stable" in str(error)


# =============================================================================
# Test WorkflowType Enum Validation
# =============================================================================

class TestValidateWorkflowType:
    """Test workflow_type enum validation."""

    @pytest.mark.unit
    def test_workflow_type_tts_finetuning_valid(self, valid_tts_template_dict):
        """Test that 'tts-finetuning' is a valid workflow_type."""
        template_dict = valid_tts_template_dict.copy()
        template_dict["workflow_type"] = "tts-finetuning"

        template = Template.model_validate(template_dict)
        assert template.workflow_type == WorkflowType.TTS_FINETUNING

    @pytest.mark.unit
    def test_workflow_type_asr_finetuning_valid(self, valid_tts_template_dict):
        """Test that 'asr-finetuning' is a valid workflow_type."""
        template_dict = valid_tts_template_dict.copy()
        template_dict["workflow_type"] = "asr-finetuning"

        template = Template.model_validate(template_dict)
        assert template.workflow_type == WorkflowType.ASR_FINETUNING

    @pytest.mark.unit
    def test_workflow_type_slm_training_valid(self, valid_tts_template_dict):
        """Test that 'slm-training' is a valid workflow_type."""
        template_dict = valid_tts_template_dict.copy()
        template_dict["workflow_type"] = "slm-training"

        template = Template.model_validate(template_dict)
        assert template.workflow_type == WorkflowType.SLM_TRAINING

    @pytest.mark.unit
    def test_workflow_type_text_classification_valid(self, valid_tts_template_dict):
        """Test that 'text-classification' is a valid workflow_type."""
        template_dict = valid_tts_template_dict.copy()
        template_dict["workflow_type"] = "text-classification"

        template = Template.model_validate(template_dict)
        assert template.workflow_type == WorkflowType.TEXT_CLASSIFICATION

    @pytest.mark.unit
    def test_workflow_type_invalid(self, valid_tts_template_dict):
        """Test that unknown workflow types are rejected."""
        template_dict = valid_tts_template_dict.copy()
        template_dict["workflow_type"] = "unknown-workflow"

        with pytest.raises(ValidationError) as exc_info:
            Template.model_validate(template_dict)

        error = exc_info.value
        assert "workflow_type" in str(error)


# =============================================================================
# Test JSON Schema Generation
# =============================================================================

class TestJsonSchema:
    """Test JSON Schema generation for documentation and IDE support."""

    @pytest.mark.unit
    def test_json_schema_generated(self):
        """Test that Template generates valid JSON Schema."""
        schema = Template.model_json_schema()

        assert schema is not None
        assert isinstance(schema, dict)
        assert "$schema" in schema or "properties" in schema

    @pytest.mark.unit
    def test_json_schema_has_required_fields(self):
        """Test that JSON Schema includes required fields."""
        schema = Template.model_json_schema()

        required = schema.get("required", [])
        assert "name" in required
        assert "version" in required
        assert "workflows" in required
        assert "metadata" in required

    @pytest.mark.unit
    def test_json_schema_gpu_memory_constraints(self):
        """Test that JSON Schema includes GPU memory constraints."""
        schema = Template.model_json_schema()

        # Navigate to metadata properties
        metadata_schema = schema["$defs"]["TemplateMetadata"]
        gpu_props = metadata_schema["properties"]["gpu_memory_required_gb"]

        assert gpu_props.get("minimum") == 1
        assert gpu_props.get("maximum") == 512
