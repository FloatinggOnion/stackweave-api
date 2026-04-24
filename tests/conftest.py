"""Pytest configuration and fixtures for Stackweave template tests."""

import pytest
import yaml
from unittest.mock import AsyncMock


@pytest.fixture
def valid_tts_template_dict():
    """Fixture providing a valid TTS fine-tuning template as a dict.

    This template includes all required fields and follows the YAML structure
    for TTS fine-tuning workflows. Used for testing successful template loading.
    """
    return {
        "name": "TTS Fine-tuning",
        "workflow_type": "tts-finetuning",
        "version": "1.0.0",
        "description": "Fine-tune Text-to-Speech models (Qwen3-TTS, SpeechT5, Kokoro)",
        "stability_status": "stable",
        "locked_at": "2026-04-24",
        "refresh_by": "2026-10-24",
        "metadata": {
            "gpu_memory_required_gb": 24,
            "recommended_gpu": ["A100", "A10", "RTX 4090"],
            "default_batch_size": 4,
            "estimated_cost_per_hour_usd": 3.06,
        },
        "workflows": {
            "tts-finetuning": {
                "description": "TTS fine-tuning environment for training on custom voice data",
                "python_version": ">=3.10,<3.13",
                "system_packages": [
                    "nvidia::cuda=11.8",
                    "nvidia::cudnn=8.9.7",
                    "libsndfile-dev=1.2.2",
                ],
                "python_packages": [
                    "torch==2.1.2",
                    "torchaudio==2.1.2",
                    "transformers==4.40.2",
                    "accelerate==0.26.0",
                    "peft==0.7.0",
                    "huggingface-hub[hf_transfer]>=0.19.0",
                ],
                "environment_variables": [
                    {"name": "HF_HUB_ENABLE_HF_TRANSFER", "value": "1"},
                    {"name": "CUDA_VISIBLE_DEVICES", "value": "0"},
                ],
            }
        },
    }


@pytest.fixture
def valid_template_yaml(valid_tts_template_dict):
    """Fixture providing a valid template as a YAML string.

    Returns YAML-formatted template string that can be loaded with yaml.safe_load().
    Used for testing YAML parsing integration with Pydantic validation.
    """
    return yaml.dump(valid_tts_template_dict)


@pytest.fixture
def invalid_template_missing_name(valid_tts_template_dict):
    """Fixture providing a template dict missing the required 'name' field.

    Used for testing that Pydantic validation correctly rejects templates
    without a required name field.
    """
    template = valid_tts_template_dict.copy()
    del template["name"]
    return template


@pytest.fixture
def invalid_template_bad_version(valid_tts_template_dict):
    """Fixture providing a template with invalid semantic version format.

    Version '1.0' does not match semantic versioning (MAJOR.MINOR.PATCH).
    Used for testing version validation.
    """
    template = valid_tts_template_dict.copy()
    template["version"] = "1.0"
    return template


@pytest.fixture
def invalid_template_empty_packages(valid_tts_template_dict):
    """Fixture providing a template with empty python_packages list.

    The @field_validator in WorkflowDefinition should reject this because
    python_packages must contain at least one package.
    """
    template = valid_tts_template_dict.copy()
    template["workflows"]["tts-finetuning"]["python_packages"] = []
    return template


@pytest.fixture
def mock_solver():
    """Fixture providing a mock Solver class for async testing.

    Returns an AsyncMock that can be used to simulate Stackweave solver
    responses without making actual solver calls. Useful for solver integration
    tests in Wave 2.
    """
    return AsyncMock()
