"""Tests for GET /api/v1/templates and GET /api/v1/templates/{name} endpoints."""

import pytest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

TEMPLATES_DIR = Path(__file__).parent.parent.parent.parent / "knot-cli" / "stackweave" / "templates"


class TestGetTemplates:
    def test_get_templates_returns_200(self):
        response = client.get("/api/v1/templates")
        assert response.status_code == 200

    def test_get_templates_returns_list(self):
        response = client.get("/api/v1/templates")
        data = response.json()
        assert isinstance(data, list)

    def test_get_templates_non_empty(self):
        """At least 4 templates from knot-cli."""
        response = client.get("/api/v1/templates")
        data = response.json()
        assert len(data) >= 4

    def test_get_templates_has_required_fields(self):
        response = client.get("/api/v1/templates")
        for tmpl in response.json():
            assert "name" in tmpl
            assert "display_name" in tmpl
            assert "description" in tmpl
            assert "supported_hardware" in tmpl
            assert "estimated_runtime_hours" in tmpl

    def test_get_templates_includes_tts_finetuning(self):
        response = client.get("/api/v1/templates")
        names = [t["name"] for t in response.json()]
        assert "tts-finetuning" in names

    def test_get_templates_hardware_has_name_and_vram(self):
        response = client.get("/api/v1/templates")
        for tmpl in response.json():
            for hw in tmpl["supported_hardware"]:
                assert "name" in hw
                assert "vram_gb" in hw

    def test_get_templates_empty_dir(self, tmp_path):
        """Returns empty list when no templates found."""
        with patch("routes.templates_ui._templates_dir", return_value=tmp_path):
            response = client.get("/api/v1/templates")
        assert response.status_code == 200
        assert response.json() == []


class TestGetTemplate:
    def test_get_template_valid_name_returns_200(self):
        response = client.get("/api/v1/templates/tts-finetuning")
        assert response.status_code == 200

    def test_get_template_invalid_name_returns_404(self):
        response = client.get("/api/v1/templates/does-not-exist")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_template_has_full_spec_fields(self):
        response = client.get("/api/v1/templates/tts-finetuning")
        data = response.json()
        assert "full_description" in data
        assert "dependencies" in data
        assert "system_requirements" in data
        assert "models" in data
        assert "python_packages" in data

    def test_get_template_has_pytorch_version(self):
        response = client.get("/api/v1/templates/tts-finetuning")
        data = response.json()
        assert data.get("pytorch_version") is not None

    def test_get_template_has_cuda_version(self):
        response = client.get("/api/v1/templates/tts-finetuning")
        data = response.json()
        assert data.get("cuda_version") is not None

    def test_get_template_dependencies_is_dict(self):
        response = client.get("/api/v1/templates/tts-finetuning")
        data = response.json()
        assert isinstance(data["dependencies"], dict)

    def test_get_all_four_templates(self):
        for name in ["tts-finetuning", "asr-finetuning", "slm-training", "text-classification"]:
            response = client.get(f"/api/v1/templates/{name}")
            assert response.status_code == 200, f"Template {name} should return 200"


class TestCustomizeAndValidate:
    def test_valid_pytorch_cuda_returns_valid(self):
        response = client.post(
            "/api/v1/templates/tts-finetuning/customize-and-validate",
            json={"customizations": {"pytorch": "2.1", "cuda": "11.8"}},
        )
        assert response.status_code == 200
        assert response.json()["valid"] is True

    def test_incompatible_pytorch_cuda_returns_conflict(self):
        response = client.post(
            "/api/v1/templates/tts-finetuning/customize-and-validate",
            json={"customizations": {"pytorch": "2.1", "cuda": "12.4"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert len(data["conflicts"]) > 0

    def test_conflict_includes_suggestions(self):
        response = client.post(
            "/api/v1/templates/tts-finetuning/customize-and-validate",
            json={"customizations": {"pytorch": "2.1", "cuda": "12.4"}},
        )
        data = response.json()
        assert len(data["suggestions"]) > 0

    def test_unknown_template_returns_404(self):
        response = client.post(
            "/api/v1/templates/no-such-template/customize-and-validate",
            json={"customizations": {}},
        )
        assert response.status_code == 404

    def test_no_pytorch_cuda_returns_valid(self):
        response = client.post(
            "/api/v1/templates/tts-finetuning/customize-and-validate",
            json={"customizations": {"batch_size": "8"}},
        )
        assert response.status_code == 200
        assert response.json()["valid"] is True


class TestProvision:
    def test_provision_returns_200(self):
        with client.stream("POST", "/api/v1/templates/tts-finetuning/provision") as r:
            assert r.status_code == 200

    def test_provision_streams_sse_events(self):
        with client.stream("POST", "/api/v1/templates/tts-finetuning/provision") as r:
            text = r.read().decode()
        assert "event: provisioning_start" in text
        assert "event: ready" in text

    def test_provision_unknown_template_returns_404(self):
        response = client.post("/api/v1/templates/no-such-template/provision")
        assert response.status_code == 404
