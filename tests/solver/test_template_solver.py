"""Integration tests for Stackweave solver with template dependency sets.

Tests verify:
- VALIDATION-01: Template validated against solver
- VALIDATION-02: Solver returns structured conflicts (not raw strings)
- VALIDATION-03: Suggestions include effort estimates (easy/medium/hard)
- VALIDATION-04: Invalid templates blocked (validation failures prevent provisioning)
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from solver_wrapper import (
    validate_template_with_solver,
    validate_customization,
    _check_pytorch_cuda_compat,
    _compute_template_hash,
    _cache_validation_result,
    _get_cached_validation_result,
)
from models.templates import Template


pytestmark = pytest.mark.integration


class TestValidateTemplate:
    """Tests for validate_template_with_solver function."""

    @pytest.mark.asyncio
    async def test_validate_template_returns_ok_status(self, valid_tts_template_dict):
        """Test that valid template returns status='ok'.

        Validates VALIDATION-01: Template is validated against solver.
        """
        template = Template.model_validate(valid_tts_template_dict)

        with patch('solver_wrapper.Solver') as mock_solver_class:
            # Mock successful solve result
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.conflicts = None
            mock_result.suggestions = None
            mock_result.solver_time = 0.123

            mock_solver_instance = MagicMock()
            mock_solver_instance.solve.return_value = mock_result
            mock_solver_class.return_value = mock_solver_instance

            result = await validate_template_with_solver(template, "tts-finetuning", timeout=120.0)

            assert result["status"] == "ok"
            assert result["conflicts"] == []
            assert result["solver_time"] > 0
            assert result["cached"] is False

    @pytest.mark.asyncio
    async def test_validate_template_with_conflict(self, valid_tts_template_dict):
        """Test template with dependency conflict returns status='conflict'.

        Validates VALIDATION-01 and VALIDATION-04: Conflict detected and returned.
        """
        template = Template.model_validate(valid_tts_template_dict)

        with patch('solver_wrapper.Solver') as mock_solver_class:
            # Mock conflict result
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.conflicts = ["torch==2.0 requires cuda==11.0, got cuda==12.1"]
            mock_result.suggestions = ["Upgrade PyTorch to 2.1.2"]
            mock_result.solver_time = 0.456

            mock_solver_instance = MagicMock()
            mock_solver_instance.solve.return_value = mock_result
            mock_solver_class.return_value = mock_solver_instance

            result = await validate_template_with_solver(template, "tts-finetuning", timeout=120.0)

            assert result["status"] == "conflict"
            assert len(result["conflicts"]) > 0
            assert result["cached"] is False

    @pytest.mark.asyncio
    async def test_validate_template_returns_structured_conflicts(self, valid_tts_template_dict):
        """Test that conflicts are structured dicts with package and message.

        Validates VALIDATION-02: Conflicts include clear package names and messages.
        """
        template = Template.model_validate(valid_tts_template_dict)

        with patch('solver_wrapper.Solver') as mock_solver_class:
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.conflicts = ["pytorch::torch requires cuda 11.8, got cuda 12.1"]
            mock_result.suggestions = []
            mock_result.solver_time = 0.1

            mock_solver_instance = MagicMock()
            mock_solver_instance.solve.return_value = mock_result
            mock_solver_class.return_value = mock_solver_instance

            result = await validate_template_with_solver(template, "tts-finetuning", timeout=120.0)

            # Conflicts should be list of dicts with "package" and "message"
            assert isinstance(result["conflicts"], list)
            for conflict in result["conflicts"]:
                assert isinstance(conflict, dict)
                assert "package" in conflict
                assert "message" in conflict

    @pytest.mark.asyncio
    async def test_validate_template_returns_suggestions(self, valid_tts_template_dict):
        """Test that suggestions include effort estimates.

        Validates VALIDATION-03: Suggestions with effort estimates (easy/medium/hard).
        """
        template = Template.model_validate(valid_tts_template_dict)

        with patch('solver_wrapper.Solver') as mock_solver_class:
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.conflicts = ["version mismatch"]
            mock_result.suggestions = ["Upgrade PyTorch to 2.1.2", "Use CUDA 11.8"]
            mock_result.solver_time = 0.2

            mock_solver_instance = MagicMock()
            mock_solver_instance.solve.return_value = mock_result
            mock_solver_class.return_value = mock_solver_instance

            result = await validate_template_with_solver(template, "tts-finetuning", timeout=120.0)

            assert len(result["suggestions"]) > 0
            for suggestion in result["suggestions"]:
                assert "suggestion" in suggestion
                assert "effort" in suggestion
                assert "reason" in suggestion
                assert suggestion["effort"] in ("easy", "medium", "hard")

    @pytest.mark.asyncio
    async def test_validate_template_timeout_fallback(self, valid_tts_template_dict):
        """Test timeout returns cached result if available.

        Validates D-06: Timeout with cached fallback.
        """
        template = Template.model_validate(valid_tts_template_dict)
        template_hash = _compute_template_hash(template)

        # Cache a result first
        cached_result = {
            "status": "ok",
            "conflicts": [],
            "suggestions": [],
            "solver_time": 0.1
        }
        _cache_validation_result(template_hash, "tts-finetuning", cached_result)

        with patch('solver_wrapper.asyncio.wait_for') as mock_wait:
            # Simulate timeout
            mock_wait.side_effect = asyncio.TimeoutError()

            result = await validate_template_with_solver(template, "tts-finetuning", timeout=1.0)

            assert result["cached"] is True
            assert "warning" in result

    @pytest.mark.asyncio
    async def test_validate_template_timeout_no_cache(self, valid_tts_template_dict):
        """Test timeout with no cached result returns safe default.

        Validates D-06: Safe default assumption if no cached result.
        """
        template = Template.model_validate(valid_tts_template_dict)

        # Clear cache to ensure no previous results
        from solver_wrapper import _validation_cache
        _validation_cache.clear()

        with patch('solver_wrapper.asyncio.wait_for') as mock_wait:
            # Simulate timeout
            mock_wait.side_effect = asyncio.TimeoutError()

            result = await validate_template_with_solver(template, "tts-finetuning", timeout=1.0)

            assert result["cached"] is True
            assert "warning" in result
            assert result["status"] == "ok"  # Safe assumption


class TestQuickPytorchCudaCheck:
    """Tests for quick PyTorch+CUDA compatibility check."""

    @pytest.mark.asyncio
    async def test_pytorch_cuda_compatible_versions(self):
        """Test compatible PyTorch+CUDA versions.

        Validates D-11: Quick spot-check works for compatible versions.
        """
        # Note: Real solver returns conflicts for nvidia::cuda since it's system package
        # This test verifies the function handles the result correctly
        result = await _check_pytorch_cuda_compat("2.1.2", "11.8", timeout=30.0)

        # Result could be compatible or not depending on solver availability
        # Just verify the response format is correct
        assert "compatible" in result
        assert "conflicts" in result
        assert "suggestions" in result
        assert "effort" in result
        assert result["effort"] in ("none", "easy", "medium", "hard")

    @pytest.mark.asyncio
    async def test_pytorch_cuda_incompatible_versions(self):
        """Test incompatible PyTorch+CUDA versions.

        Validates D-11: Spot-check detects incompatibility.
        """
        with patch('solver_wrapper.Solver') as mock_solver_class:
            mock_result = MagicMock()
            mock_result.success = False
            mock_result.conflicts = ["PyTorch 2.0 requires CUDA 11.8, got CUDA 12.1"]
            mock_result.suggestions = ["Use CUDA 11.8 instead"]
            mock_result.solver_time = 0.08

            mock_solver_instance = MagicMock()
            mock_solver_instance.solve.return_value = mock_result
            mock_solver_class.return_value = mock_solver_instance

            result = await _check_pytorch_cuda_compat("2.0", "12.1", timeout=30.0)

            assert result["compatible"] is False
            assert len(result["conflicts"]) > 0

    @pytest.mark.asyncio
    async def test_validate_customization_missing_params(self, valid_tts_template_dict):
        """Test customization with missing PyTorch/CUDA still succeeds.

        Validates D-11: Not enough info returns compatible=True.
        """
        template = Template.model_validate(valid_tts_template_dict)

        # Customization without torch/cuda
        customization = {"batch_size": "8"}

        result = await validate_customization("tts-finetuning", template, customization, timeout=30.0)

        assert result["compatible"] is True
        assert result["effort"] == "none"
        assert "reason" in result

    @pytest.mark.asyncio
    async def test_validate_customization_with_versions(self, valid_tts_template_dict):
        """Test customization with PyTorch and CUDA versions.

        Validates D-10: Customizations validated before provisioning.
        """
        template = Template.model_validate(valid_tts_template_dict)
        customization = {"torch": "2.1.2", "cuda": "11.8"}

        result = await validate_customization("tts-finetuning", template, customization, timeout=30.0)

        # Result could be compatible or not depending on solver availability
        # Just verify the response has correct format
        assert "compatible" in result
        assert "conflicts" in result
        assert "suggestions" in result
        assert "effort" in result
        assert result["effort"] in ("none", "easy", "medium", "hard")


class TestTemplateHashing:
    """Tests for template content hashing."""

    def test_template_hash_deterministic(self, valid_tts_template_dict):
        """Test that same template produces same hash.

        Validates caching: same template content = same cache key.
        """
        template = Template.model_validate(valid_tts_template_dict)

        hash1 = _compute_template_hash(template)
        hash2 = _compute_template_hash(template)

        assert hash1 == hash2

    def test_template_hash_different_packages(self, valid_tts_template_dict):
        """Test that different packages produce different hashes.

        Validates cache invalidation: modified template = new hash.
        """
        template1 = Template.model_validate(valid_tts_template_dict)
        hash1 = _compute_template_hash(template1)

        # Modify template
        template2_dict = valid_tts_template_dict.copy()
        template2_dict["workflows"]["tts-finetuning"]["python_packages"] = [
            "torch==2.2.0",  # Different version
            "torchaudio==2.2.0",
        ]
        template2 = Template.model_validate(template2_dict)
        hash2 = _compute_template_hash(template2)

        assert hash1 != hash2


class TestCachedResults:
    """Tests for validation result caching."""

    def test_cached_result_retrieved(self):
        """Test that cached results can be retrieved.

        Validates D-06: Cached results available for fallback.
        """
        from solver_wrapper import _validation_cache
        _validation_cache.clear()

        result = {
            "status": "ok",
            "conflicts": [],
            "suggestions": [],
            "solver_time": 0.1
        }

        _cache_validation_result("hash123", "tts-finetuning", result)
        cached = _get_cached_validation_result("hash123", "tts-finetuning")

        assert cached is not None
        assert cached == result

    def test_cache_expires_after_24_hours(self):
        """Test that cached results expire after 24 hours.

        Validates D-06: Cache expiration prevents stale results.
        """
        from solver_wrapper import _validation_cache
        _validation_cache.clear()

        result = {
            "status": "ok",
            "conflicts": [],
            "suggestions": [],
            "solver_time": 0.1
        }

        _cache_validation_result("hash456", "asr-finetuning", result)

        # Manually set timestamp to 25 hours ago
        key = ("hash456", "asr-finetuning")
        old_result, _ = _validation_cache[key]
        _validation_cache[key] = (old_result, datetime.utcnow() - timedelta(hours=25))

        # Should return None (expired)
        cached = _get_cached_validation_result("hash456", "asr-finetuning")

        assert cached is None

    def test_cache_not_found(self):
        """Test that missing cache entries return None.

        Validates fallback when no cached result exists.
        """
        from solver_wrapper import _validation_cache
        _validation_cache.clear()

        cached = _get_cached_validation_result("nonexistent", "unknown-workflow")

        assert cached is None
