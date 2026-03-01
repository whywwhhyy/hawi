"""Tests for Model parameter validation and error handling.

Tests that model classes validate their inputs properly and
that errors are properly propagated as events.
"""

import asyncio
import pytest
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

from hawi.agent import HawiAgent
from hawi.models import DeepSeekModel, KimiModel
from hawi.events import AgentErrorEvent, ModelErrorEvent
from hawi.errors import AgentError, ModelError


class TestModelValidation:
    """Tests for model input validation."""

    def test_deepseek_model_rejects_list_model_id(self):
        """Test that DeepSeekModel rejects list model_id with clear error."""
        with pytest.raises((ValueError, TypeError)) as exc_info:
            DeepSeekModel(
                model_id=["deepseek-chat", "deepseek-reasoner"],
                api_key="test-key",
            )
        assert "model_id" in str(exc_info.value).lower() or "string" in str(exc_info.value).lower()

    def test_deepseek_model_rejects_invalid_model_id_type(self):
        """Test that DeepSeekModel rejects non-string model_id."""
        with pytest.raises((ValueError, TypeError)):
            DeepSeekModel(
                model_id={"name": "deepseek-chat"},
                api_key="test-key",
            )

    def test_deepseek_model_accepts_string_model_id(self):
        """Test that DeepSeekModel accepts string model_id."""
        # Should not raise
        model = DeepSeekModel(
            model_id="deepseek-chat",
            api_key="test-key",
        )
        assert model.model_id == "deepseek-chat"

    def test_kimi_model_rejects_list_model_id(self):
        """Test that KimiModel rejects list model_id with clear error."""
        with pytest.raises((ValueError, TypeError)) as exc_info:
            KimiModel(
                model_id=["kimi-k2.5", "kimi-k2.0"],
                api_key="test-key",
            )
        assert "model_id" in str(exc_info.value).lower() or "string" in str(exc_info.value).lower()

    def test_kimi_model_accepts_string_model_id(self):
        """Test that KimiModel accepts string model_id."""
        model = KimiModel(
            model_id="kimi-k2.5",
            api_key="test-key",
        )
        assert model.model_id == "kimi-k2.5"


class TestAgentErrorHandling:
    """Tests for agent error event propagation."""

    @pytest.mark.asyncio
    async def test_model_error_raises_exception(self):
        """Test that model errors result in exceptions being raised (not silently swallowed).

        This is a regression test for the bug where model preparation errors
        were silently swallowed and no events were produced.
        """
        from hawi.errors import AgentError

        # Create a mock model that fails during preparation
        mock_model = MagicMock()
        mock_model.model_id = "test-model"

        # Make astream return an async context manager that raises on enter
        class FailingAsyncContextManager:
            async def __aenter__(self):
                raise AttributeError("'list' object has no attribute 'startswith'")
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        mock_model.astream = MagicMock(return_value=FailingAsyncContextManager())

        agent = HawiAgent(
            model=mock_model,
            system_prompt="You are a test assistant.",
        )

        # Exception should be raised (not silently swallowed)
        with pytest.raises(AgentError) as exc_info:
            await agent.arun("test message")

        # The error should contain the original error message
        assert "startswith" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_model_init_error_is_not_silent(self):
        """Test that model initialization errors are not silently swallowed.

        This ensures that if a model's __init__ or astream raises an exception,
        the error propagates properly rather than being silently ignored.
        """
        from hawi.errors import AgentError

        # Create a mock model that fails when astream is called
        mock_model = MagicMock()
        mock_model.model_id = "test-model"

        # Make astream return an async context manager that raises on enter
        class BrokenAsyncContextManager:
            async def __aenter__(self):
                raise AttributeError("'list' object has no attribute 'startswith'")
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        mock_model.astream = MagicMock(return_value=BrokenAsyncContextManager())

        agent = HawiAgent(
            model=mock_model,
            system_prompt="Test",
        )

        # Exception should be raised (not silently swallowed)
        with pytest.raises(AgentError) as exc_info:
            await agent.arun("test")

        # The error should contain the original error message
        assert "startswith" in str(exc_info.value)
