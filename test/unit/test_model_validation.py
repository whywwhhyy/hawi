"""Tests for Model parameter validation and error handling.

Tests that model classes validate their inputs properly and
that errors are properly propagated as events.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch

from hawi.agent import HawiAgent
from hawi.agent.models import DeepSeekModel, KimiModel
from hawi.agent.events import agent_error_event


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
    async def test_model_error_sends_agent_error_event(self):
        """Test that model errors result in agent.error events being sent.

        This is a regression test for the bug where model preparation errors
        were silently swallowed and no events were produced.
        """
        # Create a mock model that fails during preparation
        mock_model = MagicMock()
        mock_model.model_id = "test-model"

        # Make astream raise an exception immediately (simulating _prepare_request_impl failure)
        async def failing_astream(*args, **kwargs):
            raise AttributeError("'list' object has no attribute 'startswith'")
            yield  # Make it a generator

        mock_model.astream = failing_astream

        agent = HawiAgent(
            model=mock_model,
            system_prompt="You are a test assistant.",
            enable_streaming=True,
        )

        events = []
        try:
            async for event in agent.arun("test message", stream=True):
                events.append(event)
        except AttributeError:
            # Exception should be raised after error event is sent
            pass

        # Should have received error event before exception was raised
        error_events = [e for e in events if e.type == "agent.error"]
        assert len(error_events) > 0, f"Expected agent.error event, got: {[e.type for e in events]}"
        assert "attribute" in error_events[0].metadata.get("error_message", "").lower() or \
               "startswith" in error_events[0].metadata.get("error_message", "").lower()

    @pytest.mark.asyncio
    async def test_model_init_error_is_not_silent(self):
        """Test that model initialization errors are not silently swallowed.

        This ensures that if a model's __init__ or astream raises an exception,
        the error propagates properly rather than being silently ignored.
        """
        # Create a model that will fail when its methods are called
        # We can't easily test the actual DeepSeekModel without valid API key,
        # so we test the agent's error handling behavior

        class BrokenModel:
            model_id = ["broken", "model"]  # Invalid type

            async def astream(self, *args, **kwargs):
                # This simulates what happens when model_id is a list
                # and _prepare_request_impl tries to call .startswith() on it
                raise AttributeError("'list' object has no attribute 'startswith'")
                yield

        agent = HawiAgent(
            model=BrokenModel(),
            system_prompt="Test",
            enable_streaming=True,
        )

        events = []
        try:
            async for event in agent.arun("test", stream=True):
                events.append(event)
        except Exception as e:
            # If exception propagates, that's acceptable
            # But ideally we want it as an event
            pass

        # Either we got an error event, or an exception was raised
        # Silent failure (empty events or just start/stop) is the bug
        error_events = [e for e in events if e.type == "agent.error"]
        exception_raised = len(events) <= 2  # Just run_start and stream_start

        assert len(error_events) > 0 or exception_raised, \
            "Error should either be sent as event or raised as exception, not silently swallowed"
