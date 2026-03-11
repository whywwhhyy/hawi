"""Unit tests for event_bus_new.py"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from hawi.agent.events.event import Event
from hawi.agent.events.event_bus import EventBus


# =============================================================================
# Test Event Classes
# =============================================================================

class AgentTestEvent(Event):
    """Test event for agent events."""
    payload: dict = {}

    def __init__(self, **data):
        data.setdefault('type', 'agent.run_start')
        data.setdefault('source', 'agent')
        super().__init__(**data)


class ModelTestEvent(Event):
    """Test event for model events."""
    payload: dict = {}

    def __init__(self, **data):
        data.setdefault('type', 'model.stream_start')
        data.setdefault('source', 'model')
        super().__init__(**data)


class ToolTestEvent(Event):
    """Test event with custom type."""
    tool_name: str = "test_tool"
    payload: dict = {}

    def __init__(self, **data):
        data.setdefault('type', 'agent.tool_call')
        data.setdefault('source', 'agent')
        super().__init__(**data)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def event_bus():
    """Create a fresh EventBus for each test."""
    bus = EventBus()
    yield bus
    bus.close(wait=False)


@pytest.fixture
def agent_event():
    """Create a test agent event."""
    return AgentTestEvent(payload={"data": 123})


@pytest.fixture
def model_event():
    """Create a test model event."""
    return ModelTestEvent(payload={"data": 456})


# =============================================================================
# Basic Tests
# =============================================================================

class TestBasicFunctionality:
    """Test basic EventBus functionality."""

    def test_sync_subscriber(self, event_bus, agent_event):
        """Test synchronous subscriber receives events."""
        results = []

        def handler(event):
            results.append(event.type)

        event_bus.subscribe(handler)
        event_bus.publish(agent_event)
        time.sleep(0.05)  # Give worker thread time to process

        assert len(results) == 1
        assert results[0] == 'agent.run_start'

    def test_async_subscriber(self, event_bus, agent_event):
        """Test asynchronous subscriber receives events."""
        results = []

        async def handler(event):
            results.append(f"async_{event.type}")

        event_bus.subscribe(handler)
        event_bus.publish(agent_event)
        time.sleep(0.05)  # Give async handler time to complete

        assert len(results) == 1
        assert results[0] == 'async_agent.run_start'

    def test_multiple_subscribers(self, event_bus, agent_event):
        """Test multiple subscribers receive same event."""
        results = []

        def handler1(event):
            results.append("handler1")

        def handler2(event):
            results.append("handler2")

        async def handler3(event):
            results.append("handler3")

        event_bus.subscribe(handler1)
        event_bus.subscribe(handler2)
        event_bus.subscribe(handler3)

        event_bus.publish(agent_event)
        time.sleep(0.05)

        assert len(results) == 3
        assert "handler1" in results
        assert "handler2" in results
        assert "handler3" in results

    def test_subscriber_execution_order(self, event_bus, agent_event):
        """Test subscribers are executed in registration order."""
        results = []

        def handler1(event):
            results.append(1)

        def handler2(event):
            results.append(2)

        def handler3(event):
            results.append(3)

        event_bus.subscribe(handler1)
        event_bus.subscribe(handler2)
        event_bus.subscribe(handler3)

        event_bus.publish(agent_event)
        time.sleep(0.05)  # Give worker thread time to process

        assert results == [1, 2, 3]


# =============================================================================
# Event Type Filtering Tests
# =============================================================================

class TestEventTypeFiltering:
    """Test event type filtering functionality."""

    def test_filter_by_single_type(self, event_bus, agent_event, model_event):
        """Test subscriber only receives filtered event type."""
        results = []

        def handler(event):
            results.append(event.type)

        event_bus.subscribe(handler, event_types=['agent.run_start'])

        event_bus.publish(model_event)
        event_bus.publish(agent_event)
        time.sleep(0.05)  # Give worker thread time to process

        assert len(results) == 1
        assert results[0] == 'agent.run_start'

    def test_filter_by_multiple_types(self, event_bus, agent_event, model_event):
        """Test subscriber receives multiple filtered event types."""
        results = []

        def handler(event):
            results.append(event.type)

        event_bus.subscribe(
            handler,
            event_types=['agent.run_start', 'model.stream_start']
        )

        event_bus.publish(model_event)
        event_bus.publish(agent_event)
        time.sleep(0.05)  # Give worker thread time to process

        assert len(results) == 2
        assert 'agent.run_start' in results
        assert 'model.stream_start' in results

    def test_no_filter_receives_all(self, event_bus, agent_event, model_event):
        """Test subscriber without filter receives all events."""
        results = []

        def handler(event):
            results.append(event.type)

        event_bus.subscribe(handler)  # No event_types filter

        event_bus.publish(model_event)
        event_bus.publish(agent_event)
        time.sleep(0.05)  # Give worker thread time to process

        assert len(results) == 2


# =============================================================================
# Blocking Mode Tests
# =============================================================================

class TestBlockingMode:
    """Test blocking subscriber mode."""

    def test_blocking_subscriber_validation(self, event_bus):
        """Test subscribe_blocking rejects async handlers."""
        async def async_handler(event):
            pass

        with pytest.raises(ValueError, match="subscribe_blocking only supports synchronous"):
            event_bus.subscribe_blocking(async_handler)

    def test_blocking_subscriber_executes(self, event_bus, agent_event):
        """Test blocking subscriber is called."""
        results = []

        def blocking_handler(event):
            results.append('blocking')

        event_bus.subscribe_blocking(blocking_handler)
        event_bus.publish(agent_event)

        assert 'blocking' in results

    def test_blocking_subscriber_with_delay(self, event_bus, agent_event):
        """Test blocking subscriber delays publish return."""
        delay_time = 0.05
        results = []

        def slow_handler(event):
            time.sleep(delay_time)
            results.append('done')

        event_bus.subscribe_blocking(slow_handler)

        start = time.time()
        event_bus.publish(agent_event)
        elapsed = time.time() - start

        assert elapsed >= delay_time
        assert 'done' in results


# =============================================================================
# Unsubscribe Tests
# =============================================================================

class TestUnsubscribe:
    """Test unsubscribe functionality."""

    def test_unsubscribe_removes_handler(self, event_bus, agent_event):
        """Test unsubscribe removes handler."""
        results = []

        def handler(event):
            results.append(event.type)

        event_bus.subscribe(handler)
        event_bus.publish(agent_event)
        time.sleep(0.05)  # Give worker thread time to process
        assert len(results) == 1

        success = event_bus.unsubscribe(handler)
        assert success is True

        event_bus.publish(agent_event)
        time.sleep(0.05)  # Give worker thread time to process
        assert len(results) == 1  # No new events

    def test_unsubscribe_unknown_handler(self, event_bus):
        """Test unsubscribe returns False for unknown handler."""
        def handler(event):
            pass

        success = event_bus.unsubscribe(handler)
        assert success is False

    def test_unsubscribe_after_close(self, event_bus):
        """Test unsubscribe after close returns False."""
        def handler(event):
            pass

        event_bus.close(wait=False)

        success = event_bus.unsubscribe(handler)
        assert success is False


# =============================================================================
# Publish Mode Tests
# =============================================================================

class TestPublishModes:
    """Test different publish modes."""

    def test_non_blocking_publish(self, event_bus, agent_event):
        """Test non-blocking publish returns immediately."""
        delay_time = 0.1
        results = []

        def slow_handler(event):
            time.sleep(delay_time)
            results.append('done')

        event_bus.subscribe(slow_handler)

        start = time.time()
        event_bus.publish(agent_event)  # Non-blocking
        elapsed = time.time() - start

        # Should return immediately, before handler completes
        assert elapsed < delay_time / 2
        assert len(results) == 0  # Handler hasn't run yet

        time.sleep(delay_time + 0.02)
        assert len(results) == 1

    def test_blocking_subscriber_in_publish_thread(self, event_bus, agent_event):
        """Test blocking subscriber executes synchronously in publish thread."""
        delay_time = 0.05
        results = []

        def slow_blocking_handler(event):
            time.sleep(delay_time)
            results.append('done')

        # Use subscribe_blocking to execute synchronously in publish thread
        event_bus.subscribe_blocking(slow_blocking_handler)

        start = time.time()
        event_bus.publish(agent_event)
        elapsed = time.time() - start

        # Should wait for blocking handler
        assert elapsed >= delay_time
        assert len(results) == 1

# =============================================================================
# Async Publish Tests
# =============================================================================

class TestAsyncPublish:
    """Test publish_async functionality."""

    @pytest.mark.asyncio
    async def test_publish_async(self, event_bus, agent_event):
        """Test async publish works correctly."""
        results = []

        def handler(event):
            results.append(event.type)

        event_bus.subscribe(handler)
        await event_bus.publish_async(agent_event)

        assert len(results) == 1
        assert results[0] == 'agent.run_start'

    @pytest.mark.asyncio
    async def test_publish_async_with_async_handler(self, event_bus, agent_event):
        """Test publish_async with async handler."""
        results = []

        async def handler(event):
            await asyncio.sleep(0.01)
            results.append(f"async_{event.type}")

        event_bus.subscribe(handler)
        await event_bus.publish_async(agent_event)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_publish_async_concurrent(self, event_bus):
        """Test multiple concurrent async publishes."""
        results = []
        lock = threading.Lock()

        def handler(event):
            with lock:
                results.append(event.payload['id'])

        event_bus.subscribe(handler)

        events = [
            AgentTestEvent(payload={'id': i})
            for i in range(10)
        ]

        await asyncio.gather(*[
            event_bus.publish_async(event)
            for event in events
        ])

        assert len(results) == 10
        assert set(results) == set(range(10))


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Test thread safety of EventBus."""

    def test_concurrent_subscribe(self, event_bus, agent_event):
        """Test concurrent subscriptions are safe."""
        results = []
        lock = threading.Lock()

        def handler_factory(idx):
            def handler(event):
                with lock:
                    results.append(idx)
            return handler

        handlers = [handler_factory(i) for i in range(20)]

        threads = [
            threading.Thread(target=lambda h=h: event_bus.subscribe(h))
            for h in handlers
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        event_bus.publish(agent_event)
        time.sleep(0.1)  # Give worker thread time to process

        assert len(results) == 20

    def test_concurrent_subscribe_and_publish(self, event_bus):
        """Test concurrent subscribe and publish."""
        results = []
        errors = []
        lock = threading.Lock()

        def handler(event):
            with lock:
                results.append(event.payload.get('id'))

        def subscribe_loop():
            try:
                for i in range(50):
                    event_bus.subscribe(handler)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(str(e))

        def publish_loop():
            try:
                for i in range(50):
                    event = AgentTestEvent(payload={'id': i})
                    event_bus.publish(event)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=subscribe_loop),
            threading.Thread(target=publish_loop),
            threading.Thread(target=subscribe_loop),
            threading.Thread(target=publish_loop),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        time.sleep(0.2)  # Let remaining events process

        assert len(errors) == 0
        # Results may vary due to race, but no crash


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling in EventBus."""

    def test_handler_exception_does_not_break_others(self, event_bus, agent_event):
        """Test exception in one handler doesn't break others."""
        results = []

        def bad_handler(event):
            raise ValueError("Intentional error")

        def good_handler(event):
            results.append('good')

        event_bus.subscribe(bad_handler)
        event_bus.subscribe(good_handler)

        event_bus.publish(agent_event)
        time.sleep(0.05)  # Give worker thread time to process

        assert len(results) == 1
        assert results[0] == 'good'

    def test_async_handler_exception(self, event_bus, agent_event):
        """Test exception in async handler doesn't break others."""
        results = []

        async def bad_handler(event):
            raise ValueError("Async error")

        def good_handler(event):
            results.append('good')

        event_bus.subscribe(bad_handler)
        event_bus.subscribe(good_handler)

        event_bus.publish(agent_event)
        time.sleep(0.05)

        assert len(results) == 1

    def test_publish_to_closed_bus(self, event_bus, agent_event):
        """Test publishing to closed bus raises error."""
        event_bus.close(wait=False)

        with pytest.raises(RuntimeError, match="EventBus is closed"):
            event_bus.publish(agent_event)

    def test_subscribe_to_closed_bus(self, event_bus):
        """Test subscribing to closed bus raises error."""
        event_bus.close(wait=False)

        def handler(event):
            pass

        with pytest.raises(RuntimeError, match="EventBus is closed"):
            event_bus.subscribe(handler)


# =============================================================================
# Lifecycle Tests
# =============================================================================

class TestLifecycle:
    """Test EventBus lifecycle."""

    def test_context_manager_sync(self, agent_event):
        """Test synchronous context manager."""
        results = []

        def handler(event):
            results.append(event.type)

        with EventBus() as bus:
            bus.subscribe(handler)
            bus.publish(agent_event)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_context_manager_async(self, agent_event):
        """Test asynchronous context manager."""
        results = []

        def handler(event):
            results.append(event.type)

        async with EventBus() as bus:
            bus.subscribe(handler)
            await bus.publish_async(agent_event)

        assert len(results) == 1

    def test_close_idempotent(self, event_bus):
        """Test close can be called multiple times."""
        event_bus.close()
        event_bus.close()  # Should not raise
        event_bus.close(wait=False)  # Should not raise

    def test_flush_timeout(self, event_bus):
        """Test flush with timeout."""
        def slow_handler(event):
            time.sleep(0.1)

        event_bus.subscribe(slow_handler)

        # Send multiple events to ensure queue stays non-empty
        for _ in range(100):
            event_bus.publish(AgentTestEvent())  # Non-blocking

        result = event_bus.flush(timeout=0.01)  # Short timeout
        assert result is False  # Timeout

    def test_flush_success(self, event_bus):
        """Test flush completes successfully."""
        def handler(event):
            pass

        event_bus.subscribe(handler)
        event_bus.publish(AgentTestEvent())

        result = event_bus.flush(timeout=1.0)
        assert result is True


# =============================================================================
# Complex Scenario Tests
# =============================================================================

class TestComplexScenarios:
    """Test complex usage scenarios."""

    def test_mixed_handlers_all_receive(self, event_bus, agent_event):
        """Test mix of sync and async handlers all receive events."""
        results = []
        lock = threading.Lock()

        def sync_handler1(event):
            with lock:
                results.append('sync1')

        async def async_handler1(event):
            await asyncio.sleep(0.001)
            with lock:
                results.append('async1')

        def sync_handler2(event):
            with lock:
                results.append('sync2')

        async def async_handler2(event):
            with lock:
                results.append('async2')

        event_bus.subscribe(sync_handler1)
        event_bus.subscribe(async_handler1)
        event_bus.subscribe(sync_handler2)
        event_bus.subscribe(async_handler2)

        event_bus.publish(agent_event)
        time.sleep(0.05)

        assert len(results) == 4
        assert set(results) == {'sync1', 'async1', 'sync2', 'async2'}

    def test_event_chain(self, event_bus):
        """Test handler publishing new events."""
        results = []
        lock = threading.Lock()

        def chain_handler(event):
            with lock:
                results.append(f"chain_{event.payload.get('depth', 0)}")

            depth = event.payload.get('depth', 0)
            if depth < 3:
                new_event = AgentTestEvent(payload={'depth': depth + 1})
                event_bus.publish(new_event)  # Non-blocking

        event_bus.subscribe(chain_handler)

        initial = AgentTestEvent(payload={'depth': 0})
        event_bus.publish(initial)
        time.sleep(0.1)

        assert len(results) >= 4  # depth 0, 1, 2, 3

    def test_high_volume(self, event_bus):
        """Test high volume of events."""
        results = []
        lock = threading.Lock()
        expected_count = 100

        def handler(event):
            with lock:
                results.append(event.payload['id'])

        event_bus.subscribe(handler)

        for i in range(expected_count):
            event = AgentTestEvent(payload={'id': i})
            event_bus.publish(event)

        # Wait for all to complete
        time.sleep(0.5)

        assert len(results) == expected_count
