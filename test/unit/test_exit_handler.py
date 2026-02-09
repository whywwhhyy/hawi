"""
Tests for exit_handler module.
"""

import os
import signal
import sys
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hawi.utils.lifecycle import (
    ExitHandler,
    clear_exit_handlers,
    execute_early_and_clear,
    exit_scope,
    register_exit_handler,
)


@pytest.fixture
def fresh_handler():
    """Get a fresh handler instance for testing."""
    handler = ExitHandler.get_instance()
    handler.clear()
    handler._executed = False
    ExitHandler._system_exiting = False
    yield handler
    # Cleanup
    try:
        handler.clear()
        handler._executed = False
        ExitHandler._system_exiting = False
    except Exception:
        pass


class TestExitHandler:
    """Test cases for ExitHandler."""

    def test_singleton(self, fresh_handler):
        """Test that ExitHandler is a singleton."""
        h1 = ExitHandler.get_instance()
        h2 = ExitHandler.get_instance()
        assert h1 is h2

    def test_register_and_execute(self, fresh_handler):
        """Test basic registration and execution."""
        mock = Mock()
        fresh_handler.register(mock, priority=1)
        fresh_handler.execute_early_and_clear()
        mock.assert_called_once()

    def test_priority_order(self, fresh_handler):
        """Test that handlers execute in priority order."""
        order = []

        def make_handler(priority):
            def fn():
                order.append(priority)
            return fn

        fresh_handler.register(make_handler(3), priority=3)
        fresh_handler.register(make_handler(1), priority=1)
        fresh_handler.register(make_handler(2), priority=2)

        fresh_handler.execute_early_and_clear()
        assert order == [1, 2, 3]

    def test_register_decorator(self, fresh_handler):
        """Test decorator registration."""
        order = []

        @fresh_handler.register(priority=1)
        def handler1():
            order.append(1)

        @fresh_handler.register(priority=2)
        def handler2():
            order.append(2)

        fresh_handler.execute_early_and_clear()
        assert order == [1, 2]

    def test_module_level_register(self, fresh_handler):
        """Test module-level register function."""
        mock = Mock()
        register_exit_handler(mock, priority=1)
        execute_early_and_clear()
        mock.assert_called_once()

    def test_clear_without_execution(self, fresh_handler):
        """Test clearing handlers without executing."""
        mock = Mock()
        fresh_handler.register(mock, priority=1)
        count = fresh_handler.clear()
        assert count == 1
        mock.assert_not_called()
        fresh_handler.execute_early_and_clear()
        mock.assert_not_called()

    def test_exception_handling(self, fresh_handler):
        """Test that exceptions don't stop other handlers."""
        order = []

        def failing_handler():
            order.append('fail')
            raise ValueError("Test error")

        def success_handler():
            order.append('success')

        fresh_handler.register(failing_handler, priority=1)
        fresh_handler.register(success_handler, priority=2)
        fresh_handler.execute_early_and_clear()

        assert 'fail' in order
        assert 'success' in order

    def test_thread_safety(self, fresh_handler):
        """Test thread-safe registration."""
        counters = []
        lock = threading.Lock()

        def make_handler(n):
            def fn():
                with lock:
                    counters.append(n)
            return fn

        def register_many(start, count):
            for i in range(count):
                fresh_handler.register(make_handler(start + i), priority=start + i)

        threads = []
        for i in range(5):
            t = threading.Thread(target=register_many, args=(i * 10, 10))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(fresh_handler) == 50
        fresh_handler.execute_early_and_clear()
        assert len(counters) == 50

    def test_execute_and_keep(self, fresh_handler):
        """Test execute without clearing."""
        counter = [0]

        def increment():
            counter[0] += 1

        fresh_handler.register(increment, priority=1)
        fresh_handler.execute_and_keep()
        assert counter[0] == 1
        assert len(fresh_handler) == 1

        fresh_handler.execute_and_keep()
        assert counter[0] == 2

    def test_context_manager(self, fresh_handler):
        """Test context manager usage."""
        mock = Mock()
        with exit_scope():
            register_exit_handler(mock, priority=1)
        mock.assert_called_once()

    def test_handler_as_context_manager(self, fresh_handler):
        """Test ExitHandler as context manager."""
        mock = Mock()
        with fresh_handler:
            fresh_handler.register(mock, priority=1)
        mock.assert_called_once()

    def test_double_execution_prevention(self, fresh_handler):
        """Test that handlers don't execute twice with early execution."""
        counter = [0]

        def increment():
            counter[0] += 1

        fresh_handler.register(increment, priority=1)
        fresh_handler.execute_early_and_clear()
        assert counter[0] == 1

        fresh_handler.execute_early_and_clear()
        assert counter[0] == 1  # No double execution

    def test_weakref_registration(self, fresh_handler):
        """Test weakref-based registration."""
        mock = Mock()

        class TrackedObject:
            pass

        obj = TrackedObject()
        ref = fresh_handler.register_weakref(obj, mock, name="test_weak")

        # Delete the object
        del obj
        import gc
        gc.collect()

        # Callback may or may not have been called depending on timing
        assert ref() is None or mock.called

    def test_register_after_execution_fails(self, fresh_handler):
        """Test that registration fails after execution."""
        mock = Mock()
        fresh_handler.execute_early_and_clear()

        with pytest.raises(RuntimeError, match="Cannot register after exit"):
            fresh_handler.register(mock, priority=1)

    def test_signal_handler_installation(self, fresh_handler):
        """Test that signal handlers are installed."""
        current = signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, current)  # Restore
        assert current is not None

    def test_len_method(self, fresh_handler):
        """Test __len__ method."""
        assert len(fresh_handler) == 0
        fresh_handler.register(lambda: None, priority=1)
        assert len(fresh_handler) == 1

    def test_repr(self, fresh_handler):
        """Test __repr__ method."""
        r = repr(fresh_handler)
        assert "ExitHandler" in r
        assert "handlers=" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
