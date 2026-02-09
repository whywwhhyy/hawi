"""
Exit handler with multi-layered guarantees for cleanup function execution.

Uses multiple mechanisms to ensure registered functions are called on exit:
1. atexit module (standard Python exit handling)
2. signal handlers (SIGTERM, SIGINT for graceful shutdown)
3. sys.excepthook (uncaught exceptions)
4. context manager protocol (for explicit scope-based cleanup)
5. weakref.finalize (for object-based cleanup as fallback)

All operations are thread-safe using atomic operations where possible.
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import FrameType, TracebackType
from typing import Any, Callable

# Type aliases
ExitFunc = Callable[[], Any]


@dataclass(order=True, frozen=True)
class _HandlerEntry:
    """Immutable entry for priority queue. Lower priority = executed first."""
    priority: int
    seq: int  # Sequence number for stable sorting
    func: ExitFunc = field(compare=False)
    name: str | None = field(compare=False, default=None)

    def __call__(self) -> Any:
        return self.func()


class ExitHandler:
    """
    Global singleton exit handler with multi-layered execution guarantees.

    Features:
    - Thread-safe registration and execution
    - Priority-based execution order (lower = earlier)
    - Signal handling for graceful shutdown
    - Early execution with atomic clearing
    - Weakref-based fallback for object cleanup

    Example:
        >>> handler = ExitHandler.get_instance()
        >>> handler.register(lambda: print("Cleaning up..."))
        >>> handler.register_at_exit(cleanup_func, priority=1)

        >>> # Early execution
        >>> handler.execute_early_and_clear()
    """

    _instance: ExitHandler | None = None
    _instance_lock: threading.Lock = threading.Lock()
    _seq_counter: int = 0
    _seq_lock: threading.Lock = threading.Lock()
    _system_exiting: bool = False

    # Instance attributes (for type checking)
    _handlers: list[_HandlerEntry] = None  # type: ignore
    _handlers_lock: threading.RLock = None  # type: ignore
    _executed: bool = False
    _execute_lock: threading.Lock = None  # type: ignore
    _initialized: bool = False
    _original_sigterm: Any = None
    _original_sigint: Any = None
    _original_excepthook: Any = None
    _atexit_registered: bool = False

    def __new__(cls) -> ExitHandler:
        raise RuntimeError("Use ExitHandler.get_instance() to get the singleton instance")

    @classmethod
    def get_instance(cls) -> ExitHandler:
        """Get the global singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls._create_instance()
        return cls._instance

    @classmethod
    def _create_instance(cls) -> ExitHandler:
        """Create and initialize the singleton instance."""
        instance = object.__new__(cls)
        instance._initialized = False
        instance._handlers = []
        instance._handlers_lock = threading.RLock()
        instance._executed = False
        instance._execute_lock = threading.Lock()
        instance._original_sigterm = None
        instance._original_sigint = None
        instance._original_excepthook = None
        instance._atexit_registered = False
        return instance

    def _initialize(self) -> None:
        """Initialize all exit mechanisms."""
        if self._initialized:
            return

        with self._instance_lock:
            if self._initialized:
                return

            # 1. Register with atexit
            if not self._atexit_registered:
                atexit.register(self._atexit_callback)
                self._atexit_registered = True

            # 2. Install signal handlers
            self._install_signal_handlers()

            # 3. Install excepthook
            self._install_excepthook()

            self._initialized = True

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown."""
        try:
            self._original_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        except (ValueError, OSError):
            pass

        try:
            self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        except (ValueError, OSError):
            pass

        try:
            if hasattr(signal, 'SIGABRT'):
                signal.signal(signal.SIGABRT, self._signal_handler)
        except (ValueError, OSError):
            pass

    def _install_excepthook(self) -> None:
        """Install sys.excepthook for uncaught exceptions."""
        self._original_excepthook = sys.excepthook
        sys.excepthook = self._excepthook_wrapper

    def _excepthook_wrapper(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        tb: TracebackType | None,
    ) -> None:
        """Wrapper to execute handlers on uncaught exception."""
        try:
            self._execute_all("exception")
        finally:
            if self._original_excepthook:
                self._original_excepthook(exc_type, exc_value, tb)

    def _signal_handler(self, signum: int, frame: FrameType | None) -> None:
        """Signal handler that executes cleanup and chains to original handler."""
        try:
            self._execute_all(f"signal_{signum}")
        finally:
            self._restore_signal_handlers()
            # Re-raise the signal using default handler
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        try:
            if self._original_sigterm is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm)
        except (ValueError, OSError):
            pass

        try:
            if self._original_sigint is not None:
                signal.signal(signal.SIGINT, self._original_sigint)
        except (ValueError, OSError):
            pass

    def _get_next_seq(self) -> int:
        """Get next sequence number atomically."""
        with self._seq_lock:
            ExitHandler._seq_counter += 1
            return ExitHandler._seq_counter

    def register(
        self,
        func: ExitFunc | None = None,
        *,
        priority: int = 100,
        name: str | None = None,
    ) -> ExitFunc | Callable[[ExitFunc], ExitFunc]:
        """
        Register a function to be called at exit.

        Can be used as a decorator or direct call.

        Args:
            func: Function to register
            priority: Lower values execute first (default 100)
            name: Optional name for debugging

        Returns:
            The registered function (for decorator use)

        Example:
            >>> handler = ExitHandler.get_instance()
            >>> @handler.register(priority=1)
            ... def cleanup(): pass
            >>>
            >>> # Or direct registration
            >>> handler.register(lambda: print("done"), priority=2)
        """
        if func is None:
            # Being used as decorator with arguments
            def decorator(f: ExitFunc) -> ExitFunc:
                self._register_impl(f, priority, name or f.__name__)
                return f
            return decorator

        # Direct registration
        self._register_impl(func, priority, name or getattr(func, '__name__', None))
        return func

    def _register_impl(
        self,
        func: ExitFunc,
        priority: int,
        name: str | None,
    ) -> None:
        """Internal registration implementation."""
        if not self._initialized:
            self._initialize()

        if self._executed or ExitHandler._system_exiting:
            raise RuntimeError("Cannot register after exit handling has begun")

        if not callable(func):
            raise TypeError(f"Expected callable, got {type(func)}")

        entry = _HandlerEntry(
            priority=priority,
            seq=self._get_next_seq(),
            func=func,
            name=name,
        )

        with self._handlers_lock:
            self._handlers.append(entry)
            # Keep sorted by priority
            self._handlers.sort(key=lambda e: (e.priority, e.seq))

    def register_at_exit(
        self,
        func: ExitFunc,
        priority: int = 100,
        name: str | None = None,
    ) -> None:
        """
        Register a function specifically for atexit (non-decorator).

        This is an alias for register() for clarity.
        """
        self._register_impl(func, priority, name)

    def register_weakref(
        self,
        obj: Any,
        func: ExitFunc,
        name: str | None = None,
    ) -> weakref.ref[Any]:
        """
        Register a function tied to an object's lifetime.

        The function will be called either at exit OR when the object is garbage collected.

        Args:
            obj: Object to track
            func: Function to call when object dies or at exit
            name: Optional name for debugging

        Returns:
            weakref to the tracked object
        """
        if not self._initialized:
            self._initialize()

        def callback(ref: weakref.ref[Any]) -> None:
            if not self._executed:
                try:
                    func()
                except Exception:
                    pass

        ref = weakref.ref(obj, callback)

        # Also register for atexit in case GC doesn't happen
        self._register_impl(func, priority=999, name=name or f"weakref_{id(obj)}")

        return ref

    def execute_early_and_clear(self) -> list[Any]:
        """
        Execute all registered handlers immediately and atomically clear them.

        This prevents any duplicate execution at actual exit.

        Returns:
            List of return values from executed handlers
        """
        return self._execute_all("early")

    def execute_and_keep(self) -> list[Any]:
        """
        Execute all handlers but keep them registered for exit.

        Returns:
            List of return values from executed handlers
        """
        return self._execute_all("keep")

    def clear(self) -> int:
        """
        Clear all registered handlers without executing.

        Returns:
            Number of handlers cleared
        """
        with self._handlers_lock:
            count = len(self._handlers)
            self._handlers.clear()
            return count

    def _execute_all(self, mode: str) -> list[Any]:
        """
        Execute all handlers with specified mode.

        Args:
            mode: 'early' (execute and clear), 'keep' (execute and keep),
                  'exception', 'signal_*' (for specific triggers)

        Returns:
            List of return values
        """
        with self._execute_lock:
            if self._executed and mode != "keep":
                return []

            if mode == "early":
                self._executed = True
                ExitHandler._system_exiting = True

            # Get handlers atomically
            with self._handlers_lock:
                if mode == "early":
                    handlers = self._handlers.copy()
                    self._handlers.clear()
                else:
                    handlers = self._handlers.copy()

            results = []

            for entry in handlers:
                try:
                    result = entry()
                    results.append(result)
                except Exception:
                    # Continue executing other handlers even if one fails
                    pass

            # In early mode, also unregister from atexit
            if mode == "early" and self._atexit_registered:
                try:
                    atexit.unregister(self._atexit_callback)
                except Exception:
                    pass
                self._atexit_registered = False

            return results

    def _atexit_callback(self) -> None:
        """Callback registered with atexit module."""
        ExitHandler._system_exiting = True
        self._execute_all("atexit")

    def __enter__(self) -> ExitHandler:
        """Context manager entry."""
        if not self._initialized:
            self._initialize()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit - executes and clears."""
        self.execute_early_and_clear()

    def __len__(self) -> int:
        """Return number of registered handlers."""
        with self._handlers_lock:
            return len(self._handlers)

    def __repr__(self) -> str:
        with self._handlers_lock:
            return f"<ExitHandler handlers={len(self._handlers)} executed={self._executed}>"


# Module-level convenience functions
def register_exit_handler(
    func: ExitFunc | None = None,
    *,
    priority: int = 100,
    name: str | None = None,
) -> ExitFunc | Callable[[ExitFunc], ExitFunc]:
    """
    Convenience function to register with global ExitHandler.

    Example:
        >>> @register_exit_handler(priority=1)
        ... def cleanup(): pass
        >>>
        >>> register_exit_handler(lambda: print("done"), priority=2)
    """
    handler = ExitHandler.get_instance()
    return handler.register(func, priority=priority, name=name)


def execute_early_and_clear() -> list[Any]:
    """Execute all handlers early and clear them."""
    return ExitHandler.get_instance().execute_early_and_clear()


def clear_exit_handlers() -> int:
    """Clear all exit handlers without executing."""
    return ExitHandler.get_instance().clear()


@contextmanager
def exit_scope():
    """
    Context manager for scope-based cleanup.

    Example:
        >>> with exit_scope():
        ...     # do work
        ...     pass  # cleanup happens here
    """
    handler = ExitHandler.get_instance()
    try:
        yield handler
    finally:
        handler.execute_early_and_clear()


# Ensure fork safety - reinitialize in child processes after fork
try:
    if hasattr(os, 'register_at_fork'):
        def _atfork_reinit():
            """Reinitialize exit handler in child process after fork."""
            ExitHandler._instance = None
            ExitHandler._system_exiting = False

        os.register_at_fork(after_in_child=_atfork_reinit)
except (AttributeError, OSError):
    pass
