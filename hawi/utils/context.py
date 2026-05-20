"""Global context management with reference counting."""

import uuid
import threading
from contextvars import ContextVar
from typing import Any, Dict, Optional
from contextlib import contextmanager


class ContextManager:
    """线程/协程安全的全局 Context 管理器。

    使用 context_id + 全局存储的代理模式，支持：
    - 跨协程/线程共享 Context（引用计数）
    - 跨作用域传递（fork 创建子 Context）
    - 自动生命周期管理（引用计数归零时清理）
    """

    _contexts: Dict[str, Dict[str, Any]] = {}
    _ref_count: Dict[str, int] = {}
    _lock: threading.Lock = threading.Lock()
    _context_id: ContextVar[Optional[str]] = ContextVar('context_id', default=None)

    @classmethod
    def create(cls, initial: Optional[Dict[str, Any]] = None) -> str:
        """创建新的 Context，返回 context_id。"""
        ctx_id = str(uuid.uuid4())[:8]
        with cls._lock:
            cls._contexts[ctx_id] = initial or {}
            cls._ref_count[ctx_id] = 1
        cls._context_id.set(ctx_id)
        return ctx_id

    @classmethod
    def attach(cls, ctx_id: str) -> None:
        """将当前上下文附加到指定 context_id（引用计数+1）。"""
        with cls._lock:
            if ctx_id in cls._ref_count:
                cls._ref_count[ctx_id] += 1
        cls._context_id.set(ctx_id)

    @classmethod
    def detach(cls, ctx_id: str) -> None:
        """分离指定 context_id（引用计数-1，可能自动清理）。"""
        should_delete = False
        with cls._lock:
            if ctx_id in cls._ref_count:
                cls._ref_count[ctx_id] -= 1
                if cls._ref_count[ctx_id] <= 0:
                    should_delete = True
                    del cls._ref_count[ctx_id]
                    del cls._contexts[ctx_id]
        if should_delete and cls._context_id.get() == ctx_id:
            cls._context_id.set(None)

    @classmethod
    def fork(cls, parent_id: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """从父 Context 创建子 Context（拷贝数据，独立引用计数）。"""
        with cls._lock:
            parent_data = cls._contexts.get(parent_id, {}).copy()
        if extra:
            parent_data.update(extra)
        return cls.create(parent_data)

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """获取当前 Context 中的值。"""
        ctx_id = cls._context_id.get()
        if not ctx_id:
            return default
        with cls._lock:
            return cls._contexts.get(ctx_id, {}).get(key, default)

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """设置当前 Context 中的值。"""
        ctx_id = cls._context_id.get()
        if not ctx_id:
            raise RuntimeError("No active context")
        with cls._lock:
            if ctx_id in cls._contexts:
                cls._contexts[ctx_id][key] = value

    @classmethod
    def current_id(cls) -> Optional[str]:
        """获取当前 context_id。"""
        return cls._context_id.get()

    @classmethod
    def clear(cls) -> None:
        """清理所有 Context（仅用于测试）。"""
        with cls._lock:
            cls._contexts.clear()
            cls._ref_count.clear()


@contextmanager
def context_scope(initial: Optional[Dict[str, Any]] = None):
    """Context 作用域上下文管理器。"""
    ctx_id = ContextManager.create(initial)
    try:
        yield ctx_id
    finally:
        ContextManager.detach(ctx_id)
