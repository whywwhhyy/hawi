"""Agent related errors."""

from __future__ import annotations

from typing import Any, Optional

from .error import HawiError, AgentErrorType


class AgentError(HawiError):
    """Agent 执行错误基类"""
    def __init__(self, error_type: AgentErrorType, msg: Optional[str]):
        super().__init__(error_type, msg)


class MaxIterationsError(AgentError):
    """达到最大迭代次数"""
    def __init__(self, msg: Optional[str] = None):
        super().__init__('max_iteration', msg or "Maximum iterations reached")


class ToolNotFoundError(AgentError):
    """工具未找到"""
    def __init__(self, msg: Optional[str] = None):
        super().__init__('tool_not_found', msg or "Tool not found")


class ToolValidationError(AgentError):
    """工具参数验证错误"""
    def __init__(self, msg: Optional[str] = None):
        super().__init__('tool_validation', msg or "Tool validation failed")


class ToolExecutionError(AgentError):
    """工具执行过程错误"""

    def __init__(self, msg: Optional[str] = None, details: dict[str, Any] | None = None):
        super().__init__('tool_execution', msg or "Tool execution failed")
        self.details = details
