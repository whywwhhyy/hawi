"""Hawi Error base class and utilities.

提供结构化的异常类，保留完整的调用栈信息。
"""

from __future__ import annotations

import traceback
from typing import Literal, Optional

AgentErrorType = Literal[
    'max_iteration',
    'tool_not_found',
    'tool_validation',
    'tool_execution',
    'unknown',
]

ModelErrorType = Literal[
    'network',
    'remote',
    'throttle',
    'access',
    'validation',
    'unknown',
]

ErrorType = AgentErrorType | ModelErrorType | Literal['configuration', 'unknown']


class HawiError(Exception):
    """Hawi 框架基础异常"""

    _error_type: ErrorType

    def __init__(self, error_type: ErrorType, message: Optional[str]):
        super().__init__(message)
        self._error_type = error_type
        self.message = message
        # 自动捕获当前调用栈（排除当前这一层）
        self.stack_trace = "".join(traceback.format_stack()[:-1])

    @property
    def error_type(self) -> ErrorType:
        return self._error_type

    def __str__(self) -> str:
        if self.message is None:
            return type(self).__name__
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"


class ConfigurationError(HawiError):
    """配置错误"""

    def __init__(self, msg: Optional[str] = None):
        super().__init__('configuration', msg or "Configuration error")

    recoverable = False

    @property
    def error_type(self) -> ErrorType:
        return 'configuration'


def get_error_stack(error: Exception) -> str:
    """获取异常的完整调用栈

    Args:
        error: 异常对象

    Returns:
        格式化的调用栈字符串
    """
    if isinstance(error, HawiError):
        return error.stack_trace
    return traceback.format_exc()
