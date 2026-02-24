"""Hawi Agent 异常体系

提供结构化的异常类，替代字符串 error_type，保留完整的调用栈信息。
"""

from __future__ import annotations

import traceback
from typing import Any


class HawiError(Exception):
    """Hawi 框架基础异常"""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        # 自动捕获当前调用栈（排除当前这一层）
        self.stack_trace = "".join(traceback.format_stack()[:-1])

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"


class AgentError(HawiError):
    """Agent 执行错误基类"""

    recoverable: bool = False  # 默认不可恢复


class MaxIterationsError(AgentError):
    """达到最大迭代次数"""

    recoverable = False


# =============================================================================
# 模型相关错误 - 替代 ModelErrorType
# =============================================================================


class ModelError(AgentError):
    """模型调用错误基类"""

    recoverable = True  # 网络/限流错误默认可重试

    @classmethod
    def classify(cls, exception: Exception) -> ModelError:
        """根据异常创建对应的 ModelError 子类实例

        Args:
            exception: 原始异常

        Returns:
            具体的 ModelError 子类实例
        """
        error_str = str(exception).lower()

        # 限流错误
        if any(
            kw in error_str
            for kw in ["rate limit", "429", "too many requests", "throttle"]
        ):
            return ThrottleError(str(exception), details={"original": exception})

        # 权限错误
        if any(
            kw in error_str
            for kw in ["unauthorized", "forbidden", "401", "403", "denied", "api key"]
        ):
            return DeniedError(str(exception), details={"original": exception})

        # 网络错误
        if any(
            kw in error_str
            for kw in ["connection", "timeout", "network", "dns", "refused", "reset"]
        ):
            return NetworkError(str(exception), details={"original": exception})

        # 未知错误
        return UnknownModelError(str(exception), details={"original": exception})


class NetworkError(ModelError):
    """网络错误（连接失败、超时等）"""

    pass


class ThrottleError(ModelError):
    """限流错误（429等）"""

    pass


class DeniedError(ModelError):
    """权限错误（认证失败、禁止访问等）"""

    recoverable = False  # 认证错误重试也没用


class UnknownModelError(ModelError):
    """未知模型错误"""

    recoverable = False  # 未知错误默认不重试


# =============================================================================
# 工具相关错误
# =============================================================================


class ToolError(AgentError):
    """工具执行错误 - 可恢复，让模型知道并尝试修复"""

    recoverable = True


class ToolNotFoundError(ToolError):
    """工具未找到"""

    pass


class ToolValidationError(ToolError):
    """工具参数验证错误"""

    pass


class ToolExecutionError(ToolError):
    """工具执行过程错误"""

    pass


# =============================================================================
# 配置相关错误
# =============================================================================


class ConfigurationError(HawiError):
    """配置错误"""

    recoverable = False


# =============================================================================
# 工具函数
# =============================================================================


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
