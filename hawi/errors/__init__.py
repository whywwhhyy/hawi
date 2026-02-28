"""Hawi Agent 异常体系

提供结构化的异常类，保留完整的调用栈信息。
"""

from .error import (
    HawiError,
    ErrorType,
    AgentErrorType,
    ModelErrorType,
    ConfigurationError,
    get_error_stack,
)

from .model_errors import (
    ModelError,
    NetworkError,
    ThrottleError,
    DeniedError,
    ValidationError,
    UnknownModelError,
)

from .agent_errors import (
    AgentError,
    MaxIterationsError,
    ToolNotFoundError,
    ToolValidationError,
    ToolExecutionError,
)

__all__ = [
    # Base
    "HawiError",
    "ErrorType",
    "AgentErrorType",
    "ModelErrorType",
    "ConfigurationError",
    "get_error_stack",
    # Model
    "ModelError",
    "NetworkError",
    "ThrottleError",
    "DeniedError",
    "ValidationError",
    "UnknownModelError",
    # Agent
    "AgentError",
    "MaxIterationsError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolExecutionError",
]
