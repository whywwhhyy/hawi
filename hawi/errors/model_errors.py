"""Model related errors."""

from __future__ import annotations

from typing import Optional, cast

from .error import HawiError, ModelErrorType


class ModelError(HawiError):
    """模型调用错误基类"""
    def __init__(self, error_type: ModelErrorType, msg: Optional[str]):
        super().__init__(error_type, msg)


    @property
    def error_type(self) -> ModelErrorType:
        return cast(ModelErrorType, self._error_type)

class NetworkError(ModelError):
    """网络错误（连接失败、超时、DNS 解析失败等）"""
    def __init__(self, msg: Optional[str] = None):
        super().__init__('network', msg or "Network error occurred")


class RemoteError(ModelError):
    """远程服务错误（服务器内部错误、服务不可用等）
    
    当远程 API 服务器返回 5xx 错误、内部错误或
    服务临时不可用时抛出。此类错误通常可以重试。
    """
    def __init__(self, msg: Optional[str] = None):
        super().__init__('remote', msg or "Remote service error occurred")


class ThrottleError(ModelError):
    """限流错误（429等）"""
    def __init__(self, msg: Optional[str] = None):
        super().__init__('throttle', msg or "Rate limit exceeded")


class DeniedError(ModelError):
    """权限错误（认证失败、禁止访问等）"""
    def __init__(self, msg: Optional[str] = None):
        super().__init__('access', msg or "Access denied")


class ValidationError(ModelError):
    """验证错误（数据格式出错等）"""
    def __init__(self, msg: Optional[str] = None):
        super().__init__('validation', msg or "Validation failed")


class UnknownModelError(ModelError):
    """未知模型错误"""
    def __init__(self, msg: Optional[str] = None):
        super().__init__('unknown', msg or "An unknown model error occurred")
