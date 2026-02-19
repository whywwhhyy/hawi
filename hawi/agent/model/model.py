"""
Hawi Agent Model 基类

提供统一的 Model 抽象，支持同步和异步操作，兼容多种 LLM 提供商。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Iterator, List, Literal

from hawi.agent.message import (
    ContentPart,
    Message,
    MessageRequest,
    MessageResponse,
    StreamPart,
    TextPart,
    ToolCallPart,
    ToolDefinition,
    ToolChoice,
)

__all__ = ["Model", "StreamPart", "BalanceInfo", "ModelErrorType", "ModelFailurePolicy", "ProviderRequest", "ProviderResponse", "ModelParams", "BalanceDetails"]

# 类型别名：提供商特定的请求/响应格式
# 这些类型是 Any 因为不同 LLM 提供商的 API 格式差异很大
ProviderRequest = dict[str, Any]
"""提供商特定的请求数据（如 OpenAI、Anthropic、DeepSeek 等各自的 API 格式）"""

ProviderResponse = dict[str, Any]
"""提供商特定的响应数据"""

ModelParams = dict[str, Any]
"""模型参数（temperature、max_tokens 等，各提供商支持不同）"""

# 余额详情类型：各提供商返回的余额信息格式不同
BalanceDetails = dict[str, Any]
"""余额详情，包含各平台特定的额外信息（如赠送余额、冻结余额等）"""


class ModelErrorType:
    """模型错误类型分类"""

    NETWORK = "network"      # 网络错误（连接失败、超时等）
    THROTTLE = "throttle"    # 限流错误（429等）
    DENIED = "denied"        # 权限错误（认证失败、禁止访问等）
    UNKNOWN = "unknown"      # 未知错误


@dataclass
class ModelFailurePolicy:
    """模型失败处理策略

    Attributes:
        error_type: 错误类型
        action: 处理方式 ("retry" 或 "stop")
        retry_count: 重试次数（仅当 action="retry" 时有效）
    """

    error_type: str
    action: Literal["retry", "stop"] = "stop"
    retry_count: int = 0


@dataclass
class BalanceInfo:
    """账户余额信息

    统一各 LLM 提供商的余额查询结果，支持多币种和不同余额类型。

    Attributes:
        currency: 货币代码，如 "CNY", "USD"
        available_balance: 可用余额（实际可使用的金额）
        total_balance: 总余额（包含所有类型的余额）
        is_available: 账户是否可用（余额是否充足）
        details: 各平台特定的额外余额详情
    """

    currency: str
    available_balance: float
    total_balance: float | None = None
    is_available: bool = True
    details: BalanceDetails = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"BalanceInfo({self.currency}: "
            f"available={self.available_balance:.4f})"
        )





class Model(ABC):
    """
    Model 抽象基类

    统一 LLM 提供商的接口，支持同步和异步操作。

    子类必须实现：
    - __init__: 初始化模型特定参数
    - _invoke_impl(): 同步调用实现
    - _prepare_request_impl(): 请求格式转换
    - _parse_response_impl(): 响应格式转换

    可选实现：
    - _ainvoke_impl(): 异步调用实现（默认使用 sync 版本）
    - _stream_impl(): 同步流式实现
    - _astream_impl(): 异步流式实现

    Example:
        # 同步调用
        model = OpenAIModel(model_id="gpt-4", api_key="...")
        response = model.invoke(messages=[create_user_message("Hello")])

        # 异步调用
        response = await model.ainvoke(messages=[create_user_message("Hello")])

        # 流式调用
        for event in model.stream(messages=[create_user_message("Hello")]):
            if event.type == "content":
                print(event.content)
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """模型标识符"""
        pass

    # ==========================================================================
    # 公共 API - 同步方法
    # ==========================================================================

    def invoke(
        self,
        messages: list[Message],
        system: str | List[ContentPart] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs,
    ) -> MessageResponse:
        """同步调用模型"""
        request = self._build_request(messages, system, tools, tool_choice, kwargs)
        return self._invoke_impl(request)

    def stream(
        self,
        messages: list[Message],
        system: str | List[ContentPart] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs,
    ) -> Iterator[StreamPart]:
        """同步流式调用模型，生成内容增量块"""
        request = self._build_request(messages, system, tools, tool_choice, kwargs)
        yield from self._stream_impl(request)

    # ==========================================================================
    # 公共 API - 异步方法
    # ==========================================================================

    async def ainvoke(
        self,
        messages: list[Message],
        system: str | List[ContentPart] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs,
    ) -> MessageResponse:
        """异步调用模型"""
        request = self._build_request(messages, system, tools, tool_choice, kwargs)
        return await self._ainvoke_impl(request)

    @asynccontextmanager
    async def astream(
        self,
        messages: list[Message],
        system: str | List[ContentPart] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs,
    ):
        """异步流式调用模型，生成内容增量块"""
        request = self._build_request(messages, system, tools, tool_choice, kwargs)
        
        # 获取底层生成器
        gen = self._astream_impl(request)
        
        try:
            yield gen  # type: ignore[misc]
        finally:
            # 确保生成器被关闭
            await gen.aclose()

    # ==========================================================================
    # 请求/响应转换 - 子类必须实现
    # ==========================================================================

    @abstractmethod
    def _prepare_request_impl(self, request: MessageRequest) -> ProviderRequest:
        """将通用请求转换为提供商特定格式"""
        pass

    @abstractmethod
    def _parse_response_impl(self, response: ProviderResponse) -> MessageResponse:
        """将提供商响应转换为通用格式"""
        pass

    # ==========================================================================
    # 调用实现 - 子类必须实现同步版本
    # ==========================================================================

    @abstractmethod
    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        """同步调用实现"""
        pass

    # ==========================================================================
    # 调用实现 - 子类可选实现（默认提供基于 sync 的 fallback）
    # ==========================================================================

    async def _ainvoke_impl(self, request: MessageRequest) -> MessageResponse:
        """异步调用实现（默认使用线程池）"""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, self._invoke_impl, request)

    def _stream_impl(self, request: MessageRequest) -> Iterator[StreamPart]:
        """同步流式实现（默认不支持）"""
        raise NotImplementedError(f"{self.__class__.__name__} does not support streaming")

    async def _astream_impl(self, request: MessageRequest) -> AsyncGenerator[StreamPart, None]:
        """异步流式实现（默认使用线程池）"""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[StreamPart | None] = asyncio.Queue()

        def stream_in_thread():
            try:
                for chunk in self._stream_impl(request):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        with ThreadPoolExecutor() as pool:
            pool.submit(stream_in_thread)
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk

    # ==========================================================================
    # 内部工具方法
    # ==========================================================================

    def _build_request(
        self,
        messages: list[Message],
        system: str | List[ContentPart] | None,
        tools: list[ToolDefinition] | None,
        tool_choice: ToolChoice | None,
        override_params: ModelParams,
    ) -> MessageRequest:
        """构建 MessageRequest 对象"""
        if isinstance(system, str):
            system = [TextPart(type='text', text=system)]

        params = self._get_params()
        merged = {**params, **override_params}

        return MessageRequest(
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=merged.get("parallel_tool_calls"),
            max_tokens=merged.get("max_tokens"),
            max_completion_tokens=merged.get("max_completion_tokens"),
            temperature=merged.get("temperature"),
            top_p=merged.get("top_p"),
            response_format=merged.get("response_format"),
            reasoning_effort=merged.get("reasoning_effort"),
            service_tier=merged.get("service_tier"),
        )

    def _get_params(self) -> ModelParams:
        """获取模型参数（子类可覆盖）"""
        return {}

    def prepare_request(self, request: MessageRequest) -> ProviderRequest:
        """将通用请求转换为提供商特定格式"""
        return self._prepare_request_impl(request)

    def parse_response(self, response: ProviderResponse) -> MessageResponse:
        """将提供商响应转换为通用格式"""
        return self._parse_response_impl(response)

    # ==========================================================================
    # 余额查询 - 可选实现
    # ==========================================================================

    def get_balance(self) -> list[BalanceInfo]:
        """
        查询账户余额

        返回各币种的余额信息列表。不同提供商返回的字段可能不同，
        详细信息存储在 BalanceInfo.details 中。

        Returns:
            BalanceInfo 对象列表，每个对象代表一个币种的余额

        Raises:
            NotImplementedError: 如果该模型不支持余额查询
            RuntimeError: 如果 API 调用失败

        Example:
            >>> model = DeepSeekModel(api_key="sk-...")
            >>> balances = model.get_balance()
            >>> for b in balances:
            ...     print(f"{b.currency}: {b.available_balance}")
            CNY: 100.00
            USD: 15.50
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support balance query")

    # ==========================================================================
    # 错误分类 - 子类可覆盖以提供提供商特定的错误映射
    # ==========================================================================

    def classify_error(self, exception: Exception) -> str:
        """分类模型调用异常为错误类型

        子类应覆盖此方法以提供提供商特定的错误分类。

        Args:
            exception: 捕获的异常

        Returns:
            错误类型 (ModelErrorType.NETWORK, THROTTLE, DENIED, UNKNOWN)
        """
        error_str = str(exception).lower()

        # 限流错误检测
        if any(kw in error_str for kw in ["rate limit", "429", "too many requests", "throttle"]):
            return ModelErrorType.THROTTLE

        # 权限错误检测
        if any(kw in error_str for kw in ["unauthorized", "forbidden", "401", "403", "denied", "api key"]):
            return ModelErrorType.DENIED

        # 网络错误检测
        if any(kw in error_str for kw in ["connection", "timeout", "network", "dns", "refused", "reset"]):
            return ModelErrorType.NETWORK

        return ModelErrorType.UNKNOWN
