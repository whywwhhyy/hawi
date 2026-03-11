"""
Hawi Agent Model 基类

提供统一的 Model 抽象，支持同步和异步操作，兼容多种 LLM 提供商。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Iterator, List, Literal, overload

from hawi.models.message import (
    ContentPart,
    Message,
    MessageRequest,
    MessageResponse,
    DeltaPart,
    TextPart,
    ToolCallPart,
    ToolDefinition,
    ToolChoice,
)
from hawi.errors import ModelError
from hawi.events.event import Event

__all__ = ["Model", "DeltaPart", "BalanceInfo", "ProviderRequest", "ProviderResponse", "ModelParams", "BalanceDetails", "ModelError"]

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

    @overload
    def invoke(
        self,
        messages: list[Message],
        *,
        streaming: Literal[False] = False,
        system: str | List[ContentPart] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs,
    ) -> MessageResponse: ...

    @overload
    def invoke(
        self,
        messages: list[Message],
        *,
        streaming: Literal[True],
        system: str | List[ContentPart] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs,
    ) -> Iterator[DeltaPart]: ...

    def invoke(
        self,
        messages: list[Message],
        *,
        streaming: bool = False,
        system: str | List[ContentPart] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs,
    ) -> MessageResponse | Iterator[DeltaPart]:
        """同步调用模型

        Args:
            messages: 消息列表
            streaming: 是否使用流式模式。False 返回 MessageResponse，True 返回 Iterator[DeltaPart]
            system: 系统提示
            tools: 工具定义列表
            tool_choice: 工具选择策略
            **kwargs: 其他参数

        Returns:
            streaming=False: MessageResponse
            streaming=True: Iterator[DeltaPart]
        """
        request = self._build_request(messages, system, tools, tool_choice, kwargs)
        if streaming:
            return self._stream_impl(request)
        return self._invoke_impl(request)

    # ==========================================================================
    # 公共 API - 异步方法
    # ==========================================================================

    async def ainvoke(
        self,
        messages: list[Message],
        *,
        streaming: bool = False,
        system: str | List[ContentPart] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs,
    ) -> AsyncGenerator[DeltaPart, None]:
        """异步调用模型，返回 DeltaPart 流

        Args:
            messages: 消息列表
            streaming: 是否使用流式 HTTP API。True=流式API，False=非流式API
            system: 系统提示
            tools: 工具定义列表
            tool_choice: 工具选择策略
            **kwargs: 其他参数

        Yields:
            DeltaPart 增量块
            - streaming=True: 实时转发 LLM API 的流式响应
            - streaming=False: 将完整响应拆分为 DeltaPart 序列
        """
        request = self._build_request(messages, system, tools, tool_choice, kwargs)
        if streaming:
            async for delta in self._astream_impl(request):
                yield delta
        else:
            async for delta in self._ainvoke_impl(request):
                yield delta

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
    # 调用实现
    # ==========================================================================

    @abstractmethod
    def _invoke_impl(
        self,
        request: MessageRequest,
    ) -> MessageResponse:
        """同步调用实现

        Args:
            request: 消息请求

        Returns:
            MessageResponse: 完整的模型响应
        """
        pass

    async def _ainvoke_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        """异步非流式实现

        调用非流式 API，将完整响应拆分为 DeltaPart 序列 yield。
        子类必须重写此方法。

        Args:
            request: 消息请求

        Yields:
            DeltaPart 增量块序列
        """
        # This is an async generator, so we need to yield something
        # Subclasses should override this method
        raise NotImplementedError(f"{self.__class__.__name__} does not support async non-streaming")
        yield  # Make this an async generator

    def _stream_impl(self, request: MessageRequest) -> Iterator[DeltaPart]:
        """同步流式实现（默认不支持）"""
        raise NotImplementedError(f"{self.__class__.__name__} does not support streaming")

    async def _astream_impl(
        self,
        request: MessageRequest,
    ) -> AsyncGenerator[DeltaPart, None]:
        """异步流式实现

        调用流式 API，实时转发 DeltaPart。
        子类必须重写此方法。

        Args:
            request: 消息请求

        Yields:
            DeltaPart 流式增量块
        """
        # This is an async generator, so we need to yield something
        # Subclasses should override this method
        raise NotImplementedError(f"{self.__class__.__name__} does not support async streaming")
        yield  # Make this an async generator

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
