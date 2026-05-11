"""
Hawi Agent Model 基类

提供统一的 Model 抽象，支持同步和异步操作，兼容多种 LLM 提供商。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import json
import math

from typing import Any, AsyncGenerator, ClassVar, Iterator, List, Literal, cast, overload

from hawi.models.message import (
    ContentPart,
    Message,
    MessageRequest,
    MessageResponse,
    STEER_MERGE_MODES,
    DeltaPart,
    SteerMergeMode,
    SteerPart,
    TextPart,
    ToolCallPart,
    ToolDefinition,
    ToolChoice,
    ToolResultPart,
    normalize_cache_point,
)
from hawi.errors import ConfigurationError, ModelError

TokenEstimateMethod = Literal[
    "provider_count",
    "provider_tokenize",
    "local_tokenizer",
    "heuristic",
    "unsupported",
]
TokenEstimateConfidence = Literal["exact", "approximate", "unsupported"]

__all__ = [
    "Model",
    "DelegateModel",
    "DeltaPart",
    "BalanceInfo",
    "ProviderRequest",
    "ProviderResponse",
    "ModelParams",
    "BalanceDetails",
    "TokenEstimate",
    "TokenEstimateMethod",
    "TokenEstimateConfidence",
    "ModelError",
]

STEER_ASSISTANT_ACK_TEXT = (
    "The user is sending a new steering message, I'll reply to it and "
    "continue the ongoing task with it"
)

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


@dataclass
class TokenEstimate:
    """Estimated token cost for a provider-ready model request.

    ``provider_count`` is the preferred method because providers count the same
    chat/message request shape that will be billed. Tokenizer and heuristic
    estimates can be useful for UI budgeting, but should be treated as
    approximate because providers may add hidden formatting tokens.
    """

    input_tokens: int | None = None
    context_tokens: int | None = None
    total_tokens: int | None = None
    method: TokenEstimateMethod = "unsupported"
    confidence: TokenEstimateConfidence = "unsupported"
    provider: str | None = None
    model_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def unsupported(
        cls,
        *,
        provider: str | None = None,
        model_id: str | None = None,
        reason: str | None = None,
    ) -> "TokenEstimate":
        details = {"reason": reason} if reason else {}
        return cls(
            method="unsupported",
            confidence="unsupported",
            provider=provider,
            model_id=model_id,
            details=details,
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

    default_steer_merge_mode: ClassVar[SteerMergeMode | None] = None
    """Concrete models must declare their preferred steer lowering strategy."""

    def __init__(self) -> None:
        """初始化模型基类，设置 _async_only 标记。"""
        # _async_only=True 表示此模型仅用于异步调用（从对象池获取的共享实例）
        # 此时同步调用 invoke/stream 会被阻止，以避免阻塞事件循环
        self._async_only: bool = False
        self._configured_steer_merge_mode: SteerMergeMode | None = None

    def reset(self) -> None:
        """重置模型状态。

        清除缓存的客户端连接和其他运行时状态。
        在切换模型时调用，确保旧模型资源被正确释放。

        子类应覆盖此方法以清理特定资源，并调用 super().reset()。
        """
        pass  # 基类无资源需要清理

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
        # 检查 _async_only 标记
        if getattr(self, '_async_only', False):
            raise RuntimeError(
                "This model was obtained with async_only=True and can only be used for async calls. "
                "Please use ainvoke() instead of invoke(), or obtain the model with async_only=False."
            )

        request = self._build_request(messages, system, tools, tool_choice, kwargs)
        if streaming:
            return self._stream_impl(request)
        return self._invoke_impl(request)

    def estimate_tokens(
        self,
        messages: list[Message],
        *,
        system: str | List[ContentPart] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs,
    ) -> TokenEstimate:
        """Estimate input/context tokens for a model request.

        Providers with official count endpoints should override
        :meth:`_estimate_tokens_impl`. The default implementation returns a
        low-confidence heuristic, useful for UI context meters but not billing.
        """
        if getattr(self, '_async_only', False):
            raise RuntimeError(
                "This model was obtained with async_only=True and can only be used for async calls. "
                "Please use aestimate_tokens(), or obtain the model with async_only=False."
            )

        request = self._build_request(messages, system, tools, tool_choice, kwargs)
        return self._estimate_tokens_impl(request)

    def list_models(self) -> list[str]:
        """Return model IDs currently available from this provider.

        Providers that expose a model-list endpoint should override this. The
        IDs returned here are provider-local IDs, not ``provider/model`` names.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support model list query"
        )

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

    async def aestimate_tokens(
        self,
        messages: list[Message],
        *,
        system: str | List[ContentPart] | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs,
    ) -> TokenEstimate:
        """Async version of :meth:`estimate_tokens`."""
        request = self._build_request(messages, system, tools, tool_choice, kwargs)
        return await self._aestimate_tokens_impl(request)

    async def alist_models(self) -> list[str]:
        """Async version of :meth:`list_models`."""
        return self.list_models()

    @staticmethod
    def _coerce_model_id_list(response: Any) -> list[str]:
        """Extract a de-duplicated list of model IDs from SDK list responses."""
        if isinstance(response, dict):
            items = response.get("data", [])
        elif hasattr(response, "data"):
            items = getattr(response, "data")
        elif not isinstance(response, (str, bytes)) and hasattr(response, "__iter__"):
            items = response
        else:
            items = response

        ids: list[str] = []
        seen: set[str] = set()
        try:
            iterator = iter(items)
        except TypeError:
            return ids

        for item in iterator:
            model_id = Model._coerce_model_id(item)
            if model_id and model_id not in seen:
                ids.append(model_id)
                seen.add(model_id)
        return ids

    @staticmethod
    async def _acoerce_model_id_list(response: Any) -> list[str]:
        """Async variant of :meth:`_coerce_model_id_list`."""
        if hasattr(response, "__aiter__"):
            ids: list[str] = []
            seen: set[str] = set()
            async for item in response:
                model_id = Model._coerce_model_id(item)
                if model_id and model_id not in seen:
                    ids.append(model_id)
                    seen.add(model_id)
            return ids
        return Model._coerce_model_id_list(response)

    @staticmethod
    def _coerce_model_id(item: Any) -> str | None:
        if isinstance(item, str):
            value = item
        elif isinstance(item, dict):
            value = item.get("id") or item.get("model") or item.get("name")
        else:
            value = (
                getattr(item, "id", None)
                or getattr(item, "model", None)
                or getattr(item, "name", None)
            )
        if value is None:
            return None
        text = str(value).strip()
        return text or None

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

    def _estimate_tokens_impl(
        self,
        request: MessageRequest,
    ) -> TokenEstimate:
        """Estimate token usage for one prepared request.

        The base implementation intentionally does not claim provider accuracy.
        It serializes the normalized request shape and uses a rough character
        heuristic so callers can still show a best-effort context meter.
        """
        return self._heuristic_token_estimate(request)

    async def _aestimate_tokens_impl(
        self,
        request: MessageRequest,
    ) -> TokenEstimate:
        """Async token estimate implementation."""
        return self._estimate_tokens_impl(request)

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

    def _heuristic_token_estimate(
        self,
        request: MessageRequest,
        *,
        details: dict[str, Any] | None = None,
    ) -> TokenEstimate:
        payload = request.model_dump(exclude_none=True, mode="json")
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        # A deliberately conservative approximation for mixed English/CJK JSON
        # request payloads. Provider-specific adapters should replace this with
        # official count APIs whenever possible.
        ascii_chars = sum(1 for char in serialized if ord(char) < 128)
        non_ascii_chars = len(serialized) - ascii_chars
        input_tokens = max(1, math.ceil(ascii_chars / 4 + non_ascii_chars / 2))
        return TokenEstimate(
            input_tokens=input_tokens,
            context_tokens=input_tokens,
            total_tokens=input_tokens,
            method="heuristic",
            confidence="approximate",
            provider=self.__class__.__name__,
            model_id=self.model_id,
            details={"serialized_chars": len(serialized), **(details or {})},
        )

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
        lowered_messages = self.lower_messages(messages)
        cache_point_value = (
            merged["cache_point"] if "cache_point" in merged else merged.get("cache_control")
        )
        cache_tool_definitions_value = (
            merged["cache_tool_definitions"]
            if "cache_tool_definitions" in merged
            else merged.get("tool_definitions_cache_point")
        )

        return MessageRequest(
            messages=lowered_messages,
            system=system,
            cache_point=normalize_cache_point(cache_point_value),
            cache_tool_definitions=normalize_cache_point(cache_tool_definitions_value),
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=merged.get("parallel_tool_calls"),
            max_output_tokens=merged.get("max_output_tokens"),
            temperature=merged.get("temperature"),
            top_p=merged.get("top_p"),
            response_format=merged.get("response_format"),
            reasoning_effort=merged.get("reasoning_effort"),
            service_tier=merged.get("service_tier"),
            thinking_budget=merged.get("thinking_budget"),
            thinking_type=merged.get("thinking_type"),
            thinking_effort=merged.get("thinking_effort"),
            output_config=merged.get("output_config"),
        )

    def _get_params(self) -> ModelParams:
        """获取模型参数（子类可覆盖）"""
        return {}

    def prepare_request(self, request: MessageRequest) -> ProviderRequest:
        """将通用请求转换为提供商特定格式"""
        lowered_messages = self.lower_messages(request.messages)
        if lowered_messages != request.messages:
            request = request.model_copy(update={"messages": lowered_messages})
        return self._prepare_request_impl(request)

    @staticmethod
    def coerce_steer_merge_mode(
        merge_mode: Any,
        *,
        source: str,
    ) -> SteerMergeMode | None:
        """Validate and normalize a steer merge mode value."""
        if merge_mode is None:
            return None
        if merge_mode in STEER_MERGE_MODES:
            return cast(SteerMergeMode, merge_mode)
        valid = ", ".join(STEER_MERGE_MODES)
        raise ConfigurationError(
            f"Invalid steer_merge_mode from {source}: {merge_mode!r}. "
            f"Valid values are: {valid}. "
            "Set steer_merge_mode explicitly in models.yaml or pass one of "
            "these values in enqueue metadata."
        )

    def configure_steer_merge_mode(self, merge_mode: str | None) -> None:
        """Attach configured steer merge mode to this model instance."""
        self._configured_steer_merge_mode = self.coerce_steer_merge_mode(
            merge_mode,
            source=f"{self.__class__.__name__}.configure_steer_merge_mode",
        )

    def configure_max_context_tokens(self, max_context_tokens: int | None) -> None:
        """Attach Hawi-level context window metadata to this model instance."""
        if max_context_tokens is not None and max_context_tokens > 0:
            self._max_context_tokens = max_context_tokens
        else:
            self._max_context_tokens = None

    def get_max_context_tokens(self) -> int | None:
        """Return configured context window size for this model, if known."""
        value = getattr(self, "_max_context_tokens", None)
        return value if isinstance(value, int) and value > 0 else None

    def get_default_steer_merge_mode(self) -> SteerMergeMode:
        """Get the default steer lowering strategy for this model."""
        mode = self.default_steer_merge_mode
        if mode in STEER_MERGE_MODES:
            return cast(SteerMergeMode, mode)

        model_id = getattr(self, "model_id", "<unknown>")
        valid = ", ".join(STEER_MERGE_MODES)
        raise ConfigurationError(
            f"Model {self.__class__.__name__}({model_id!r}) does not declare "
            "default_steer_merge_mode, and no steer_merge_mode override was "
            "configured. Steering mode has no framework default by design. "
            f"Declare default_steer_merge_mode on the model class or set "
            f"steer_merge_mode in models.yaml. Valid values: {valid}."
        )

    def get_configured_steer_merge_mode(self) -> SteerMergeMode | None:
        """Get steer merge mode configured for this model instance."""
        return getattr(self, "_configured_steer_merge_mode", None)

    def get_effective_steer_merge_mode(self) -> SteerMergeMode:
        """Return the configured mode or the concrete model declaration."""
        configured = self.get_configured_steer_merge_mode()
        if configured is not None:
            return configured
        return self.get_default_steer_merge_mode()

    def validate_steer_merge_mode_config(self, *, source: str | None = None) -> None:
        """Fail early if this model cannot resolve a steer merge mode."""
        try:
            self.get_effective_steer_merge_mode()
        except ConfigurationError as exc:
            prefix = f"{source}: " if source else ""
            raise ConfigurationError(prefix + str(exc)) from exc

    def lower_messages(self, messages: list[Message]) -> list[Message]:
        """Lower Hawi IR messages into provider-ready messages."""
        lowered: list[Message] = []
        for message in messages:
            steer_part = self._extract_steer_part(message)
            if steer_part is None:
                lowered.append(deepcopy(message))
                continue
            self._lower_steer_message(lowered, message, steer_part)
        return lowered

    def _extract_steer_part(self, message: Message) -> SteerPart | None:
        """Extract the SteerPart from a message if it is a steer message."""
        if message.get("role") != "user":
            return None
        content = message.get("content", [])
        if len(content) != 1:
            return None
        part = content[0]
        if isinstance(part, dict) and part.get("type") == "steer":
            return cast(SteerPart, part)
        return None

    def _lower_steer_message(
        self,
        lowered: list[Message],
        original_message: Message,
        steer_part: SteerPart,
    ) -> None:
        """Lower one steer message according to the model's strategy."""
        steer_content = deepcopy(steer_part.get("content", []))
        plain_user_message = self._build_plain_user_message(
            original_message,
            steer_content,
        )
        tool_index = self._find_related_tool_message_index(
            lowered,
            steer_part.get("tool_call_id"),
        )
        if tool_index is None:
            lowered.append(plain_user_message)
            return

        merge_mode = self._resolve_steer_merge_mode(steer_part)
        if merge_mode == "append_to_tool_result":
            lowered[tool_index] = self._append_steer_to_tool_message(
                lowered[tool_index],
                steer_part.get("tool_call_id"),
                steer_content,
            )
            return

        if merge_mode == "user_message_template":
            tool_message = lowered.pop(tool_index)
            combined_text = self._build_user_message_template_text(
                tool_message,
                steer_content,
                steer_part.get("tool_call_id"),
            )
            lowered.append({
                "role": "user",
                "content": [{"type": "text", "text": combined_text}],
                "name": original_message.get("name"),
                "metadata": original_message.get("metadata"),
            })
            return

        if merge_mode == "tool_result_assistant_template_and_user_message":
            lowered.append({
                "role": "assistant",
                "content": [{"type": "text", "text": STEER_ASSISTANT_ACK_TEXT}],
                "name": None,
                "metadata": None,
            })
            lowered.append(plain_user_message)
            return

        lowered.append(plain_user_message)

    def _resolve_steer_merge_mode(self, steer_part: SteerPart) -> SteerMergeMode:
        """Resolve the effective steer merge mode for one steer message."""
        preferred = steer_part.get("preferred_merge_mode")
        if preferred is not None:
            return self.coerce_steer_merge_mode(
                preferred,
                source="steer message preferred_merge_mode",
            ) or self.get_effective_steer_merge_mode()
        return self.get_effective_steer_merge_mode()

    def _build_plain_user_message(
        self,
        original_message: Message,
        steer_content: list[ContentPart],
    ) -> Message:
        """Convert a steer message into a plain user message."""
        return {
            "role": "user",
            "content": steer_content,
            "name": original_message.get("name"),
            "metadata": original_message.get("metadata"),
        }

    def _find_related_tool_message_index(
        self,
        messages: list[Message],
        tool_call_id: str | None,
    ) -> int | None:
        """Find the most recent related tool message for the steer message."""
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.get("role") != "tool":
                continue
            tool_part = self._extract_tool_result_part(message, tool_call_id)
            if tool_part is not None:
                return index
        return None

    def _extract_tool_result_part(
        self,
        message: Message,
        tool_call_id: str | None,
    ) -> ToolResultPart | None:
        """Extract the related tool result part from a tool message."""
        for part in message.get("content", []):
            if not isinstance(part, dict) or part.get("type") != "tool_result":
                continue
            tool_part = cast(ToolResultPart, part)
            if tool_call_id is None or tool_part.get("tool_call_id") == tool_call_id:
                return tool_part
        return None

    def _append_steer_to_tool_message(
        self,
        message: Message,
        tool_call_id: str | None,
        steer_content: list[ContentPart],
    ) -> Message:
        """Append steer content to the related tool result message."""
        updated_message = deepcopy(message)
        for part in updated_message.get("content", []):
            if not isinstance(part, dict) or part.get("type") != "tool_result":
                continue
            tool_part = cast(ToolResultPart, part)
            if tool_call_id is not None and tool_part.get("tool_call_id") != tool_call_id:
                continue
            nested_content = tool_part.get("content", [])
            if isinstance(nested_content, str):
                new_content: list[ContentPart] = [{"type": "text", "text": nested_content}]
            else:
                new_content = list(nested_content)
            tool_part["content"] = new_content + steer_content
            break
        return updated_message

    def _build_user_message_template_text(
        self,
        tool_message: Message,
        steer_content: list[ContentPart],
        tool_call_id: str | None,
    ) -> str:
        """Build the legacy combined user-message template text."""
        tool_part = self._extract_tool_result_part(tool_message, tool_call_id)
        tool_result_text = ""
        is_error = False
        if tool_part is not None:
            is_error = bool(tool_part.get("is_error"))
            nested_content = tool_part.get("content", [])
            if isinstance(nested_content, str):
                tool_result_text = nested_content
            else:
                tool_result_text = self._serialize_content_parts(list(nested_content))

        steer_text = self._serialize_content_parts(steer_content)
        sections = [
            f"[system] 用户发送了新的消息：{steer_text}",
            "[tool result] 另外，工具返回结果为：",
            tool_result_text,
        ]
        if is_error:
            sections.insert(2, "[tool result status] 该工具结果为错误结果。")
        return "\n".join(section for section in sections if section)

    def _serialize_content_parts(self, content: list[ContentPart]) -> str:
        """Serialize content parts into readable plain text."""
        chunks: list[str] = []
        for part in content:
            part_type = part.get("type")
            if part_type == "text":
                chunks.append(part.get("text", ""))
            elif part_type == "reasoning":
                chunks.append(part.get("reasoning") or "")
            elif part_type == "steer":
                nested_text = self._serialize_content_parts(
                    list(cast(SteerPart, part).get("content", []))
                )
                if nested_text:
                    chunks.append(nested_text)
            elif part_type == "tool_result":
                nested_content = part.get("content", [])
                if isinstance(nested_content, str):
                    chunks.append(nested_content)
                else:
                    nested_text = self._serialize_content_parts(list(nested_content))
                    if nested_text:
                        chunks.append(nested_text)
            else:
                chunks.append(str(part))
        return "\n".join(chunk for chunk in chunks if chunk.strip())

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


class DelegateModel(Model):
    """委托模型包装器

    将所有调用转发给内部委托模型实例，用于实现工厂入口类（如 DeepSeekModel、KimiModel），
    使其成为具体类而非抽象类，避免 Pylance 的不完整实现警告。

    子类在 __init__ 中创建实际的委托模型，并调用 super().__init__(delegate)。
    """

    def __init__(self, delegate: Model) -> None:
        super().__init__()
        self._delegate = delegate

    @property
    def model_id(self) -> str:
        return self._delegate.model_id

    def _prepare_request_impl(self, request: MessageRequest) -> ProviderRequest:
        return self._delegate._prepare_request_impl(request)

    def _parse_response_impl(self, response: ProviderResponse) -> MessageResponse:
        return self._delegate._parse_response_impl(response)

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        return self._delegate._invoke_impl(request)

    async def _ainvoke_impl(self, request: MessageRequest) -> AsyncGenerator[DeltaPart, None]:
        async for delta in self._delegate._ainvoke_impl(request):
            yield delta

    def _stream_impl(self, request: MessageRequest) -> Iterator[DeltaPart]:
        return self._delegate._stream_impl(request)

    def _estimate_tokens_impl(self, request: MessageRequest) -> TokenEstimate:
        return self._delegate._estimate_tokens_impl(request)

    async def _aestimate_tokens_impl(self, request: MessageRequest) -> TokenEstimate:
        return await self._delegate._aestimate_tokens_impl(request)

    async def _astream_impl(self, request: MessageRequest) -> AsyncGenerator[DeltaPart, None]:
        async for delta in self._delegate._astream_impl(request):
            yield delta

    def _get_params(self) -> ModelParams:
        return self._delegate._get_params()

    def get_default_steer_merge_mode(self) -> SteerMergeMode:
        return self._delegate.get_default_steer_merge_mode()

    def get_balance(self) -> list[BalanceInfo]:
        return self._delegate.get_balance()

    def list_models(self) -> list[str]:
        return self._delegate.list_models()

    async def alist_models(self) -> list[str]:
        return await self._delegate.alist_models()
