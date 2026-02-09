"""
Kimi API (Moonshot) 兼容性适配器

解决的问题:
1. Kimi K2.5 thinking 模式要求保留 reasoning_content 字段
   错误: "thinking is enabled but reasoning_content is missing in assistant tool call message"

参考资料:
- https://github.com/danny-avila/LibreChat/issues/11563
- https://github.com/anomalyco/opencode/issues/10996
- https://github.com/CherryHQ/cherry-studio/issues/12619
- https://platform.moonshot.ai/docs/guide/kimi-k2-5-quickstart
"""

import logging
from collections.abc import AsyncGenerator
from typing import Mapping, Any, override

import openai
from openai import AsyncOpenAI

from strands.models.openai import OpenAIModel
from strands.types.tools import ToolSpec, ToolChoice
from strands.types.content import Messages, ContentBlock, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException

logger = logging.getLogger(__name__)


class KimiOpenAIModel(OpenAIModel):
    """Kimi API 兼容模型

    基于 OpenAIModel，修复了 Kimi K2.5 thinking 模式的兼容性问题。

    Kimi API 的特殊要求:
    - 默认启用 thinking 模式
    - thinking 模式下，assistant 消息的 reasoning_content 字段必须保留
    - tool 调用消息如果来自 thinking 模型，也必须包含 reasoning_content

    使用示例:
        # 启用 thinking 模式 (默认)
        model = KimiOpenAIModel(
            client_args={
                "api_key": "sk-...",
                "base_url": "https://api.moonshot.cn/v1",
            },
            model_id="kimi-k2.5",
        )

        # 禁用 thinking 模式 (如果需要 tool calling 且不想处理 reasoning_content)
        model = KimiOpenAIModel(
            client_args={...},
            model_id="kimi-k2.5",
            params={"temperature": 0.6, "thinking": {"type": "disabled"}},
        )
    """

    @classmethod
    def _extract_reasoning_content(cls, contents: list[ContentBlock]) -> str | None:
        """从消息内容中提取 reasoning_content

        Args:
            contents: 消息内容列表

        Returns:
            reasoning_content 文本，如果没有则返回 None
        """
        for content in contents:
            if "reasoningContent" in content:
                reasoning_text = content["reasoningContent"].get("reasoningText", {})
                return reasoning_text.get("text", "")
        return None

    @classmethod
    @override
    def _format_regular_messages(cls, messages: Messages, **kwargs: Any) -> list[dict[str, Any]]:
        """格式化常规消息，保留 Kimi 的 reasoning_content

        重写父类方法以：
        1. 移除关于 reasoningContent 的警告（Kimi API 支持 reasoning_content）
        2. 保留 reasoningContent 并转换为 reasoning_content 字段

        Args:
            messages: 消息对象列表
            **kwargs: 额外参数

        Returns:
            格式化后的消息列表
        """
        formatted_messages = []

        for message in messages:
            contents = message["content"]

            # 提取 reasoning_content（Kimi K2.5 Interleaved Thinking 支持）
            reasoning_content = cls._extract_reasoning_content(contents)

            formatted_contents = [
                cls.format_request_message_content(content)
                for content in contents
                if not any(block_type in content for block_type in ["toolResult", "toolUse", "reasoningContent"])
            ]
            formatted_tool_calls = [
                cls.format_request_message_tool_call(content["toolUse"]) for content in contents if "toolUse" in content
            ]
            formatted_tool_messages = [
                cls.format_request_tool_message(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

            formatted_message: dict[str, Any] = {
                "role": message["role"],
                "content": formatted_contents,
            }

            # 添加 tool_calls（如果有）
            if formatted_tool_calls:
                formatted_message["tool_calls"] = formatted_tool_calls

            # 添加 reasoning_content（Kimi K2.5 必需）
            # 对于 assistant 消息，如果有 reasoning_content 或者是 tool call，都需要添加
            if reasoning_content is not None:
                formatted_message["reasoning_content"] = reasoning_content
            elif message["role"] == "assistant" and formatted_tool_calls:
                # tool call 消息必须有 reasoning_content（即使是空字符串）
                formatted_message["reasoning_content"] = ""

            formatted_messages.append(formatted_message)

            # 处理工具消息（提取图片到 user 消息）
            for tool_msg in formatted_tool_messages:
                tool_msg_clean, user_msg_with_images = cls._split_tool_message_images(tool_msg)
                formatted_messages.append(tool_msg_clean)
                if user_msg_with_images:
                    formatted_messages.append(user_msg_with_images)

        return formatted_messages

    @classmethod
    @override
    def format_request_messages(
        cls,
        messages: Messages,
        system_prompt: str | None = None,
        *,
        system_prompt_content=None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """格式化消息，保留 reasoning_content 字段

        Kimi K2.5 thinking 模式要求:
        - assistant 消息的 reasoning_content 必须保留
        - tool 调用消息不能丢失 reasoning_content
        """
        # 使用重写后的 _format_system_messages 和 _format_regular_messages
        formatted_messages = cls._format_system_messages(system_prompt, system_prompt_content=system_prompt_content)
        formatted_messages.extend(cls._format_regular_messages(messages))

        return [message for message in formatted_messages if message["content"] or "tool_calls" in message]

    def _is_thinking_disabled(self) -> bool:
        """检查是否禁用了 thinking 模式

        Returns:
            True 如果 thinking 模式被禁用
        """
        params = self.config.get("params", {})
        if isinstance(params, Mapping):
            thinking_config = params.get("thinking", {})
            return thinking_config.get("type") == "disabled"
        return False

    def _validate_kimi_k25_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """验证并修正 Kimi K2.5 模型的固定参数

        根据官方文档，Kimi K2.5 有以下固定参数要求:
        - temperature: thinking 模式为 1.0，非 thinking 模式为 0.6
        - top_p: 固定为 0.95
        - n: 固定为 1
        - presence_penalty: 固定为 0.0
        - frequency_penalty: 固定为 0.0

        Args:
            params: 原始参数字典

        Returns:
            修正后的参数字典
        """
        model_id = self.config.get("model_id", "")
        if model_id not in ("kimi-k2.5", "kimi-k2-thinking", "kimi-k2-thinking-turbo"):
            return params

        validated = dict(params)

        # 根据是否启用 thinking 模式强制设置 temperature
        if self._is_thinking_disabled():
            validated["temperature"] = 0.6
        else:
            validated["temperature"] = 1.0

        # 强制设置其他固定参数
        validated["top_p"] = 0.95
        validated["n"] = 1
        validated["presence_penalty"] = 0.0
        validated["frequency_penalty"] = 0.0

        logger.debug(
            "Kimi K2.5 强制使用固定参数: temperature=%s, top_p=0.95, n=1",
            validated["temperature"]
        )

        return validated

    def _get_default_max_tokens(self, model_id: str) -> int:
        """获取模型的默认 max_tokens 值

        Args:
            model_id: 模型 ID

        Returns:
            默认 max_tokens 值
        """
        defaults = {
            "kimi-k2.5": 32768,
            "kimi-k2-thinking": 64000,
            "kimi-k2-thinking-turbo": 64000,
        }
        return defaults.get(model_id, 4096)

    @override
    def format_request(
        self,
        messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: ToolChoice | None = None,
        *,
        system_prompt_content=None,
        **kwargs,
    ) -> dict[str, Any]:
        """格式化请求，支持禁用 thinking 模式和 K2.5 固定参数"""
        request = super().format_request(
            messages,
            tool_specs,
            system_prompt,
            tool_choice,
            system_prompt_content=system_prompt_content,
            **kwargs,
        )

        # 对 K2.5 模型应用固定参数
        request = self._validate_kimi_k25_params(request)

        # 检查是否禁用了 thinking 模式
        if self._is_thinking_disabled():
            # 通过 extra_body 传递参数
            request["extra_body"] = {"thinking": {"type": "disabled"}}
            # 从 request 中移除 thinking 参数
            if "thinking" in request:
                del request["thinking"]

        # 设置默认 max_tokens（如果未指定）
        model_id = self.config.get("model_id", "")
        if "max_tokens" not in request:
            request["max_tokens"] = self._get_default_max_tokens(model_id)

        return request

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """流式处理响应，保留 tool call 时的 reasoning_content

        重写父类方法以收集 tool call 过程中的 reasoning_content，
        避免在消息历史中丢失推理内容。
        """
        request = self.format_request(messages, tool_specs, system_prompt, tool_choice)

        async with AsyncOpenAI(**self.client_args) as client:
            try:
                response = await client.chat.completions.create(**request)
            except openai.BadRequestError as e:
                if hasattr(e, "code") and e.code == "context_length_exceeded":
                    raise ContextWindowOverflowException(str(e)) from e
                raise
            except openai.RateLimitError as e:
                raise ModelThrottledException(str(e)) from e

            yield self.format_chunk({"chunk_type": "message_start"})

            tool_calls: dict[int, list[Any]] = {}
            data_type = None
            finish_reason = None
            reasoning_content_parts: list[str] = []  # 收集 reasoning_content
            event = None

            async for event in response:
                if not getattr(event, "choices", None):
                    continue
                choice = event.choices[0]

                # 收集 reasoning_content
                if hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content:
                    reasoning_content_parts.append(choice.delta.reasoning_content)
                    chunks, data_type = self._stream_switch_content("reasoning_content", data_type)
                    for chunk in chunks:
                        yield chunk
                    yield self.format_chunk({
                        "chunk_type": "content_delta",
                        "data_type": data_type,
                        "data": choice.delta.reasoning_content,
                    })

                if choice.delta.content:
                    chunks, data_type = self._stream_switch_content("text", data_type)
                    for chunk in chunks:
                        yield chunk
                    yield self.format_chunk({
                        "chunk_type": "content_delta",
                        "data_type": data_type,
                        "data": choice.delta.content,
                    })

                for tool_call in choice.delta.tool_calls or []:
                    tool_calls.setdefault(tool_call.index, []).append(tool_call)

                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                    if data_type:
                        yield self.format_chunk({"chunk_type": "content_stop", "data_type": data_type})
                    break

            # 处理 tool calls，同时保留收集到的 reasoning_content
            full_reasoning_content = "".join(reasoning_content_parts)
            for tool_deltas in tool_calls.values():
                # 在 tool call 开始时，先输出收集到的 reasoning_content
                if full_reasoning_content:
                    yield self.format_chunk({
                        "chunk_type": "content_delta",
                        "data_type": "reasoning_content",
                        "data": full_reasoning_content,
                    })

                yield self.format_chunk({
                    "chunk_type": "content_start",
                    "data_type": "tool",
                    "data": tool_deltas[0],
                })
                for tool_delta in tool_deltas:
                    yield self.format_chunk({
                        "chunk_type": "content_delta",
                        "data_type": "tool",
                        "data": tool_delta,
                    })
                yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool"})

            yield self.format_chunk({"chunk_type": "message_stop", "data": finish_reason or "end_turn"})

            async for event in response:
                _ = event

            if event and hasattr(event, "usage") and event.usage:
                yield self.format_chunk({"chunk_type": "metadata", "data": event.usage})


# 预配置的 Kimi 模型实例
def create_kimi_model(
    api_key: str,
    model_id: str = "kimi-k2.5",
    base_url: str = "https://api.moonshot.cn/v1",
    enable_thinking: bool = True,
    **kwargs
) -> KimiOpenAIModel:
    """创建 Kimi 模型实例

    Args:
        api_key: Kimi/Moonshot API key
        model_id: 模型 ID，默认为 "kimi-k2.5"
        base_url: API base URL，默认为 Moonshot 官方地址
        enable_thinking: 是否启用 thinking 模式，默认为 True
        **kwargs: 其他参数传递给 KimiOpenAIModel

    Returns:
        KimiOpenAIModel 实例
    """
    params = kwargs.pop("params", {})

    if not enable_thinking:
        # 禁用 thinking 模式时的默认参数
        params.setdefault("temperature", 0.6)
        params["thinking"] = {"type": "disabled"}

    return KimiOpenAIModel(
        client_args={
            "api_key": api_key,
            "base_url": base_url,
        },
        model_id=model_id,
        params=params,
        **kwargs
    )
