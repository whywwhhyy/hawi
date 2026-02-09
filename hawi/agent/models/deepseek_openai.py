"""DeepSeek OpenAI API 兼容性适配器

解决的问题:
1. DeepSeek API 的 tool 消息 content 字段只接受字符串，不接受数组格式
   (OpenAI API 支持两种格式)
2. DeepSeek Reasoner 模型(thinking mode)的特殊参数限制
3. DeepSeek Reasoner 的 reasoning_content 在多轮对话中的处理

基于 OpenAI API 格式，使用 base_url=https://api.deepseek.com

参考资料:
- https://github.com/marimo-team/marimo/issues/7036
- https://github.com/microsoft/vscode-ai-toolkit/issues/264
- https://github.com/cline/cline/issues/230
- https://api-docs.deepseek.com/guides/thinking_mode
- https://api-docs.deepseek.com/guides/openai_api
"""

import json
import logging
from typing import Any, cast

from strands.models.openai import OpenAIModel
from strands.types.content import ContentBlock, Messages, SystemContentBlock
from strands.types.tools import ToolResult


logger = logging.getLogger(__name__)

# DeepSeek Reasoner 模型不支持的参数 (设置了无效)
UNSUPPORTED_REASONER_PARAMS = {
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
}

# DeepSeek Reasoner 模型会报错的参数
ERROR_REASONER_PARAMS = {
    "logprobs",
    "top_logprobs",
}


class DeepSeekOpenAIModel(OpenAIModel):
    """DeepSeek OpenAI API 兼容模型

    基于 OpenAIModel，但修复了消息格式兼容性问题，并支持 Thinking Mode。

    DeepSeek API 与 OpenAI API 的差异:
    - OpenAI: tool 消息的 content 可以是 str 或 Iterable[ContentPart]
    - DeepSeek: tool 消息的 content 必须是 str

    DeepSeek Reasoner (Thinking Mode) 特殊限制:
    - 不支持: temperature, top_p, presence_penalty, frequency_penalty (设置了无效)
    - 报错: logprobs, top_logprobs
    - 响应包含 reasoning_content 字段，工具调用场景需回传以继续推理

    使用示例:
        # 普通模型
        model = DeepSeekOpenAIModel(
            client_args={
                "api_key": "sk-...",
                "base_url": "https://api.deepseek.com",
            },
            model_id="deepseek-chat",
            params={"temperature": 1, "max_tokens": 1024},
        )

        # Reasoner 模型 (Thinking Mode) - 工具调用场景需回传 reasoning_content
        model = DeepSeekOpenAIModel(
            client_args={
                "api_key": "sk-...",
                "base_url": "https://api.deepseek.com",
            },
            model_id="deepseek-reasoner",
            params={"max_tokens": 4096},  # 不要设置 temperature/top_p
            include_reasoning_in_context=True,  # 工具调用场景需要开启
        )
    """

    @classmethod
    def format_request_tool_message(cls, tool_result: ToolResult, **kwargs: Any) -> dict[str, Any]:
        """格式化 OpenAI 兼容的 tool 消息。

        覆盖父类方法，将 content 从数组格式转换为字符串格式，
        因为 DeepSeek API 只接受字符串格式的 content。

        Args:
            tool_result: 工具执行结果。
            **kwargs: 额外参数。

        Returns:
            OpenAI 兼容的 tool 消息，content 为字符串格式。
        """
        contents = cast(
            list[ContentBlock],
            [
                {"text": json.dumps(content["json"])} if "json" in content else content
                for content in tool_result["content"]
            ],
        )

        # DeepSeek API 只接受字符串格式的 content，不接受数组
        # 将所有内容合并为单个字符串
        text_parts = []
        for content in contents:
            if "text" in content:
                text_parts.append(content["text"])
            elif "image" in content:
                # 图片内容在 DeepSeek 中不被支持，记录警告
                logger.warning("DeepSeek API 不支持 tool 消息中的图片内容，已忽略")
                text_parts.append("[图片内容]")
            else:
                text_parts.append(str(content))

        # 合并所有文本内容
        # DeepSeek API 不接受空的 content，所以至少返回一个空格
        combined_content = "\n".join(text_parts) if text_parts else " "

        return {
            "role": "tool",
            "tool_call_id": tool_result["toolUseId"],
            "content": combined_content,
        }

    def __init__(
        self,
        client=None,
        client_args: dict[str, Any] | None = None,
        include_reasoning_in_context: bool = False,
        **model_config,
    ) -> None:
        """初始化 DeepSeek 模型。

        Args:
            client: 预配置的 OpenAI 兼容客户端
            client_args: 客户端参数
            include_reasoning_in_context: 是否在多轮对话中回传 reasoning_content。
                默认为 False。仅在工具调用场景需要设为 True。
            **model_config: 模型配置
        """
        super().__init__(client=client, client_args=client_args, **model_config)
        self.include_reasoning_in_context = include_reasoning_in_context

    @classmethod
    def _format_regular_messages(
        cls,
        messages: Messages,
        include_reasoning: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """格式化普通消息，支持可选的 reasoning_content 保留。

        覆盖父类方法，添加 include_reasoning 参数控制是否保留 reasoningContent。

        Args:
            messages: 消息列表
            include_reasoning: 是否保留 reasoningContent 内容块
            **kwargs: 额外参数

        Returns:
            格式化后的消息列表
        """
        formatted_messages = []

        for message in messages:
            contents = message["content"]

            # 检查是否有 reasoningContent
            has_reasoning = any("reasoningContent" in content for content in contents)

            if has_reasoning and not include_reasoning:
                # 父类行为：过滤 reasoningContent 并警告
                logger.warning(
                    "reasoningContent is not included in the context. "
                    "Set include_reasoning_in_context=True for tool call scenarios."
                )

            formatted_contents = [
                cls.format_request_message_content(content)
                for content in contents
                if not any(block_type in content for block_type in ["toolResult", "toolUse"])
                and (include_reasoning or "reasoningContent" not in content)
            ]
            formatted_tool_calls = [
                cls.format_request_message_tool_call(content["toolUse"])
                for content in contents
                if "toolUse" in content
            ]
            formatted_tool_messages = [
                cls.format_request_tool_message(content["toolResult"])
                for content in contents
                if "toolResult" in content
            ]

            formatted_message = {
                "role": message["role"],
                "content": formatted_contents,
                **({"tool_calls": formatted_tool_calls} if formatted_tool_calls else {}),
            }
            formatted_messages.append(formatted_message)

            # 处理 tool 消息，提取图片到独立的 user 消息
            for tool_msg in formatted_tool_messages:
                tool_msg_clean, user_msg_with_images = cls._split_tool_message_images(tool_msg)
                formatted_messages.append(tool_msg_clean)
                if user_msg_with_images:
                    formatted_messages.append(user_msg_with_images)

        return formatted_messages

    @classmethod
    def format_request_messages(
        cls,
        messages: Messages,
        system_prompt: str | None = None,
        *,
        system_prompt_content: list[SystemContentBlock] | None = None,
        include_reasoning_in_context: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """格式化 OpenAI 兼容的消息数组。

        Args:
            messages: 消息对象列表
            system_prompt: 系统提示词
            system_prompt_content: 系统提示词内容块
            include_reasoning_in_context: 是否保留 reasoning_content
            **kwargs: 额外参数

        Returns:
            OpenAI 兼容的消息数组
        """
        formatted_messages = cls._format_system_messages(
            system_prompt, system_prompt_content=system_prompt_content
        )
        formatted_messages.extend(
            cls._format_regular_messages(messages, include_reasoning=include_reasoning_in_context)
        )

        return [msg for msg in formatted_messages if msg.get("content") or "tool_calls" in msg]

    def _validate_reasoner_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """校验并清理 DeepSeek Reasoner 模型不支持的参数

        Args:
            params: 原始参数字典

        Returns:
            清理后的参数字典
        """
        model_id = self.config.get("model_id", "")
        if model_id != "deepseek-reasoner":
            return params

        cleaned_params = dict(params)

        # 检查并移除会报错的参数
        for param in ERROR_REASONER_PARAMS:
            if param in cleaned_params:
                logger.warning(
                    "DeepSeek Reasoner 不支持 '%s' 参数，已移除", param
                )
                del cleaned_params[param]

        # 检查无效参数
        for param in UNSUPPORTED_REASONER_PARAMS:
            if param in cleaned_params:
                logger.warning(
                    "DeepSeek Reasoner 不支持 '%s' 参数，设置无效", param
                )

        return cleaned_params

    def _get_cleaned_params(self) -> dict[str, Any]:
        """获取清理后的参数，对 Reasoner 模型移除不支持的参数

        Returns:
            清理后的参数字典
        """
        raw_params = self.config.get("params")
        params: dict[str, Any] = dict(raw_params) if isinstance(raw_params, dict) else {}

        # 对 Reasoner 模型进行参数校验
        if self.config.get("model_id") == "deepseek-reasoner":
            return self._validate_reasoner_params(params)

        return params

    def format_request(
        self,
        messages,
        tool_specs=None,
        system_prompt: str | None = None,
        tool_choice=None,
        *,
        system_prompt_content=None,
        **kwargs,
    ) -> dict[str, Any]:
        """格式化请求，支持 Reasoner 模型的参数校验

        对 deepseek-reasoner 模型:
        - 移除会报错的参数 (logprobs, top_logprobs)
        - 警告无效参数 (temperature, top_p 等)
        """
        from typing import cast

        formatted_messages = self._format_request_messages_instance(
            messages, system_prompt, system_prompt_content=system_prompt_content
        )

        return {
            "messages": [msg for msg in formatted_messages if msg.get("content") or "tool_calls" in msg],
            "model": self.config["model_id"],
            "stream": True,
            "stream_options": {"include_usage": True},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_spec["name"],
                        "description": tool_spec["description"],
                        "parameters": tool_spec["inputSchema"]["json"],
                    },
                }
                for tool_spec in tool_specs or []
            ],
            **(self._format_request_tool_choice(tool_choice)),
            **cast(dict[str, Any], self._get_cleaned_params()),
        }

    def _format_request_messages_instance(
        self,
        messages: Messages,
        system_prompt: str | None = None,
        *,
        system_prompt_content: list[SystemContentBlock] | None = None,
    ) -> list[dict[str, Any]]:
        """实例方法版本，使用实例的 include_reasoning_in_context 设置。"""
        return self._format_system_messages(
            system_prompt, system_prompt_content=system_prompt_content
        ) + self._format_regular_messages(
            messages, include_reasoning=self.include_reasoning_in_context
        )


# 预配置的 DeepSeek 模型实例 (方便直接使用)
def create_deepseek_model(
    api_key: str,
    model_id: str = "deepseek-chat",
    **kwargs
) -> DeepSeekOpenAIModel:
    """创建 DeepSeek 模型实例

    Args:
        api_key: DeepSeek API key
        model_id: 模型 ID，默认为 "deepseek-chat"
        **kwargs: 其他参数传递给 DeepSeekOpenAIModel

    Returns:
        DeepSeekOpenAIModel 实例
    """
    return DeepSeekOpenAIModel(
        client_args={
            "api_key": api_key,
            "base_url": "https://api.deepseek.com",
        },
        model_id=model_id,
        **kwargs
    )


# 预配置的 DeepSeek Reasoner 模型实例
def create_deepseek_reasoner(
    api_key: str,
    include_reasoning_in_context: bool = True,
    **kwargs
) -> DeepSeekOpenAIModel:
    """创建 DeepSeek Reasoner 模型实例 (Thinking Mode)

    注意: Reasoner 模型不支持 temperature, top_p 等参数

    Args:
        api_key: DeepSeek API key
        include_reasoning_in_context: 是否在多轮对话中回传 reasoning_content。
            默认为 True（与工具调用场景兼容）。普通对话可设为 False。
        **kwargs: 其他参数传递给 DeepSeekOpenAIModel (建议只设置 max_tokens)

    Returns:
        DeepSeekOpenAIModel 实例 (model_id="deepseek-reasoner")
    """
    # 检查并警告不支持的参数
    params = kwargs.get("params", {})
    for param in UNSUPPORTED_REASONER_PARAMS | ERROR_REASONER_PARAMS:
        if param in params:
            logger.warning(
                "DeepSeek Reasoner 不支持 '%s' 参数，已自动忽略", param
            )

    return DeepSeekOpenAIModel(
        client_args={
            "api_key": api_key,
            "base_url": "https://api.deepseek.com",
        },
        model_id="deepseek-reasoner",
        include_reasoning_in_context=include_reasoning_in_context,
        **kwargs
    )


# 向后兼容：DeepSeekModel 是 DeepSeekOpenAIModel 的别名
DeepSeekModel = DeepSeekOpenAIModel
