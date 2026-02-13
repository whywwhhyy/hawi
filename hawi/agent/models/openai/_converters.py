"""
OpenAI 消息格式转换器

提供通用消息格式与 OpenAI API 格式之间的转换函数。
所有函数均为纯函数，无状态依赖。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from hawi.agent.messages import (
    ContentPart,
    Message,
    MessageRequest,
    ToolDefinition,
    ToolChoice,
)

logger = logging.getLogger(__name__)


def prepare_request(
    request: MessageRequest,
    model_id: str,
    params: dict[str, Any],
    converter=None,
) -> dict[str, Any]:
    """将通用请求转换为 OpenAI 格式

    Args:
        request: 通用消息请求
        model_id: 模型标识符
        params: 额外的模型参数 (temperature, max_tokens 等)
        converter: 可选的消息转换函数，默认为 convert_message_to_openai

    Returns:
        OpenAI API 格式的请求字典
    """
    if converter is None:
        converter = convert_message_to_openai

    openai_messages = [converter(m) for m in request.messages]

    # 将 system_prompt 转换为第一条 system/developer 消息
    # o1/o3 系列模型使用 "developer" 角色，其他使用 "system"
    if request.system:
        system_role = "developer" if model_id.startswith(("o1", "o3")) else "system"
        system_content = convert_content_to_openai(request.system)
        openai_messages.insert(0, {"role": system_role, "content": system_content})

    req: dict[str, Any] = {
        "model": model_id,
        "messages": openai_messages,
    }

    # 添加工具定义
    if request.tools:
        req["tools"] = [convert_tool_definition(t) for t in request.tools]

    # 添加工具选择
    if request.tool_choice:
        req["tool_choice"] = convert_tool_choice(request.tool_choice)

    # 添加请求参数（request 中的值优先于 params）
    # 注意：request 中的值已由基类 _build_request 合并过 params 和 override_params
    # 这里只需要应用 request 中的显式设置，其余参数通过 req.update 补充
    request_params = {
        "max_tokens": request.max_tokens,
        "max_completion_tokens": request.max_completion_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "parallel_tool_calls": request.parallel_tool_calls,
        "response_format": request.response_format,
        "reasoning_effort": request.reasoning_effort,
        "service_tier": request.service_tier,
    }

    # 先应用传入的 params（作为默认值）
    req.update(params)

    # 然后应用 request 中显式设置的值（覆盖 params）
    for key, value in request_params.items():
        if value is not None:
            req[key] = value

    # 处理 max_completion_tokens 优先级：如果设置了 max_completion_tokens，优先使用它
    # OpenAI 推荐优先使用 max_completion_tokens 替代 max_tokens
    if request.max_completion_tokens is not None:
        req["max_completion_tokens"] = request.max_completion_tokens
        # 如果同时设置了 max_tokens，移除它以避免冲突
        if "max_tokens" in req and request.max_tokens is None:
            # 只有当 max_tokens 来自 params 而非 request 时才移除
            pass  # 保留两者，让 API 决定

    return req


def convert_tool_definition(tool: ToolDefinition) -> dict[str, Any]:
    """转换工具定义为 OpenAI 格式

    Args:
        tool: 通用工具定义 (扁平格式: name, description, schema)

    Returns:
        OpenAI 格式的工具定义 (嵌套 function 格式)
    """
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["schema"],
        },
    }


def convert_tool_choice(tool_choice: ToolChoice) -> str | dict[str, Any]:
    """转换工具选择为 OpenAI 格式

    Args:
        tool_choice: 通用工具选择

    Returns:
        OpenAI 格式的 tool_choice
    """
    tc_type = tool_choice.get("type")

    if tc_type == "none":
        return "none"
    elif tc_type == "auto":
        return "auto"
    elif tc_type == "any":
        # OpenAI API 使用 "required" 来强制调用任意工具
        return "required"
    elif tc_type == "tool" and tool_choice.get("name"):
        return {
            "type": "function",
            "function": {"name": tool_choice["name"]},
        }

    # 默认 fallback
    logger.warning("Unknown tool_choice type: %s, using 'auto'", tc_type)
    return "auto"


def convert_message_to_openai(message: Message) -> dict[str, Any]:
    """将通用消息转换为 OpenAI 格式

    Args:
        message: 通用消息

    Returns:
        OpenAI 格式的消息字典
    """
    role = message["role"]

    if role == "tool":
        return convert_tool_message(message)

    # OpenAI 支持 system 角色
    content = convert_content_to_openai(message["content"])
    result: dict[str, Any] = {"role": role, "content": content}

    if role == "assistant" and message.get("tool_calls"):
        result["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"]),
                },
            }
            for tc in (message["tool_calls"] or ())
        ]

    if message.get("name"):
        result["name"] = message["name"]

    return result


def convert_tool_message(message: Message) -> dict[str, Any]:
    """转换 tool 角色消息为 OpenAI 格式

    OpenAI API 要求 tool 消息的 content 为字符串格式。

    Args:
        message: tool 角色消息

    Returns:
        OpenAI 格式的 tool 消息
    """
    content = serialize_content(message["content"])
    return {
        "role": "tool",
        "tool_call_id": message.get("tool_call_id") or "",
        "content": content,
    }


def convert_content_to_openai(
    content: list[ContentPart],
) -> str | list[dict[str, Any]]:
    """将 ContentPart 列表转换为 OpenAI 内容格式

    Args:
        content: 内容部分列表

    Returns:
        字符串或 OpenAI 内容块列表
    """
    if len(content) == 1 and content[0]["type"] == "text":
        return content[0]["text"]

    openai_content: list[dict[str, Any]] = []
    for part in content:
        if part["type"] == "text":
            openai_content.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image":
            source = part["source"]
            openai_content.append({
                "type": "image_url",
                "image_url": {
                    "url": source["url"],
                    "detail": source.get("detail") or "auto",
                },
            })
        elif part["type"] == "document":
            source = part["source"]
            title = part.get("title") or "Document"
            openai_content.append({
                "type": "text",
                "text": f"[{title}: {source['url']}]",
            })

    return openai_content if openai_content else ""


def convert_openai_content_to_part(
    part: dict[str, Any]
) -> list[ContentPart]:
    """将 OpenAI 内容块转换为 ContentPart 列表

    Args:
        part: OpenAI 内容块

    Returns:
        ContentPart 列表
    """
    p_type = part.get("type")

    if p_type == "text":
        return [{"type": "text", "text": part.get("text", "")}]
    elif p_type == "image_url":
        image_url = part.get("image_url", {})
        return [
            {
                "type": "image",
                "source": {
                    "url": image_url.get("url", ""),
                    "detail": image_url.get("detail"),
                },
            }
        ]

    return []


def serialize_content(content: list[ContentPart]) -> str:
    """将 ContentPart 列表序列化为字符串

    Args:
        content: 内容部分列表

    Returns:
        序列化后的字符串
    """
    texts = []
    for part in content:
        if part["type"] == "text":
            texts.append(part["text"])
        elif part["type"] == "image":
            texts.append(f"[Image: {part['source']['url']}]")
        else:
            texts.append(str(part))
    return "\n".join(texts) if texts else ""


def split_tool_message_images(
    tool_message: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """将 tool 消息中的图片提取到独立的 user 消息

    OpenAI API 限制图片只能出现在 user 角色消息中。

    Args:
        tool_message: 格式化后的 tool 消息

    Returns:
        (无图片的 tool 消息, 包含图片的 user 消息或 None)
    """
    if tool_message.get("role") != "tool":
        return tool_message, None

    content = tool_message.get("content", [])
    if not isinstance(content, list):
        return tool_message, None

    # 分离图片和非图片内容
    text_content = []
    image_content = []

    for item in content:
        if isinstance(item, dict) and item.get("type") == "image_url":
            image_content.append(item)
        else:
            text_content.append(item)

    # 如果没有图片，返回原始消息
    if not image_content:
        return tool_message, None

    logger.warning(
        "tool_call_id=%s | 将图片从 tool 消息移到 user 消息以兼容 OpenAI",
        tool_message["tool_call_id"],
    )

    # 在文本内容中添加提示
    text_content.append({
        "type": "text",
        "text": "Tool returned an image. See the following user message for the image.",
    })

    # 创建干净的 tool 消息
    tool_message_clean = {
        "role": "tool",
        "tool_call_id": tool_message["tool_call_id"],
        "content": text_content,
    }

    # 创建包含图片的 user 消息
    user_message_with_images = {"role": "user", "content": image_content}

    return tool_message_clean, user_message_with_images


def map_stop_reason(reason: str) -> str | None:
    """映射 OpenAI 停止原因到通用格式

    Args:
        reason: OpenAI 停止原因

    Returns:
        通用格式的停止原因
    """
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "content_filter",
    }
    return mapping.get(reason, reason)
