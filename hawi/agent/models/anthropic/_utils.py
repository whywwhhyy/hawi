"""
Anthropic 模型辅助工具函数
"""

import base64
import logging
from collections.abc import Mapping, Sequence
from typing import Any,cast

from hawi.agent.messages import (
    Message,
    ContentPart,
    TextPart
)

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}


def parse_data_uri(data_uri: str) -> tuple[str, str]:
    """解析 data URI，返回 (media_type, base64_data)

    Args:
        data_uri: Data URI 字符串，如 "data:image/jpeg;base64,/9j/4AAQ..."

    Returns:
        (media_type, base64_data) 元组
    """
    if "," not in data_uri:
        return ("application/octet-stream", data_uri)

    header, data = data_uri.split(",", 1)
    parts = header.replace("data:", "").split(";")
    media_type = parts[0] if parts else "application/octet-stream"
    return (media_type, data)


def map_stop_reason(reason: str | None) -> str | None:
    """映射 Anthropic 停止原因到通用格式

    Args:
        reason: Anthropic 停止原因

    Returns:
        通用格式的停止原因
    """
    mapping = {
        "end_turn": "end_turn",
        "max_tokens": "max_tokens",
        "stop_sequence": "stop_sequence",
        "tool_use": "tool_use",
        "pause_turn": "pause_turn",
    }
    if reason is None:
        return None
    return mapping.get(reason, reason)


def convert_system_prompt(
    system: list[ContentPart] | None,
) -> str | list[dict[str, Any]] | None:
    """将 system_prompt 转换为 Anthropic API 的 system 字段格式

    Anthropic API 支持两种 system 格式：
    - 简单字符串（单段纯文本）
    - 对象数组（多段，或包含 cache_control）

    Args:
        system: ContentPart 列表

    Returns:
        字符串、对象数组或 None
    """
    if not system:
        return None

    # 转换为 Anthropic system 块列表
    blocks: list[dict[str, Any]] = []
    for part in system:
        if part.get("type") == "text":
            blocks.append({"type": "text", "text": cast(TextPart, part)["text"]})
        elif part.get("type") == "cache_control":
            # cache_control 标记，附加到前一个块
            if blocks:
                blocks[-1]["cache_control"] = {"type": "ephemeral"}
        # Anthropic system 字段只支持文本和 cache_control，忽略其他类型

    if not blocks:
        return None

    # 只有一段且无 cache_control 时返回字符串，否则返回数组
    if len(blocks) == 1 and "cache_control" not in blocks[0]:
        return blocks[0]["text"]

    return blocks
