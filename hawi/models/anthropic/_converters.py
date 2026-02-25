"""
Anthropic 内容转换器

处理 ContentPart 到 Anthropic API 格式的转换，包括：
- 文本、图片、文档转换
- Tool call 和 tool result 转换
- 远程图片异步下载
- Cache control 支持
"""

from __future__ import annotations

import base64
import logging
from typing import Any, cast

import httpx

from hawi.agent.message import (
    CacheControlPart,
    CacheControl,
    ContentPart,
    DocumentPart,
    DocumentSource,
    GuardContentPart,
    ImagePart,
    Message,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from ._utils import SUPPORTED_IMAGE_TYPES, parse_data_uri

logger = logging.getLogger(__name__)


class ContentConverter:
    """同步内容转换器"""

    def __init__(self, enable_image_download: bool = True):
        self.enable_image_download = enable_image_download

    def convert_content(
        self, content: list[ContentPart]
    ) -> list[dict[str, Any]]:
        """将 ContentPart 列表转换为 Anthropic 格式

        特殊处理：
        - CacheControlPart 会被粘合到前一个内容 Part 上
        """
        result: list[dict[str, Any]] = []
        pending_cache_control: CacheControl | None = None

        for part in content:
            # 处理 CacheControlPart：缓存并应用到下一个内容 Part
            if part["type"] == "cache_control":
                part = cast(CacheControlPart, part)
                pending_cache_control = part["cache_control"]
                continue

            converted = self.convert_single_part(part)
            if converted:
                # 应用待处理的 cache_control
                if pending_cache_control:
                    converted["cache_control"] = pending_cache_control
                    pending_cache_control = None
                result.append(converted)

        # 如果最后一个 part 是 cache_control（没有内容可以粘合），忽略它
        # 或者可以选择报错，但通常这种情况是用户错误
        return result

    def convert_single_part(self, part: ContentPart) -> dict[str, Any] | None:
        """转换单个 ContentPart"""
        p_type = part["type"]

        if p_type == "text":
            return self._convert_text(cast(TextPart, part))
        elif p_type == "image":
            return self._convert_image(cast(ImagePart, part))
        elif p_type == "document":
            return self._convert_document(cast(DocumentPart, part))
        elif p_type == "tool_call":
            return self._convert_tool_call(cast(ToolCallPart, part))
        elif p_type == "tool_result":
            return self._convert_tool_result(cast(ToolResultPart, part))
        elif p_type == "reasoning":
            return self._convert_reasoning(cast(ReasoningPart, part))
        elif p_type == "guard_content":
            return self._convert_guard_content(cast(GuardContentPart, part))
        elif p_type == "cache_control":
            # CacheControlPart 在 convert_content 中处理，这里返回 None
            return None

        return None

    def convert_message(
        self, message: Message
    ) -> dict[str, Any] | None:
        """将通用消息转换为 Anthropic 格式"""
        role = message["role"]

        # Anthropic 只支持 user/assistant 角色
        if role == "tool":
            return self._convert_tool_message(message)

        content = self.convert_content(message["content"])
        
        # 对于 assistant 消息，还需要处理 tool_calls
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if tool_calls:
                # 将 tool_calls 转换为 tool_use 块并添加到 content
                for tc in tool_calls:
                    tool_use_block = self._convert_tool_call(tc)
                    content.append(tool_use_block)
        
        if not content:
            return None

        return {"role": role, "content": content}

    def _convert_tool_message(
        self, message: Message
    ) -> dict[str, Any] | None:
        """将 tool 消息转换为包含 tool_result 的 user 消息"""
        tool_call_id = message.get("tool_call_id")
        content_parts = []

        for part in message["content"]:
            if part["type"] == "tool_result":
                result_part = cast(ToolResultPart, part)
                item: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": result_part["tool_call_id"],
                    "content": self._convert_tool_result_content(
                        result_part["content"]
                    ),
                }
                if result_part.get("is_error"):
                    item["is_error"] = True
                content_parts.append(item)
            elif tool_call_id:
                converted = self.convert_content([part])
                content_parts.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": converted,
                })
            else:
                converted = self.convert_content([part])
                content_parts.extend(converted)

        if not content_parts:
            return None

        return {"role": "user", "content": content_parts}

    def _convert_text(self, part: TextPart) -> dict[str, Any]:
        """转换文本部分"""
        return {
            "type": "text",
            "text": part["text"],
        }

    def _convert_image(self, part: ImagePart) -> dict[str, Any]:
        """转换图片部分（同步版本）"""
        source = part["source"]
        url = source["url"]

        if url.startswith("data:"):
            media_type, data = parse_data_uri(url)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            }

        # 远程 URL - 同步模式下报错或降级
        if self.enable_image_download:
            raise RuntimeError(
                "Remote image URL requires async mode. "
                "Use ainvoke() or enable auto-async in invoke()."
            )
        logger.warning(f"Remote image URL not supported in sync mode: {url}")
        return {"type": "text", "text": f"[Image: {url}]"}

    def _convert_document(self, part: DocumentPart) -> dict[str, Any]:
        """转换文档部分（强制 base64）"""

        source = cast(DocumentSource, part["source"])
        url = source["url"]

        if not url.startswith("data:"):
            raise ValueError(
                "Anthropic API requires base64 encoded documents (data URI). "
                f"Received URL: {url[:50]}... "
                "Please encode the document as a data URI."
            )

        media_type, data = parse_data_uri(url)
        doc_item: dict[str, Any] = {
            "type": "document",
            "source": {
                "type": "base64" if media_type != "text/plain" else "text",
                "media_type": media_type or source.get("mime_type", "application/pdf"),
                "data": data,
            },
        }

        if part.get("title"):
            doc_item["title"] = part["title"]
        if part.get("context"):
            doc_item["context"] = part["context"]

        return doc_item

    def _convert_tool_call(self, part: ToolCallPart) -> dict[str, Any]:
        """转换 tool call"""
        return {
            "type": "tool_use",
            "id": part["id"],
            "name": part["name"],
            "input": part["arguments"],
        }

    def _convert_tool_result(self, part: ToolResultPart) -> dict[str, Any]:
        """转换 tool result"""
        result: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": part["tool_call_id"],
            "content": self._convert_tool_result_content(part["content"]),
        }
        if part.get("is_error"):
            result["is_error"] = True
        return result

    def _convert_reasoning(self, part: ReasoningPart) -> dict[str, Any] | None:
        """转换 reasoning/thinking"""
        # 处理 redacted_content（Anthropic 加密的安全推理内容）
        redacted = part.get("redacted_content")
        if redacted:
            return {
                "type": "redacted_thinking",
                "data": redacted.decode("utf-8", errors="replace"),
            }

        return {
            "type": "thinking",
            "thinking": part.get("reasoning") or "",
            "signature": part.get("signature"),
        }

    def _convert_guard_content(self, part: GuardContentPart) -> dict[str, Any]:
        """转换 guard_content（Anthropic Guardrails）"""
        return {
            "type": "guard_content",
            "guard_content": {
                "text": {
                    "text": part["text"],
                    "qualifiers": part.get("qualifiers", ["guard_content"]),
                }
            },
        }

    def _convert_tool_result_content(
        self, content: str | list[Any]
    ) -> str | list[dict[str, Any]]:
        """转换 tool result 内容"""
        if isinstance(content, str):
            return content
        return self.convert_content(content)


class AsyncContentConverter(ContentConverter):
    """异步内容转换器（支持远程图片下载）"""

    async def convert_content_async(
        self, content: list[ContentPart]
    ) -> list[dict[str, Any]]:
        """异步转换 ContentPart 列表

        特殊处理：
        - CacheControlPart 会被粘合到前一个内容 Part 上
        """
        result: list[dict[str, Any]] = []
        pending_cache_control: CacheControl | None = None

        for part in content:
            # 处理 CacheControlPart：缓存并应用到下一个内容 Part
            if part["type"] == "cache_control":
                part = cast(CacheControlPart, part)
                pending_cache_control = part["cache_control"]
                continue

            converted = await self.convert_single_part_async(part)
            if converted:
                # 应用待处理的 cache_control
                if pending_cache_control:
                    converted["cache_control"] = pending_cache_control
                    pending_cache_control = None
                result.append(converted)

        return result

    async def convert_single_part_async(
        self, part: ContentPart
    ) -> dict[str, Any] | None:
        """异步转换单个 ContentPart"""
        p_type = part["type"]

        if p_type == "image":
            return await self._convert_image_async(cast(ImagePart, part))
        elif p_type == "document":
            return self._convert_document(cast(DocumentPart, part))

        # 其他类型使用同步版本
        return self.convert_single_part(part)

    async def convert_message_async(
        self, message: Message
    ) -> dict[str, Any] | None:
        """异步转换消息"""
        role = message["role"]

        if role == "tool":
            return self._convert_tool_message(message)

        content = await self.convert_content_async(message["content"])
        
        # 对于 assistant 消息，还需要处理 tool_calls
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            if tool_calls:
                # 将 tool_calls 转换为 tool_use 块并添加到 content
                for tc in tool_calls:
                    tool_use_block = self._convert_tool_call(tc)
                    content.append(tool_use_block)
        
        if not content:
            return None

        return {"role": role, "content": content}

    async def _convert_image_async(self, part: ImagePart) -> dict[str, Any]:
        """转换图片（异步版本，支持下载）"""
        source = part["source"]
        url = source["url"]

        if url.startswith("data:"):
            media_type, data = parse_data_uri(url)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            }

        # 远程 URL - 异步下载
        return await self._download_image(part)

    async def _download_image(self, part: ImagePart) -> dict[str, Any]:
        """下载远程图片并转为 base64"""
        url = part["source"]["url"]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()

            content_type = (
                response.headers.get("content-type", "image/jpeg").split(";")[0]
            )
            if content_type not in SUPPORTED_IMAGE_TYPES:
                logger.warning(f"Unsupported image type: {content_type}")
                return {"type": "text", "text": f"[Image: {url}]"}

            data = base64.b64encode(response.content).decode("utf-8")
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": content_type,
                    "data": data,
                },
            }

        except Exception as e:
            logger.warning(f"Failed to download image from {url}: {e}")
            return {"type": "text", "text": f"[Image: {url}]"}


def needs_async_conversion(
    messages: list[Message], enable_image_download: bool
) -> bool:
    """检查是否需要异步处理（有远程图片）"""
    if not enable_image_download:
        return False

    for msg in messages:
        for part in msg.get("content", []):
            if part["type"] == "image":
                url = part.get("source", {}).get("url", "")
                if url and not url.startswith("data:"):
                    return True
    return False
