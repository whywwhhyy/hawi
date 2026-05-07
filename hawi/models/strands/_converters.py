"""
Content and message conversion utilities for Strands model adapter.
"""

import json
import logging
from typing import Any, Sequence, cast

from hawi.models import (
    AudioPart,
    ContentPart,
    DocumentPart,
    ImagePart,
    Message,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
    VideoPart,
)

logger = logging.getLogger(__name__)


def _convert_messages_to_strands(
    messages: list[Message],
) -> list[dict[str, Any]]:
    """
    Convert Hawi Message list to Strands format.

    Args:
        messages: List of Hawi messages

    Returns:
        List of Strands format messages
    """
    strands_messages = []

    for msg in messages:
        strands_msg = _convert_single_message_to_strands(msg)
        strands_messages.append(strands_msg)

    return strands_messages


def _convert_single_message_to_strands(msg: Message) -> dict[str, Any]:
    """
    Convert single Hawi Message to Strands format.

    Args:
        msg: Hawi message

    Returns:
        Strands format message dict
    """
    role = msg["role"]

    # Strands uses role, content format
    strands_msg: dict[str, Any] = {"role": role}

    # Convert content
    if msg["content"]:
        strands_content = _convert_content_to_strands(msg["content"])
        strands_msg["content"] = strands_content

    # Handle tool_calls (assistant role) - extract from content as ToolCallPart
    if role == "assistant" and msg["content"]:
        tool_call_parts = [p for p in msg["content"] if p.get("type") == "tool_call"]
        if tool_call_parts:
            strands_msg["tool_calls"] = [
                _convert_tool_call_part_to_strands(cast(ToolCallPart, tc))
                for tc in tool_call_parts
            ]

    # Handle tool_call_id (tool role) - get from ToolResultPart in content
    if role == "tool" and msg["content"]:
        for part in msg["content"]:
            if part.get("type") == "tool_result":
                tool_call_id = cast(ToolResultPart, part).get("tool_call_id")
                if tool_call_id:
                    strands_msg["tool_call_id"] = tool_call_id
                break

    # Handle name
    if msg.get("name"):
        strands_msg["name"] = msg["name"]

    return strands_msg


def _convert_content_to_strands(
    content: Sequence[ContentPart],
) -> list[dict[str, Any]]:
    """
    Convert Hawi ContentPart list to Strands ContentBlock list.

    Args:
        content: Sequence of Hawi content parts

    Returns:
        List of Strands format content blocks
    """
    strands_content = []

    for part in content:
        block = _convert_part_to_strands_block(part)
        if block:
            strands_content.append(block)

    return strands_content


def _convert_part_to_strands_block(part: ContentPart) -> dict[str, Any] | None:
    """
    Convert single ContentPart to Strands ContentBlock.

    Args:
        part: Hawi content part

    Returns:
        Strands format content block dict, or None if not supported
    """
    p_type = part.get("type")

    if p_type == "text":
        return {"text": part.get("text", "")}
    elif p_type == "image":
        part = cast(ImagePart, part)
        return {
            "image": {
                "url": part["source"]["url"],
                "detail": part["source"].get("detail"),
            }
        }
    elif p_type == "document":
        part = cast(DocumentPart, part)
        return {
            "document": {
                "url": part["source"]["url"],
                "mime_type": part["source"].get("mime_type"),
                "title": part.get("title"),
                "context": part.get("context"),
            }
        }
    elif p_type == "tool_call":
        part = cast(ToolCallPart, part)
        return {
            "toolUse": {
                "toolUseId": part["id"],
                "name": part["name"],
                "input": part["arguments"],
            }
        }
    elif p_type == "tool_result":
        part = cast(ToolResultPart, part)
        return {
            "toolResult": {
                "toolUseId": part["tool_call_id"],
                "content": part["content"],
                "is_error": part.get("is_error"),
            }
        }
    elif p_type == "reasoning":
        part = cast(ReasoningPart, part)
        return {
            "reasoningContent": {
                "reasoningText": {
                    "text": part.get("reasoning") or "",
                    "signature": part.get("signature"),
                }
            }
        }
    elif p_type in {"cache_point", "cache_control"}:
        # Strands may not support explicit cache points, skip or convert
        logger.debug("Cache point marker skipped in strands conversion")
        return None
    elif p_type == "video":
        part = cast(VideoPart, part)
        return {
            "video": {
                "source": {
                    "bytes": part["source"].get("data", ""),
                },
                "format": part["source"].get("format", "mp4"),
            }
        }
    elif p_type == "audio":
        part = cast(AudioPart, part)
        source = part["source"]
        # Strands audio format: prefer data, otherwise use url
        audio_data = source.get("data") or source.get("url") or ""
        return {
            "audio": {
                "source": {"bytes": audio_data},
                "format": source.get("format", "wav"),
            }
        }

    logger.warning(f"Unknown content part type: {p_type}")
    return None


def _convert_strands_block_to_part(block: dict[str, Any]) -> ContentPart | None:
    """
    Convert Strands ContentBlock to Hawi ContentPart.

    Args:
        block: Strands content block dict

    Returns:
        Hawi content part, or None if unknown type
    """
    if "text" in block:
        return {"type": "text", "text": block["text"]}
    elif "image" in block:
        image = block["image"]
        return {
            "type": "image",
            "source": {
                "url": image.get("url", ""),
                "detail": image.get("detail"),
            },
        }
    elif "document" in block:
        doc = block["document"]
        return {
            "type": "document",
            "source": {
                "url": doc.get("url", ""),
                "mime_type": doc.get("mime_type"),
            },
            "title": doc.get("title"),
            "context": doc.get("context"),
        }
    elif "reasoningContent" in block:
        # Strands reasoningContent format
        reasoning = block["reasoningContent"]
        # Handle redacted_content (encrypted secure reasoning content)
        if "redactedContent" in reasoning:
            redacted_data = reasoning["redactedContent"]
            if isinstance(redacted_data, str):
                redacted_bytes = redacted_data.encode("utf-8")
            else:
                redacted_bytes = redacted_data
            return cast(ReasoningPart, {
                "type": "reasoning",
                "reasoning": None,
                "signature": None,
                "redacted_content": redacted_bytes,
            })
        if "reasoningText" in reasoning:
            text = reasoning["reasoningText"].get("text", "")
            signature = reasoning["reasoningText"].get("signature")
        else:
            text = reasoning.get("text", "")
            signature = reasoning.get("signature")
        return cast(ReasoningPart, {
            "type": "reasoning",
            "reasoning": text,
            "signature": signature,
            "redacted_content": None,
        })
    elif "toolUse" in block:
        # Strands toolUse is also part of content blocks
        return _convert_strands_tool_use_to_part(block["toolUse"])
    elif "toolResult" in block:
        # toolResult block
        tool_result = block["toolResult"]
        return {
            "type": "tool_result",
            "tool_call_id": tool_result.get("toolUseId", ""),
            "content": tool_result.get("content", ""),
            "is_error": tool_result.get("status") == "error",
        }
    elif "video" in block:
        # Strands video content
        video = block["video"]
        source = video.get("source", {})
        video_data = source.get("bytes", "")
        return cast(VideoPart, {
            "type": "video",
            "source": {
                "data": video_data if isinstance(video_data, str) else "",
                "format": video.get("format", "mp4"),
            },
        })
    elif "audio" in block:
        # Strands audio content
        audio = block["audio"]
        source = audio.get("source", {})
        audio_data = source.get("bytes", "")
        return cast(AudioPart, {
            "type": "audio",
            "source": {
                "data": audio_data if isinstance(audio_data, str) else "",
                "format": audio.get("format", "wav"),
            },
        })

    logger.debug(f"Unknown strands block: {block.keys()}")
    return None


def _convert_tool_definition_to_strands(
    tool: ToolDefinition,
) -> dict[str, Any]:
    """
    Convert Hawi ToolDefinition to Strands ToolSpec.

    Args:
        tool: Hawi tool definition

    Returns:
        Strands format tool spec dict
    """
    return {
        "name": tool["name"],
        "description": tool["description"],
        "inputSchema": {
            "json": tool["schema"],
        },
    }


def _convert_strands_tool_use_to_part(tool_use: dict[str, Any]) -> ToolCallPart:
    """
    Convert Strands toolUse to Hawi ToolCallPart.

    Args:
        tool_use: Strands toolUse dict

    Returns:
        Hawi tool call part
    """
    # Parse arguments
    input_data = tool_use.get("input", {})
    if isinstance(input_data, str):
        try:
            input_data = json.loads(input_data)
        except json.JSONDecodeError:
            input_data = {}

    return {
        "type": "tool_call",
        "id": tool_use.get("toolUseId", ""),
        "name": tool_use.get("name", ""),
        "arguments": input_data,
    }


def _convert_tool_call_part_to_strands(part) -> dict[str, Any]:
    """
    Convert Hawi ToolCallPart to Strands toolUse format.

    Args:
        part: Hawi tool call part

    Returns:
        Strands format toolUse dict
    """
    return {
        "toolUse": {
            "toolUseId": part["id"],
            "name": part["name"],
            "input": part["arguments"],
        }
    }


def _convert_tool_choice_to_strands(
    tool_choice: ToolChoice,
) -> dict[str, Any]:
    """
    Convert Hawi ToolChoice to Strands ToolChoice.

    Args:
        tool_choice: Hawi tool choice

    Returns:
        Strands format tool choice dict
    """
    tc_type = tool_choice.get("type", "auto")

    mapping = {
        "none": {"type": "none"},
        "auto": {"type": "auto"},
        "any": {"type": "any"},
        "tool": {"type": "tool", "name": tool_choice.get("name", "")},
    }

    return mapping.get(tc_type, {"type": "auto"})
