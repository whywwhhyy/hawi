"""
消息抽象层核心类型定义

使用 TypedDict 实现 Tagged Union 设计，支持完整的类型检查。
"""

from typing import Any, Literal, Required, TypeAlias, TypedDict, cast

from pydantic import BaseModel


# =============================================================================
# ContentPart 类型 - 消息内容的最小单元
# =============================================================================


class CacheControl(TypedDict):
    """Prompt caching 控制（Anthropic 支持）"""
    type: Literal["ephemeral"]


class TokenUsage(BaseModel):
    """Token 使用统计"""

    input_tokens: int
    output_tokens: int
    cache_write_tokens: int | None = None  # Prompt caching: tokens written to cache
    cache_read_tokens: int | None = None  # Prompt caching: tokens read from cache


class TextPart(TypedDict):
    """文本内容"""

    type: Literal["text"]
    text: str


class ImageSource(TypedDict):
    """图片来源"""

    url: str  # http URL 或 data:image/xxx;base64,... data URI
    detail: Literal["auto", "low", "high"] | None  # OpenAI 的 detail 参数


class ImagePart(TypedDict):
    """图片内容"""

    type: Literal["image"]
    source: ImageSource


class DocumentSource(TypedDict):
    """文档来源"""

    url: str  # http URL 或 base64 data URI
    mime_type: str | None  # e.g., "application/pdf"


class DocumentPart(TypedDict):
    """文档内容"""

    type: Literal["document"]
    source: DocumentSource
    title: str | None
    context: str | None


class ToolCallPart(TypedDict):
    """工具调用"""

    type: Literal["tool_call"]
    id: str  # 工具调用唯一标识
    name: str  # 工具名称
    arguments: dict[str, Any]  # 参数（已解析的 dict，非 JSON 字符串）


class ToolResultPart(TypedDict):
    """工具调用结果"""

    type: Literal["tool_result"]
    tool_call_id: str  # 对应 ToolCallPart.id
    content: str | list["ContentPart"]  # 结果内容（支持多模态）
    is_error: bool | None  # 是否错误（Anthropic 支持）


class ReasoningPart(TypedDict, total=False):
    """推理/思考内容"""

    type: Required[Literal["reasoning"]]
    reasoning: str | None  # 普通推理文本
    signature: str | None  # Anthropic 的验证签名
    redacted_content: bytes | None  # Anthropic 加密的安全推理内容


class CacheControlPart(TypedDict):
    """
    Prompt caching 控制标记（Anthropic 支持）

    设计说明：
    - cache_control 作为独立 Part，而非嵌入到内容 Part 中
    - 这样内容 Part 可以保持严格的 TypedDict 定义（无 total=False）
    - 模型适配层在转换时负责将 CacheControlPart 与前一个内容 Part 粘合

    使用示例：
        content = [
            text_part("Long document content..."),
            cache_control_part(),  # 标记前一个内容应用 caching
        ]
    """

    type: Literal["cache_control"]
    cache_control: CacheControl


# =============================================================================
# 扩展 ContentPart 类型 - 支持更多 LLM API 功能
# 详见: todo/message-abstraction-gap-analysis.md
# =============================================================================


class AudioSource(TypedDict, total=False):
    """
    音频数据抽象

    通用音频数据源，支持输入/输出双向音频，不绑定特定 API。
    字段根据使用场景（输入或输出）选择性填充。

    输入音频示例：
        AudioSource(url="data:audio/wav;base64,...", format="wav")

    输出音频示例（模型生成）：
        AudioSource(
            id="audio_xxx",
            data="base64encoded...",
            format="wav",
            transcript="Hello!",
            metadata={"expires_at": 1234567890}
        )
    """

    # 核心字段
    data: str  # base64 编码的音频数据（优先于 url）
    url: str  # 数据 URI 或 http URL 或外部引用 ID
    format: Literal["wav", "mp3", "flac", "opus", "pcm16"]  # 音频编码格式

    # 元数据（输出音频常见）
    id: str  # 服务端音频资源唯一标识
    transcript: str  # 音频转录文本（便于不支持音频的模型共享上下文）
    metadata: dict[str, Any]  # 扩展元数据（过期时间等）


class AudioPart(TypedDict):
    """音频内容"""

    type: Literal["audio"]
    source: AudioSource


class VideoSource(TypedDict):
    """视频来源 (Strands)"""

    url: str  # data URI: data:video/mp4;base64,...
    format: Literal["mp4", "mov", "webm", "mkv", "avi", "flv", "mpeg", "mpg", "three_gp", "wmv"]


class VideoPart(TypedDict):
    """视频内容 (Strands)"""

    type: Literal["video"]
    source: VideoSource


class FileSource(TypedDict):
    """文件来源 (OpenAI File API)"""

    file_id: str  # OpenAI File API 返回的 file_id
    filename: str | None


class FilePart(TypedDict):
    """文件内容引用 (OpenAI File API)"""

    type: Literal["file"]
    source: FileSource


class RefusalPart(TypedDict):
    """拒绝内容 (OpenAI) - 当模型拒绝生成内容时返回"""

    type: Literal["refusal"]
    refusal: str


class GuardContentPart(TypedDict):
    """Guardrails 内容安全评估 (Anthropic)"""

    type: Literal["guard_content"]
    text: str
    qualifiers: list[Literal["grounding_source", "query", "guard_content"]]


# ContentPart 联合类型
ContentPart: TypeAlias = (
    TextPart | ImagePart | DocumentPart | AudioPart | VideoPart | FilePart |
    ToolCallPart | ToolResultPart | ReasoningPart | CacheControlPart | RefusalPart | GuardContentPart
)


# =============================================================================
# StreamPart 类型 - Model 流式输出的增量内容块
# =============================================================================
# StreamPart 与 ContentPart 字段结构保持对应关系：
# - text_delta.delta 对应 TextPart.text
# - tool_call_delta.arguments_delta 对应 ToolCallPart.arguments（JSON片段）
# - StreamPart 包含增量标记（is_start, is_end, index），ContentPart 包含完整数据


class StreamTextPart(TypedDict):
    """文本增量块"""

    type: Literal["text_delta"]
    index: int              # 内容块序号，用于区分多个内容块
    delta: str              # 文本增量
    is_start: bool          # 是否是该块的开始
    is_end: bool            # 是否是该块的结束


class StreamThinkingPart(TypedDict):
    """推理/思考增量块"""

    type: Literal["thinking_delta"]
    index: int
    delta: str              # 推理内容增量
    is_start: bool
    is_end: bool


class StreamToolCallPart(TypedDict):
    """工具调用增量块"""

    type: Literal["tool_call_delta"]
    index: int
    id: str | None          # 工具调用ID（is_start 时可能为 None）
    name: str | None        # 工具名称（is_start 时可能为 None）
    arguments_delta: str    # 参数JSON片段
    is_start: bool
    is_end: bool


class StreamFinishPart(TypedDict):
    """流式响应结束标记"""

    type: Literal["finish"]
    stop_reason: str
    usage: dict[str, int] | None  # {"input_tokens": 100, "output_tokens": 50}


# StreamPart 联合类型
StreamPart: TypeAlias = StreamTextPart | StreamThinkingPart | StreamToolCallPart | StreamFinishPart


# =============================================================================
# Message 类型
# =============================================================================


class MessageMetadata(TypedDict, total=False):
    """消息元数据，用于上下文管理"""

    tokens: int  # 预计算的 token 数
    importance: float  # 重要性分数 (0-1)，用于压缩决策
    timestamp: float  # 创建时间戳
    compression_level: int  # 已被压缩的次数
    source: str  # 来源标识
    summarized: bool  # 是否已被摘要


class Message(TypedDict):
    """
    通用消息格式

    关键设计：
    - content 始终为 list[ContentPart]，简化处理逻辑
    - 构造函数接受 str | list[ContentPart]，自动规范化
    - metadata 可选，用于上下文管理

    Role 设计：
    - user: 用户输入
    - assistant: AI 响应
    - tool: 工具调用结果

    注意：系统提示词通过 MessageRequest.system 传递，不在 messages 中使用 role=system
    """

    role: Literal["user", "assistant", "tool"]
    content: list[ContentPart]  # 始终为数组，内部统一

    # 以下字段仅在特定 role 下使用
    name: str | None  # 区分同名角色的不同参与者
    tool_calls: list[ToolCallPart] | None  # assistant role: 模型请求调用工具
    tool_call_id: str | None  # tool role: 对应 tool_calls 的 id

    # 元数据（可选，用于上下文管理）
    metadata: MessageMetadata | None


# =============================================================================
# 音频处理工具函数
# =============================================================================

def transcribe_audio(audio_source: AudioSource) -> str:
    """
    语音识别接口 - 将音频转换为文本

    TODO: 当前为占位实现，后续接入实际语音识别引擎（如 Whisper）

    Args:
        audio_source: 音频数据源

    Returns:
        识别出的文本内容，如果无法识别则返回提示信息
    """
    # 如果已经有转录文本，直接返回
    transcript = audio_source.get("transcript")
    if transcript:
        return transcript

    # TODO: 接入实际的语音识别引擎
    # 示例: return whisper_client.transcribe(audio_source["data"])

    return "[语音消息 - 暂不支持语音识别，请使用支持音频的模型]"


def convert_audio_part_to_text(part: AudioPart) -> TextPart:
    """
    将 AudioPart 转换为 TextPart

    用于不支持音频的模型，将音频降级为文本处理。
    如果音频包含转录文本则使用，否则调用语音识别接口。

    Args:
        part: 音频内容部分

    Returns:
        文本内容部分
    """
    source = part["source"]

    # 优先使用已有的转录文本
    transcript = source.get("transcript")
    if transcript:
        text = transcript
    else:
        # 调用语音识别接口
        text = transcribe_audio(source)

    return {"type": "text", "text": text}


def downgrade_audio_content(content: list[ContentPart]) -> list[ContentPart]:
    """
    将内容中的 AudioPart 降级为 TextPart

    用于不支持音频输入的模型，在请求转换前调用。

    Args:
        content: 原始内容列表

    Returns:
        处理后的内容列表（所有 AudioPart 被替换为 TextPart）
    """
    result: list[ContentPart] = []

    for part in content:
        if part["type"] == "audio":
            # 将音频降级为文本
            result.append(convert_audio_part_to_text(part))
        elif part["type"] == "tool_result":
            # 递归处理 tool_result 中的内容
            tool_part = cast(ToolResultPart, part)
            tool_content = tool_part.get("content")
            if isinstance(tool_content, list):
                new_part: ToolResultPart = {
                    "type": "tool_result",
                    "tool_call_id": tool_part["tool_call_id"],
                    "content": downgrade_audio_content(tool_content),
                    "is_error": tool_part.get("is_error"),
                }
                result.append(new_part)
            else:
                result.append(part)
        else:
            result.append(part)

    return result


def downgrade_messages_audio(messages: list[Message]) -> list[Message]:
    """
    将消息列表中的所有 AudioPart 降级为 TextPart

    用于不支持音频输入的模型，在请求转换前调用。

    Args:
        messages: 原始消息列表

    Returns:
        处理后的消息列表
    """
    result: list[Message] = []

    for msg in messages:
        # 复制消息并处理 content
        new_msg: Message = {
            "role": msg["role"],
            "content": downgrade_audio_content(msg["content"]),
            "name": msg.get("name"),
            "tool_calls": msg.get("tool_calls"),
            "tool_call_id": msg.get("tool_call_id"),
            "metadata": msg.get("metadata"),
        }
        result.append(new_msg)

    return result


# =============================================================================
# 请求/响应类型
# =============================================================================


class ToolDefinition(TypedDict):
    """工具定义"""

    type: Literal["function","mcp"]
    name: str
    description: str
    schema: dict[str, Any]  # JSON Schema


class ToolChoice(TypedDict):
    """工具选择"""

    type: Required[Literal["none", "auto", "any", "tool"]]
    name: str | None  # type="tool" 时指定工具名


class ResponseFormat(TypedDict):
    """响应格式配置（用于结构化输出）"""

    type: Literal["text", "json_object", "json_schema"]
    json_schema: dict[str, Any] | None  # type="json_schema" 时使用


class MessageRequest(BaseModel):
    """请求消息容器"""

    messages: list[Message]
    system: list[ContentPart] | None = None  # 系统提示词（ContentPart 列表）

    # 工具定义
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None  # 是否允许并行工具调用

    # 可选参数
    max_tokens: int | None = None  # 已弃用，推荐使用 max_completion_tokens
    max_completion_tokens: int | None = None  # 输出 token 预算
    temperature: float | None = None
    top_p: float | None = None

    # 结构化输出
    response_format: dict[str, Any] | None = None  # JSON mode / structured outputs

    # 推理模型参数
    reasoning_effort: Literal["low", "medium", "high"] | None = None  # o1, o3 等推理模型

    # 服务层级
    service_tier: Literal["auto", "default", "flex"] | None = None  # flex 模式

    # Anthropic-specific parameters
    top_k: int | None = None  # Anthropic top_k sampling
    stop_sequences: list[str] | None = None  # Anthropic stop sequences
    metadata: dict[str, Any] | None = None  # Anthropic metadata (e.g., user_id)


class MessageResponse(BaseModel):
    """响应消息容器"""

    id: str
    role: Literal["assistant"] = "assistant"
    content: list[ContentPart]
    stop_reason: str | None = None
    usage: TokenUsage | None = None
    reasoning_content: str | None = None  # DeepSeek/Kimi 思考内容
