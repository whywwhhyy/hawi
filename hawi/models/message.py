"""
消息抽象层核心类型定义

使用 TypedDict 实现 Tagged Union 设计，支持完整的类型检查。
"""

from typing import Any, Sequence, Literal, Required, NotRequired, TypeAlias, TypedDict, cast

from pydantic import BaseModel


# =============================================================================
# Token Usage 类型
# =============================================================================

class TokenUsage(TypedDict):
    """Token 使用统计"""

    input_tokens: int
    output_tokens: int
    cache_write_tokens: NotRequired[int | None]  # Prompt caching: tokens written to cache
    cache_read_tokens: NotRequired[int | None]  # Prompt caching: tokens read from cache


# =============================================================================
# ContentPart 类型 - 消息内容的最小单元
# =============================================================================


# ContentPart 的 type 字段字面量类型
# 用于 ContentPart 联合类型中各个成员的 type 字段
ContentPartType = Literal[
    "text",
    "image",
    "document",
    "audio",
    "video",
    "file",
    "tool_call",
    "tool_result",
    "reasoning",
    "cache_control",
    "refusal",
    "guard_content",
    "citation",
]


# 流式内容块类型 - 事件系统中使用的块类型
# 注意：这是 ContentPartType 的子集，用于流式块处理
DeltaPartType = Literal["text", "thinking", "tool_use", "redacted_thinking"]


class CacheControl(TypedDict):
    """Prompt caching 控制（Anthropic 支持）"""
    type: Literal["ephemeral"]


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


# =============================================================================
# Citation 类型 - Anthropic 引用内容
# =============================================================================
# Anthropic 支持多种引用位置类型：
# - char_location: 字符位置（PDF 等文档）
# - page_location: 页面位置（PDF）
# - content_block_location: 内容块位置
# - web_search_result_location: Web 搜索结果
# - search_result_location: 搜索结果


class CitationLocationBase(TypedDict, total=False):
    """Citation 位置基类 - 所有引用类型的公共字段"""

    cited_text: str  # 被引用的文本
    document_index: int | None  # 文档索引
    document_title: str | None  # 文档标题
    file_id: str | None  # 文件 ID
    title: str | None  # 标题（用于搜索结果）


class CitationCharLocation(CitationLocationBase):
    """字符位置引用 (Anthropic)

    用于引用 PDF 等文档中的特定字符范围。
    """

    type: Literal["char_location"]
    start_char_index: int  # 起始字符索引
    end_char_index: int    # 结束字符索引


class CitationPageLocation(CitationLocationBase):
    """页面位置引用 (Anthropic)

    用于引用 PDF 等文档中的特定页面。
    """

    type: Literal["page_location"]
    start_page_number: int  # 起始页码
    end_page_number: int    # 结束页码


class CitationContentBlockLocation(CitationLocationBase):
    """内容块位置引用 (Anthropic)

    用于引用消息中的特定内容块。
    """

    type: Literal["content_block_location"]
    start_block_index: int  # 起始块索引
    end_block_index: int    # 结束块索引


class CitationsWebSearchResultLocation(TypedDict):
    """Web 搜索结果引用 (Anthropic)

    用于引用 Web 搜索结果。
    """

    type: Literal["web_search_result_location"]
    cited_text: str  # 被引用的文本
    encrypted_index: str  # 加密的索引
    url: str  # 引用 URL
    title: str | None  # 标题


class CitationsSearchResultLocation(TypedDict):
    """搜索结果引用 (Anthropic)

    用于引用搜索结果。
    """

    type: Literal["search_result_location"]
    cited_text: str  # 被引用的文本
    source: str  # 来源
    search_result_index: int  # 搜索结果索引
    start_block_index: int  # 起始块索引
    end_block_index: int    # 结束块索引
    title: str | None  # 标题


# CitationLocation 联合类型 - 所有可能的引用位置类型
CitationLocation: TypeAlias = (
    CitationCharLocation |
    CitationPageLocation |
    CitationContentBlockLocation |
    CitationsWebSearchResultLocation |
    CitationsSearchResultLocation
)


class CitationPart(TypedDict):
    """引用内容 (Anthropic/Kimi)

    模型输出中引用的来源信息。
    作为完整内容传递（非流式增量）。
    """

    type: Literal["citation"]
    citations: list[CitationLocation]  # 引用列表


# ContentPart 联合类型
ContentPart: TypeAlias = (
    TextPart | ImagePart | DocumentPart | AudioPart | VideoPart | FilePart |
    ToolCallPart | ToolResultPart | ReasoningPart | CacheControlPart | 
    RefusalPart | GuardContentPart | CitationPart
)


# =============================================================================
# DeltaPart 类型 - Model 流式输出的增量内容块
# =============================================================================
# DeltaPart 与 ContentPart 字段结构保持对应关系：
# - text_delta.delta 对应 TextPart.text
# - tool_call_delta.arguments_delta 对应 ToolCallPart.arguments（JSON片段）
# - DeltaPart 包含增量标记（is_start, is_end, index），ContentPart 包含完整数据


class DeltaTextPart(TypedDict):
    """文本增量块"""

    type: Literal["text_delta"]
    index: int              # 内容块序号，用于区分多个内容块
    delta: str              # 文本增量
    is_start: bool          # 是否是该块的开始
    is_end: bool            # 是否是该块的结束


class DeltaThinkingPart(TypedDict):
    """推理/思考增量块"""

    type: Literal["thinking_delta"]
    index: int
    delta: str              # 推理内容增量
    is_start: bool
    is_end: bool


class DeltaSignaturePart(TypedDict):
    """推理签名增量块 (Anthropic)

    Thinking 块的签名增量，用于验证推理完整性。
    通常与 thinking_delta 配合使用。
    """

    type: Literal["signature_delta"]
    index: int
    delta: str              # 签名增量
    is_start: bool
    is_end: bool



class DeltaToolCallPart(TypedDict):
    """工具调用增量块"""

    type: Literal["tool_call_delta"]
    index: int
    id: str | None          # 工具调用ID（is_start 时可能为 None）
    name: str | None        # 工具名称（is_start 时可能为 None）
    arguments_delta: str    # 参数JSON片段
    is_start: bool
    is_end: bool


class DeltaMetadataPart(TypedDict):
    """元数据增量块 - 用于携带输出内容的元数据（如引用）

    Citation 是对模型输出的位置标注（字符位置、页面位置等），
    作为元数据在 Delta 流中传递，而非增量文本。

    示例：
        - citations: 引用列表（char_location, page_location 等）
    """

    type: Literal["metadata_delta"]
    index: int
    metadata: dict[str, Any]  # 元数据（如 {"citations": [...]}）
    # 字符位置（可选，None 表示 block 全文）
    start_char: int | None
    end_char: int | None
    is_start: bool
    is_end: bool


class DeltaFinishPart(TypedDict):
    """流式响应结束标记"""

    type: Literal["finish"]
    stop_reason: str
    usage: TokenUsage | dict[str, int] | None  # {"input_tokens": 100, "output_tokens": 50}


# DeltaPart 联合类型
# 注意：
# - refusal、guard_content 等非流式内容类型不在此处
# - 它们只在非流式响应的 ContentPart 中出现
# - Citation 作为元数据通过 DeltaMetadataPart 在流中传递
DeltaPart: TypeAlias = (
    DeltaTextPart | 
    DeltaThinkingPart | 
    DeltaSignaturePart |
    DeltaToolCallPart | 
    DeltaMetadataPart |
    DeltaFinishPart
)


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
    - tool_calls 作为 ToolCallPart 存储在 content 中

    Role 设计：
    - user: 用户输入
    - assistant: AI 响应（包含 ToolCallPart 表示 tool_calls）
    - tool: 工具调用结果（包含 ToolResultPart，其中带有 tool_call_id）

    注意：系统提示词通过 MessageRequest.system 传递，不在 messages 中使用 role=system
    """

    role: Literal["user", "assistant", "tool"]
    content: list[ContentPart]  # 始终为数组，包含 text/image/tool_call/tool_result 等

    # 以下字段仅在特定 role 下使用
    name: str | None  # 区分同名角色的不同参与者

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
    content: Sequence[ContentPart]
    # 内容元数据（如 citations）- 与 content 按索引对应
    content_metadata: list[ContentPart] | None = None
    stop_reason: str | None = None
    usage: TokenUsage | None = None
    reasoning_content: str | None = None  # DeepSeek/Kimi 思考内容

