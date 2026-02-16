"""
消息抽象层核心类型定义

使用 TypedDict 实现 Tagged Union 设计，支持完整的类型检查。
"""

from typing import Any, List, Literal, Required, TypeAlias, TypedDict

from pydantic import BaseModel


# =============================================================================
# ContentPart 类型 - 消息内容的最小单元
# =============================================================================


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


class ReasoningPart(TypedDict):
    """推理/思考内容"""

    type: Literal["reasoning"]
    reasoning: str  # 推理过程文本
    signature: str | None  # Anthropic 的验证签名


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


# ContentPart 联合类型
ContentPart: TypeAlias = (
    TextPart | ImagePart | DocumentPart | ToolCallPart | ToolResultPart | ReasoningPart | CacheControlPart
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


class TokenUsage(BaseModel):
    """Token 使用统计"""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None  # Anthropic prompt caching
    cache_read_input_tokens: int | None = None


class MessageResponse(BaseModel):
    """响应消息容器"""

    id: str
    role: Literal["assistant"] = "assistant"
    content: list[ContentPart]
    stop_reason: str | None = None
    usage: TokenUsage | None = None
    reasoning_content: str | None = None  # DeepSeek/Kimi 思考内容
