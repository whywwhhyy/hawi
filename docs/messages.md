# 消息类型系统

Hawi 提供丰富的消息类型系统，支持文本、图片、文档、音频、视频、工具调用等多种内容形式。

## 概述

消息类型分为三个层次：

1. **ContentPart** - 消息内容的最小单元
2. **Message** - 包含角色和内容的消息
3. **MessageRequest/MessageResponse** - 模型交互的完整请求/响应

## ContentPart 类型

ContentPart 是消息内容的基本单元，使用 Tagged Union 设计。

### 基础内容类型

#### TextPart - 文本

```python
from hawi.models import TextPart

text_part: TextPart = {
    "type": "text",
    "text": "Hello, world!"
}
```

#### ImagePart - 图片

```python
from hawi.models import ImagePart, ImageSource

# URL 图片
image_part: ImagePart = {
    "type": "image",
    "source": {
        "url": "https://example.com/image.png",
        "detail": "high"  # "auto", "low", "high"
    }
}

# Base64 图片
image_part: ImagePart = {
    "type": "image",
    "source": {
        "url": "data:image/png;base64,iVBORw0KGgo...",
        "detail": "auto"
    }
}
```

#### DocumentPart - 文档

```python
from hawi.models import DocumentPart

doc_part: DocumentPart = {
    "type": "document",
    "source": {
        "url": "https://example.com/doc.pdf",
        "mime_type": "application/pdf"
    },
    "title": "Document Title",
    "context": "Optional context"
}
```

### 工具相关类型

#### ToolCallPart - 工具调用

```python
from hawi.models import ToolCallPart

tool_call: ToolCallPart = {
    "type": "tool_call",
    "id": "call_123",
    "name": "calculator",
    "arguments": {"expression": "1 + 1"}
}
```

#### ToolResultPart - 工具结果

```python
from hawi.models import ToolResultPart

# 成功结果
tool_result: ToolResultPart = {
    "type": "tool_result",
    "tool_call_id": "call_123",
    "content": "2",
    "is_error": False
}

# 错误结果
tool_result: ToolResultPart = {
    "type": "tool_result",
    "tool_call_id": "call_123",
    "content": "Error: Division by zero",
    "is_error": True
}

# 多模态结果（支持图片等）
tool_result: ToolResultPart = {
    "type": "tool_result",
    "tool_call_id": "call_456",
    "content": [
        {"type": "text", "text": "Generated image:"},
        {"type": "image", "source": {"url": "data:image/png;base64,..."}}
    ],
    "is_error": False
}
```

### 高级类型

#### ReasoningPart - 推理内容

用于展示模型的思考过程（DeepSeek Reasoner 等模型支持）。

```python
from hawi.models import ReasoningPart

reasoning: ReasoningPart = {
    "type": "reasoning",
    "reasoning": "Let me calculate this step by step...",
    "signature": None,  # Anthropic 验证签名
    "redacted_content": None  # 加密的推理内容
}
```

#### CacheControlPart - 提示缓存控制

Anthropic 模型的提示缓存功能。

```python
from hawi.models import CacheControlPart

cache_control: CacheControlPart = {
    "type": "cache_control",
    "cache_control": {"type": "ephemeral"}
}

# 使用示例：标记长文档应用缓存
content = [
    {"type": "text", "text": "Long document content..."},
    {"type": "cache_control", "cache_control": {"type": "ephemeral"}},
]
```

## Message 类型

Message 表示对话中的一条消息。

```python
from hawi.models import Message

# 用户消息
user_message: Message = {
    "role": "user",
    "content": [{"type": "text", "text": "Hello!"}],
    "name": None,      # 可选：标识发送者
    "metadata": None   # 可选：元数据（时间戳、token 数等）
}

# 助手消息
assistant_message: Message = {
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Hello! How can I help?"}
    ],
    "name": None,
    "metadata": None
}

# 工具消息
tool_message: Message = {
    "role": "tool",
    "content": [
        {
            "type": "tool_result",
            "tool_call_id": "call_123",
            "content": "2",
            "is_error": False
        }
    ],
    "name": None,
    "metadata": None
}
```

## MessageRequest

发送给模型的请求。

```python
from hawi.models import MessageRequest, ToolDefinition

request = MessageRequest(
    messages=[
        {"role": "user", "content": [{"type": "text", "text": "Hello!"}], "name": None, "metadata": None}
    ],
    system=[{"type": "text", "text": "You are a helpful assistant."}],
    tools=[
        ToolDefinition(
            type="function",
            name="calculator",
            description="Calculate math expressions",
            schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        )
    ]
)
```

## MessageResponse

模型的响应。

```python
from hawi.models import MessageResponse

response = MessageResponse(
    message={
        "role": "assistant",
        "content": [
            {"type": "text", "text": "The result is 2."}
        ],
        "name": None,
        "metadata": None
    },
    stop_reason="end_turn",
    usage={"input_tokens": 10, "output_tokens": 5}
)
```

## 流式 Delta 类型

流式响应使用 DeltaPart 类型表示增量更新。

```python
from hawi.models import DeltaTextPart, DeltaThinkingPart, DeltaToolCallPart

# 文本增量
text_delta: DeltaTextPart = {
    "type": "text_delta",
    "delta": "Hello",
    "index": 0,
    "is_start": True,
    "is_end": False
}

# 思考增量
reasoning_delta: DeltaThinkingPart = {
    "type": "reasoning_delta",
    "delta": "Let me think...",
    "index": 0,
    "is_start": True,
    "is_end": False,
    "signature": None
}

# 工具调用增量
tool_delta: DeltaToolCallPart = {
    "type": "tool_call_delta",
    "index": 0,
    "id": "call_123",
    "name": "calculator",
    "arguments_delta": '{"expr"',
    "is_start": True,
    "is_end": False
}
```

## 完整示例

```python
from hawi.models import (
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    ImagePart,
)

# 构建多模态对话
messages: list[Message] = [
    # 用户发送图片和文字
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image",
                "source": {
                    "url": "https://example.com/photo.jpg",
                    "detail": "high"
                }
            }
        ],
        "name": None,
        "metadata": None
    },
    # 助手调用工具
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me analyze the image."},
            {
                "type": "tool_call",
                "id": "call_1",
                "name": "image_analyzer",
                "arguments": {"url": "https://example.com/photo.jpg"}
            }
        ],
        "name": None,
        "metadata": None
    },
    # 工具返回结果
    {
        "role": "tool",
        "content": [
            {
                "type": "tool_result",
                "tool_call_id": "call_1",
                "content": "The image shows a cat sitting on a couch.",
                "is_error": False
            }
        ],
        "name": None,
        "metadata": None
    },
    # 助手最终回复
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "The image shows a cat sitting on a couch."}
        ],
        "name": None,
        "metadata": None
    }
]
```

## 类型导出

所有消息类型都可以从 `hawi.models` 导入：

```python
from hawi.models import (
    # 基础类型
    Message,
    MessageRequest,
    MessageResponse,
    ContentPart,
    ContentPartType,
    
    # 内容部件
    TextPart,
    ImagePart,
    ImageSource,
    DocumentPart,
    AudioPart,
    VideoPart,
    
    # 工具相关
    ToolCallPart,
    ToolResultPart,
    ToolDefinition,
    ToolChoice,
    
    # 高级类型
    ReasoningPart,
    CacheControlPart,
    
    # 流式类型
    DeltaPart,
    DeltaTextPart,
    DeltaThinkingPart,
    DeltaToolCallPart,
    
    # Token 统计
    TokenUsage,
)
```
