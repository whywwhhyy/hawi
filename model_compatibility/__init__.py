"""
模型兼容性模块

为不同 LLM API 提供兼容性适配，处理格式差异。

支持的模型:
- DeepSeek: 通过 DeepSeekModel 类支持 tool calling 和消息格式转换

使用示例:
    from model_compatibility import DeepSeekModel

    model = DeepSeekModel(
        api_key="your-api-key",
        model_id="deepseek-chat",
    )
    agent = Agent(model=model, ...)
"""

from .deepseek import DeepSeekModel, create_deepseek_model
from .kimi_openai import KimiOpenAIModel, create_kimi_model
from .kimi_anthropic import KimiAnthropicModel

__all__ = [
    "DeepSeekModel",
    "create_deepseek_model",
    "KimiOpenAIModel",
    "create_kimi_model",
    "KimiAnthropicModel",
]
