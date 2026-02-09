"""
模型兼容性模块

为不同 LLM API 提供兼容性适配，处理格式差异。

支持的模型:
- DeepSeek OpenAI: 通过 DeepSeekOpenAIModel 类支持 OpenAI 兼容 API
- DeepSeek Anthropic: 通过 DeepSeekAnthropicModel 类支持 Anthropic 兼容 API
- Kimi OpenAI: 通过 KimiOpenAIModel 类支持 OpenAI 兼容 API
- Kimi Anthropic: 通过 KimiAnthropicModel 类支持 Anthropic 兼容 API

使用示例:
    from hawi.agent.models import DeepSeekOpenAIModel

    model = DeepSeekOpenAIModel(
        client_args={
            "api_key": "your-api-key",
            "base_url": "https://api.deepseek.com",
        },
        model_id="deepseek-chat",
    )
"""

from .deepseek_openai import DeepSeekOpenAIModel, create_deepseek_model
from .deepseek_anthropic import DeepSeekAnthropicModel, create_deepseek_anthropic_model
from .kimi_openai import KimiOpenAIModel, create_kimi_model
from .kimi_anthropic import KimiAnthropicModel

# 向后兼容：DeepSeekModel 是 DeepSeekOpenAIModel 的别名
DeepSeekModel = DeepSeekOpenAIModel

__all__ = [
    # DeepSeek OpenAI API
    "DeepSeekOpenAIModel",
    "create_deepseek_model",
    # DeepSeek Anthropic API
    "DeepSeekAnthropicModel",
    "create_deepseek_anthropic_model",
    # 向后兼容
    "DeepSeekModel",
    # Kimi OpenAI API
    "KimiOpenAIModel",
    "create_kimi_model",
    # Kimi Anthropic API
    "KimiAnthropicModel",
]
