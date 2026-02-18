"""
MiniMax 模型实现

提供 MiniMax M2.5/M2.1 等模型的支持，兼容 OpenAI 和 Anthropic API 格式。
"""

from .minimax import MiniMaxModel
from .minimax_openai import MiniMaxOpenAIModel
from .minimax_anthropic import MiniMaxAnthropicModel

__all__ = [
    "MiniMaxModel",
    "MiniMaxOpenAIModel",
    "MiniMaxAnthropicModel",
]
