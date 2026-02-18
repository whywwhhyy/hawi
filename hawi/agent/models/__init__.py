"""
Hawi Agent 模型实现

提供各 LLM 提供商的具体实现。

Example:
    from hawi.agent.models import OpenAIModel
    from hawi.agent.model import ModelConfig

    model = OpenAIModel(config=ModelConfig(
        model_id="gpt-4",
        api_key="..."
    ))
"""

from typing import Optional

from hawi.agent.model import BalanceInfo
from .openai import OpenAIModel
from .anthropic import AnthropicModel
from .deepseek import DeepSeekModel
from .kimi import KimiModel
from .minimax import MiniMaxModel
from .strands import StrandsModel

def get_model_class(name:str) -> Optional[type]:
    return {
        "OpenAIModel": OpenAIModel,
        "AnthropicModel": AnthropicModel,
        "DeepSeekModel": DeepSeekModel,
        "KimiModel": KimiModel,
        "MiniMaxModel": MiniMaxModel,
        "StrandsModel": StrandsModel,
    }.get(name)

__all__ = [
    "BalanceInfo",
    "OpenAIModel",
    "AnthropicModel",
    "DeepSeekModel",
    "KimiModel",
    "MiniMaxModel",
    "StrandsModel",
    "get_model_class",
]
