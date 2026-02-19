"""
Models integration tests for Hawi Agent.

API Key configuration (priority order):
1. Environment variables (DEEPSEEK_API_KEY, KIMI_API_KEY, MINIMAX_API_KEY)
2. apikey.yaml file in project root
"""

from test.integration import (
    get_deepseek_api_key,
    get_kimi_openai_api_key,
    get_kimi_anthropic_api_key,
    get_minimax_api_key,
)

__all__ = [
    "get_deepseek_api_key",
    "get_kimi_openai_api_key",
    "get_kimi_anthropic_api_key",
    "get_minimax_api_key",
]
