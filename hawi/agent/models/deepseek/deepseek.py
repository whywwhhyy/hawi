"""
DeepSeek 统一入口

根据传入参数自动选择 OpenAI 格式或 Anthropic 格式的模型。
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from hawi.agent.model import Model
from hawi.agent.models._utils import detect_deepseek_api_type
from .deepseek_openai import DeepSeekOpenAIModel
from .deepseek_anthropic import DeepSeekAnthropicModel

logger = logging.getLogger(__name__)

__all__ = ["DeepSeekModel"]

# 默认端点
DEFAULT_OPENAI_URL = "https://api.deepseek.com"
DEFAULT_ANTHROPIC_URL = "https://api.deepseek.com/anthropic"


class DeepSeekModel:
    """
    DeepSeek 模型统一入口

    根据传入的 url 和 api 参数自动选择使用 OpenAI 格式还是 Anthropic 格式的 API。

    Args:
        model_id: 模型标识符，如 "deepseek-chat", "deepseek-reasoner"
        api_key: API 密钥
        base_url: API 基础 URL，默认根据 api 参数决定
        api: API 格式选择，"auto" 根据 URL 自动检测，"openai" 使用 OpenAI 格式，"anthropic" 使用 Anthropic 格式
        **params: 其他参数传递给具体的模型类

    Example:
        # 自动检测（默认使用 OpenAI 格式）
        model = DeepSeekModel(
            model_id="deepseek-chat",
            api_key="sk-...",
        )

        # 明确指定 OpenAI 格式
        model = DeepSeekModel(
            model_id="deepseek-chat",
            api_key="sk-...",
            api="openai",
        )

        # 明确指定 Anthropic 格式
        model = DeepSeekModel(
            model_id="deepseek-reasoner",
            api_key="sk-...",
            api="anthropic",
        )

        # 使用自定义端点（自动检测）
        model = DeepSeekModel(
            model_id="deepseek-chat",
            api_key="sk-...",
            base_url="https://api.deepseek.com/anthropic",
            # api="auto" 会检测到 anthropic 路径
        )
    """

    def __new__(
        cls,
        *,
        model_id: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str | None = None,
        api: Literal["auto", "openai", "anthropic"] = "auto",
        **params: Any,
    ) -> Model:
        """
        创建 DeepSeek 模型实例

        根据 api 参数和 base_url 自动选择合适的模型类。
        """
        # 确定 API 类型
        if api == "auto":
            detected_api = detect_deepseek_api_type(base_url)
            logger.debug(f"Auto-detected API type: {detected_api} for URL: {base_url}")
        else:
            detected_api = api

        # 使用默认端点（如果未提供）
        if base_url is None:
            base_url = (
                DEFAULT_ANTHROPIC_URL
                if detected_api == "anthropic"
                else DEFAULT_OPENAI_URL
            )

        # 创建对应的模型实例
        if detected_api == "anthropic":
            logger.debug(f"Creating DeepSeekAnthropicModel with base_url={base_url}")
            return DeepSeekAnthropicModel(
                model_id=model_id,
                api_key=api_key,
                base_url=base_url,
                **params,
            )
        else:
            logger.debug(f"Creating DeepSeekOpenAIModel with base_url={base_url}")
            return DeepSeekOpenAIModel(
                model_id=model_id,
                api_key=api_key,
                base_url=base_url,
                **params,
            )
