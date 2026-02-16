"""
Kimi 统一入口

根据传入参数自动选择 OpenAI 格式或 Anthropic 格式的模型。
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from hawi.agent.model import Model
from hawi.agent.models._utils import detect_kimi_api_type
from .kimi_openai import KimiOpenAIModel
from .kimi_anthropic import KimiAnthropicModel

logger = logging.getLogger(__name__)

__all__ = ["KimiModel", "create_kimi_model"]

# 默认端点
DEFAULT_OPENAI_URL = "https://api.moonshot.cn/v1"
DEFAULT_ANTHROPIC_URL = "https://api.kimi.com/coding/"


class KimiModel:
    """
    Kimi 模型统一入口

    根据传入的 url 和 api 参数自动选择使用 OpenAI 格式还是 Anthropic 格式的 API。

    Args:
        model_id: 模型标识符，如 "kimi-k2", "kimi-k2.5"
        api_key: API 密钥
        base_url: API 基础 URL，默认根据 api 参数决定
        api: API 格式选择，"auto" 根据 URL 自动检测，"openai" 使用 OpenAI 格式，"anthropic" 使用 Anthropic 格式
        enable_thinking: 是否启用 thinking 模式（仅 K2.5，OpenAI 格式下有效）
        **params: 其他参数传递给具体的模型类

    Example:
        # 自动检测（默认使用 OpenAI/Moonshot 格式）
        model = KimiModel(
            model_id="kimi-k2.5",
            api_key="sk-...",
        )

        # 明确指定 OpenAI 格式（Moonshot 端点）
        model = KimiModel(
            model_id="kimi-k2.5",
            api_key="sk-...",
            api="openai",
        )

        # 明确指定 Anthropic 格式（Kimi 端点）
        model = KimiModel(
            model_id="kimi-k2.5",
            api_key="sk-...",
            api="anthropic",
        )

        # 使用自定义端点（自动检测）
        model = KimiModel(
            model_id="kimi-k2.5",
            api_key="sk-...",
            base_url="https://api.kimi.com/coding/",
            # api="auto" 会检测到 anthropic 格式
        )

        # K2.5 禁用 thinking 模式
        model = KimiModel(
            model_id="kimi-k2.5",
            api_key="sk-...",
            enable_thinking=False,
        )
    """

    def __new__(
        cls,
        *,
        model_id: str = "kimi-k2.5",
        api_key: str | None = None,
        base_url: str | None = None,
        api: Literal["auto", "openai", "anthropic"] = "auto",
        enable_thinking: bool = True,
        **params: Any,
    ) -> Model:
        """
        创建 Kimi 模型实例

        根据 api 参数和 base_url 自动选择合适的模型类。
        """
        # 验证 model_id 类型
        if not isinstance(model_id, str):
            raise TypeError(
                f"model_id must be a string, got {type(model_id).__name__}. "
                f"Did you pass a list instead of a single model ID?"
            )

        # 确定 API 类型
        if api == "auto":
            detected_api = detect_kimi_api_type(base_url)
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
            logger.debug(f"Creating KimiAnthropicModel with base_url={base_url}")
            # Anthropic 格式不支持 enable_thinking 参数
            return KimiAnthropicModel(
                model_id=model_id,
                api_key=api_key,
                base_url=base_url,
                **params,
            )
        else:
            logger.debug(f"Creating KimiOpenAIModel with base_url={base_url}")
            return KimiOpenAIModel(
                model_id=model_id,
                api_key=api_key,
                base_url=base_url,
                enable_thinking=enable_thinking,
                **params,
            )
