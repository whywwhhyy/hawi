"""
MiniMax 统一入口

根据传入参数自动选择 OpenAI 格式或 Anthropic 格式的模型。
使用专属的 MiniMaxOpenAIModel 和 MiniMaxAnthropicModel 处理 MiniMax 特有的格式差异。
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from hawi.agent.model import Model
from hawi.agent.models._utils import (
    detect_minimax_api_type,
    complete_url_and_api,
    MINIMAX_URL_API_CANDIDATES,
)
from .minimax_openai import MiniMaxOpenAIModel
from .minimax_anthropic import MiniMaxAnthropicModel

logger = logging.getLogger(__name__)

__all__ = ["MiniMaxModel"]

# 默认端点
DEFAULT_OPENAI_URL = "https://api.minimaxi.com/v1"
DEFAULT_ANTHROPIC_URL = "https://api.minimaxi.com/anthropic"


class MiniMaxModel:
    """
    MiniMax 模型统一入口

    根据传入的 base_url 和 api 参数自动选择使用 OpenAI 格式还是 Anthropic 格式的 API。
    使用专属的 MiniMaxOpenAIModel 和 MiniMaxAnthropicModel 处理 MiniMax 特有的格式差异。

    Args:
        model_id: 模型标识符，如 "MiniMax-M2.5", "MiniMax-M2.1"
        api_key: API 密钥
        base_url: API 基础 URL，默认根据 api 参数决定
        api: API 格式选择，"auto" 根据 URL 自动检测，"openai" 使用 OpenAI 格式，"anthropic" 使用 Anthropic 格式
        **params: 其他参数传递给具体的模型类

    Example:
        # 自动检测（默认使用 OpenAI 格式）
        model = MiniMaxModel(
            model_id="MiniMax-M2.5",
            api_key="sk-...",
        )

        # 明确指定 OpenAI 格式
        model = MiniMaxModel(
            model_id="MiniMax-M2.5",
            api_key="sk-...",
            api="openai",
        )

        # 明确指定 Anthropic 格式
        model = MiniMaxModel(
            model_id="MiniMax-M2.5",
            api_key="sk-...",
            api="anthropic",
        )

        # 使用自定义端点（自动检测）
        model = MiniMaxModel(
            model_id="MiniMax-M2.5",
            api_key="sk-...",
            base_url="https://api.minimaxi.com/anthropic",
            # api="auto" 会检测到 anthropic 路径
        )
    """

    def __new__(
        cls,
        *,
        model_id: str = "MiniMax-M2.5",
        api_key: str,
        base_url: str | None = None,
        api: Literal["auto", "openai", "anthropic"] = "auto",
        **params: Any,
    ) -> Model:
        """
        创建 MiniMax 模型实例

        根据 api 参数和 base_url 自动选择合适的模型类。
        支持参数补全：base_url 和 api 可以互相推导。
        """
        # 验证 model_id 类型
        if not isinstance(model_id, str):
            raise TypeError(
                f"model_id must be a string, got {type(model_id).__name__}. "
                f"Did you pass a list instead of a single model ID?"
            )

        # 处理 api="auto" 的情况
        query_api = None if api == "auto" else api

        # 使用补全函数确定 base_url 和 api
        # 如果两者都提供，不进行验证（validate=False）
        final_url, final_api = complete_url_and_api(
            base_url,
            query_api,
            MINIMAX_URL_API_CANDIDATES,
            validate=False if (base_url is not None and query_api is not None) else True,
        )
        
        # 如果用户提供了 base_url 但没有提供 api，需要根据 base_url 检测
        if api == "auto":
            detected_api = detect_minimax_api_type(base_url)
            logger.debug(f"Auto-detected API type: {detected_api} for URL: {base_url}")
        else:
            detected_api = final_api
        
        base_url = final_url

        # 创建对应的模型实例（使用专属的 MiniMax 模型类）
        if detected_api == "anthropic":
            logger.debug(f"Creating MiniMaxAnthropicModel with base_url={base_url}")
            return MiniMaxAnthropicModel(
                model_id=model_id,
                api_key=api_key,
                base_url=base_url,
                **params,
            )
        else:
            logger.debug(f"Creating MiniMaxOpenAIModel with base_url={base_url}")
            return MiniMaxOpenAIModel(
                model_id=model_id,
                api_key=api_key,
                base_url=base_url,
                **params,
            )
