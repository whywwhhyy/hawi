"""
模型工具函数

提供 API 类型检测等公共工具。
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "detect_api_type",
    "detect_deepseek_api_type",
    "detect_kimi_api_type",
    # DeepSeek 官方端点
    "DEEPSEEK_OPENAI_URL",
    "DEEPSEEK_ANTHROPIC_URL",
    # Kimi 官方端点
    "KIMI_OPENAI_URL",
    "KIMI_ANTHROPIC_URL",
]

# =============================================================================
# 官方 API 端点
# =============================================================================

# DeepSeek 官方端点
DEEPSEEK_OPENAI_URL = "https://api.deepseek.com"
DEEPSEEK_ANTHROPIC_URL = "https://api.deepseek.com/anthropic"

# Kimi/Moonshot 官方端点
KIMI_OPENAI_URL = "https://api.moonshot.cn/v1"
KIMI_ANTHROPIC_URL = "https://api.kimi.com/coding/"


def _normalize_url(url: str) -> str:
    """标准化 URL（去除末尾斜杠并转为小写）"""
    return url.rstrip("/").lower()


def detect_api_type(
    url: str | None,
    *,
    official_urls: dict[str, str] | None = None,
    provider_hints: dict[str, str] | None = None,
) -> Literal["openai", "anthropic"]:
    """
    根据 URL 自动检测 API 类型（OpenAI 格式或 Anthropic 格式）

    Args:
        url: API 基础 URL
        official_urls: 官方 URL 全字符串匹配表
            支持以下 key:
            - "openai": OpenAI 格式的官方 URL
            - "anthropic": Anthropic 格式的官方 URL
        provider_hints: 可选的提供商特定提示，用于增强检测准确性
            支持以下 key:
            - "anthropic_path": Anthropic 格式的路径特征（如 "anthropic", "coding"）
            - "openai_path": OpenAI 格式的路径特征（如 "v1", "openai"）
            - "provider_domain": 提供商域名特征（如 "kimi.com", "moonshot.cn"）

    Returns:
        "openai" 或 "anthropic"

    Example:
        # 基础检测
        api_type = detect_api_type("https://api.deepseek.com/anthropic")
        # Returns: "anthropic"

        # 带官方 URL 匹配的检测
        api_type = detect_api_type(
            "https://api.deepseek.com",
            official_urls={
                "openai": "https://api.deepseek.com",
                "anthropic": "https://api.deepseek.com/anthropic",
            }
        )

        # 带提示的检测
        api_type = detect_api_type(
            "https://custom.kimi.com/coding/",
            provider_hints={
                "anthropic_path": "coding",
                "provider_domain": "kimi.com",
            }
        )
        # Returns: "anthropic"
    """
    if url is None:
        return "openai"

    url_normalized = _normalize_url(url)
    url_lower = url.lower()

    # 1. 全字符串匹配官方 URL（最高优先级）
    if official_urls:
        if "openai" in official_urls:
            if url_normalized == _normalize_url(official_urls["openai"]):
                return "openai"
        if "anthropic" in official_urls:
            if url_normalized == _normalize_url(official_urls["anthropic"]):
                return "anthropic"

    # 2. 检查明确的 Anthropic 路径特征
    anthropic_indicators = ["anthropic"]
    if provider_hints and "anthropic_path" in provider_hints:
        anthropic_indicators.append(provider_hints["anthropic_path"].lower())

    for indicator in anthropic_indicators:
        if indicator in url_lower:
            return "anthropic"

    # 3. 检查提供商特定域名（Kimi 使用 Anthropic 格式）
    if provider_hints and "provider_domain" in provider_hints:
        domain = provider_hints["provider_domain"].lower()
        if domain in url_lower:
            # Kimi 的域名通常对应 Anthropic 格式
            return "anthropic"

    # 4. 检查明确的 OpenAI 路径特征
    openai_indicators = ["moonshot.cn", "openai"]  # moonshot 使用 OpenAI 格式
    if provider_hints and "openai_path" in provider_hints:
        openai_indicators.append(provider_hints["openai_path"].lower())

    for indicator in openai_indicators:
        if indicator in url_lower:
            return "openai"

    # 默认使用 OpenAI 格式
    return "openai"


def detect_deepseek_api_type(url: str | None) -> Literal["openai", "anthropic"]:
    """
    DeepSeek 特定的 API 类型检测

    DeepSeek 端点:
    - OpenAI: https://api.deepseek.com
    - Anthropic: https://api.deepseek.com/anthropic
    """
    return detect_api_type(
        url,
        official_urls={
            "openai": DEEPSEEK_OPENAI_URL,
            "anthropic": DEEPSEEK_ANTHROPIC_URL,
        },
        provider_hints={
            "anthropic_path": "anthropic",
        },
    )


def detect_kimi_api_type(url: str | None) -> Literal["openai", "anthropic"]:
    """
    Kimi/Moonshot 特定的 API 类型检测

    Kimi/Moonshot 端点:
    - OpenAI (Moonshot): https://api.moonshot.cn/v1
    - Anthropic (Kimi): https://api.kimi.com/coding/
    """
    return detect_api_type(
        url,
        official_urls={
            "openai": KIMI_OPENAI_URL,
            "anthropic": KIMI_ANTHROPIC_URL,
        },
        provider_hints={
            "anthropic_path": "coding",
            "provider_domain": "kimi.com",
            "openai_path": "moonshot.cn",
        },
    )
