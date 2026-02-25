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
    "detect_minimax_api_type",
    # DeepSeek 官方端点
    "DEEPSEEK_OPENAI_URL",
    "DEEPSEEK_ANTHROPIC_URL",
    # Kimi 官方端点
    "KIMI_OPENAI_URL",
    "KIMI_ANTHROPIC_URL",
    # MiniMax 官方端点
    "MINIMAX_OPENAI_URL",
    "MINIMAX_ANTHROPIC_URL",
    # 多元组补全
    "tuple_completion",
    "TupleCompletionError",
    # 参数补全
    "complete_url_and_api",
    "complete_model_and_thinking",
    # 候选列表
    "DEEPSEEK_URL_API_CANDIDATES",
    "KIMI_URL_API_CANDIDATES",
    "MINIMAX_URL_API_CANDIDATES",
    "DEEPSEEK_MODEL_THINKING_CANDIDATES",
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


def detect_minimax_api_type(url: str | None) -> Literal["openai", "anthropic"]:
    """
    MiniMax 特定的 API 类型检测

    MiniMax 端点:
    - OpenAI: https://api.minimaxi.com/v1
    - Anthropic: https://api.minimaxi.com/anthropic
    """
    return detect_api_type(
        url,
        official_urls={
            "openai": MINIMAX_OPENAI_URL,
            "anthropic": MINIMAX_ANTHROPIC_URL,
        },
        provider_hints={
            "anthropic_path": "anthropic",
            "provider_domain": "minimaxi.com",
        },
    )


class TupleCompletionError(ValueError):
    """多元组补全错误，当匹配结果不唯一时抛出"""
    pass


def tuple_completion(
    elements: tuple, 
    candidates: list[tuple], 
    *,
    validate: bool = True,
) -> tuple:
    """
    使用候选列表补全多元组中缺失的值（None）。
    
    匹配规则：逐一检查候选，非 None 位置必须精确相等，None 位置匹配任意值。
    要求最终必须恰好匹配一个候选，否则报错。
    
    Args:
        elements: 需要补全的多元组，缺失值用 None 表示
        candidates: 候选多元组列表
        validate: 是否验证完整元组。当为 False 且 elements 中没有 None 时，
                 直接返回 elements 而不检查是否在候选列表中
    
    Returns:
        补全后的多元组（从候选中找到的完整匹配，或原元组）
    
    Raises:
        TupleCompletionError: 当匹配结果为 0 个或多个时
    
    Example:
        >>> candidates = [
        ...     ("a", 1, "x"),
        ...     ("a", 2, "y"),
        ...     ("b", 3, "z"),
        ... ]
        >>> tuple_completion(("a", None, None), candidates)
        ('a', 1, 'x')  # 错误：两个匹配
        
        >>> tuple_completion(("b", 3, None), candidates)
        ('b', 3, 'z')  # 正确：唯一匹配
        
        >>> tuple_completion(("c", 4, "w"), candidates, validate=False)
        ('c', 4, 'w')  # 无 None 且不验证，直接返回
    """
    # 如果不需要验证且所有元素都已提供，直接返回
    if not validate and None not in elements:
        return elements
    
    if not candidates:
        raise TupleCompletionError(f"多元组补全失败：候选列表为空。输入: {elements}")
    
    # 检查维度一致性
    expected_len = len(elements)
    for i, cand in enumerate(candidates):
        if len(cand) != expected_len:
            raise ValueError(
                f"候选列表中第 {i} 个元素维度不匹配: "
                f"期望 {expected_len} 维，实际 {len(cand)} 维"
            )
    
    # 筛选匹配的候选
    matches = []
    for candidate in candidates:
        is_match = True
        for elem, cand_elem in zip(elements, candidate):
            if elem is not None and elem != cand_elem:
                is_match = False
                break
        if is_match:
            matches.append(candidate)
    
    # 验证匹配结果数量
    if len(matches) == 0:
        raise TupleCompletionError(
            f"多元组补全失败：没有找到匹配的候选。输入: {elements}"
        )
    elif len(matches) > 1:
        raise TupleCompletionError(
            f"多元组补全失败：找到 {len(matches)} 个匹配候选，期望恰好 1 个。"
            f"输入: {elements}, 匹配候选: {matches}"
        )
    
    return matches[0]

# =============================================================================
# 参数补全：base_url 与 api_type
# =============================================================================

# DeepSeek 的 base_url + api 候选
DEEPSEEK_URL_API_CANDIDATES: list[tuple[str | None, str | None]] = [
    ("https://api.deepseek.com", "openai"),
    ("https://api.deepseek.com/anthropic", "anthropic"),
]


# Kimi 的 base_url + api 候选
KIMI_URL_API_CANDIDATES: list[tuple[str | None, str | None]] = [
    ("https://api.moonshot.cn/v1", "openai"),
    ("https://api.kimi.com/coding/", "anthropic"),
]


# MiniMax 官方端点
MINIMAX_OPENAI_URL = "https://api.minimaxi.com/v1"
MINIMAX_ANTHROPIC_URL = "https://api.minimaxi.com/anthropic"


# MiniMax 的 base_url + api 候选
MINIMAX_URL_API_CANDIDATES: list[tuple[str | None, str | None]] = [
    ("https://api.minimaxi.com/v1", "openai"),
    ("https://api.minimaxi.com/anthropic", "anthropic"),
]


def complete_url_and_api(
    base_url: str | None,
    api: str | None,
    candidates: list[tuple[str | None, str | None]],
    *,
    validate: bool = True,
    default_on_none: bool = True,
) -> tuple[str, str]:
    """
    补全 base_url 和 api_type。

    根据候选列表补全缺失的 base_url 或 api。如果 validate=False 且两者都已指定，
    直接返回原值而不验证。

    Args:
        base_url: API 基础 URL，可为 None
        api: API 类型（"openai" 或 "anthropic"），可为 None
        candidates: 候选 (base_url, api) 列表，None 表示通配
        validate: 是否验证完整参数。当为 False 且 base_url 和 api 都提供时，
                 直接返回而不检查是否在候选列表中
        default_on_none: 当 base_url 和 api 都为 None 时，是否使用第一个候选作为默认值

    Returns:
        (补全后的 base_url, 补全后的 api)

    Raises:
        TupleCompletionError: 当无法唯一确定补全结果时

    Example:
        >>> candidates = [
        ...     ("https://api.example.com", "openai"),
        ...     ("https://api.example.com/anthropic", "anthropic"),
        ... ]
        >>> complete_url_and_api(None, "openai", candidates)
        ("https://api.example.com", "openai")

        >>> complete_url_and_api("https://api.example.com", None, candidates)
        ("https://api.example.com", "openai")

        >>> complete_url_and_api("custom", "openai", candidates, validate=False)
        ("custom", "openai")  # 不验证，直接返回
    """
    # 当两者都为 None 且允许使用默认值时，直接返回第一个候选
    if default_on_none and base_url is None and api is None:
        if not candidates:
            raise TupleCompletionError("候选列表为空，无法提供默认值")
        first_candidate = candidates[0]
        if first_candidate[0] is None or first_candidate[1] is None:
            raise TupleCompletionError(f"第一个候选包含 None: {first_candidate}")
        return first_candidate  # type: ignore
    
    query = (base_url, api)
    result = tuple_completion(query, candidates, validate=validate)

    # 确保返回的都是非 None 值
    final_url, final_api = result
    if final_url is None or final_api is None:
        raise TupleCompletionError(
            f"参数补全失败：补全结果包含 None。输入: ({base_url}, {api}), "
            f"结果: ({final_url}, {final_api})"
        )

    return (final_url, final_api)


# =============================================================================
# 参数补全：model_id 与 thinking
# =============================================================================

# DeepSeek 模型与 thinking 支持候选
DEEPSEEK_MODEL_THINKING_CANDIDATES: list[tuple[str | None, bool | None]] = [
    ("deepseek-reasoner", True),   # reasoner 支持 thinking
    ("deepseek-chat", False),       # chat 不支持 thinking
]


def complete_model_and_thinking(
    model_id: str | None,
    thinking: bool | None,
    candidates: list[tuple[str | None, bool | None]],
    *,
    validate: bool = True,
) -> tuple[str, bool]:
    """
    补全 model_id 和 thinking 参数。

    根据候选列表补全缺失的 model_id 或 thinking。如果 validate=False 且两者都已指定，
    直接返回原值而不验证。

    Args:
        model_id: 模型标识符，可为 None
        thinking: 是否启用 thinking 模式，可为 None
        candidates: 候选 (model_id, thinking) 列表，None 表示通配
        validate: 是否验证完整参数。当为 False 且 model_id 和 thinking 都提供时，
                 直接返回而不检查是否在候选列表中

    Returns:
        (补全后的 model_id, 补全后的 thinking)

    Raises:
        TupleCompletionError: 当无法唯一确定补全结果时

    Example:
        >>> candidates = [
        ...     ("model-reasoner", True),
        ...     ("model-chat", False),
        ... ]
        >>> complete_model_and_thinking(None, True, candidates)
        ("model-reasoner", True)

        >>> complete_model_and_thinking("model-chat", None, candidates)
        ("model-chat", False)

        >>> complete_model_and_thinking("custom", True, candidates, validate=False)
        ("custom", True)  # 不验证，直接返回
    """
    query = (model_id, thinking)
    result = tuple_completion(query, candidates, validate=validate)

    # 确保返回的都是非 None 值
    final_model, final_thinking = result
    if final_model is None or final_thinking is None:
        raise TupleCompletionError(
            f"参数补全失败：补全结果包含 None。输入: ({model_id}, {thinking}), "
            f"结果: ({final_model}, {final_thinking})"
        )

    return (final_model, final_thinking)
