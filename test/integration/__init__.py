"""
Integration test utilities for loading API keys.

Supports loading API keys from:
1. Environment variables (highest priority)
2. models.yaml in project root (new format with api_keys section)
3. ~/.hawi/models.yaml (user-level config)

models.yaml format:
    api_keys:
      deepseek: sk-...
      moonshot: sk-...
      kimi: sk-...

    factories:
      deepseek-chat:
        class: DeepSeekOpenAIModel
        model_id: deepseek-chat
        api_key: ${api_key:deepseek}
"""

import os
from pathlib import Path
from typing import Any

import yaml


def load_models_yaml(path: Path | None = None) -> dict[str, Any]:
    """Load models.yaml config if it exists.

    Args:
        path: Optional specific path to models.yaml. If not provided,
              searches project root and ~/.hawi/models.yaml.

    Returns:
        Dict with 'api_keys' and 'factories' sections, or empty dict.
    """
    if path is not None:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    # Search for models.yaml in standard locations
    # 1. Project root (current working directory)
    project_config = Path.cwd() / "models.yaml"
    if project_config.exists():
        with open(project_config, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # 2. User-level config
    user_config = Path.home() / ".hawi" / "models.yaml"
    if user_config.exists():
        with open(user_config, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    return {}


def get_api_key_from_models_yaml(key_alias: str) -> str | None:
    """Get API key from models.yaml api_keys section.

    Args:
        key_alias: The alias name in api_keys section (e.g., "deepseek", "moonshot")

    Returns:
        API key string or None if not found.
    """
    config = load_models_yaml()
    api_keys = config.get("api_keys", {})
    if isinstance(api_keys, dict):
        return api_keys.get(key_alias)
    return None


def get_api_key(
    key_aliases: list[str],
    env_vars: list[str],
) -> str | None:
    """Get API key with fallback chain.

    Priority:
    1. Environment variables (first match)
    2. models.yaml api_keys section (first alias match)

    Args:
        key_aliases: List of alias names to try in models.yaml (e.g., ["deepseek", "moonshot"])
        env_vars: List of environment variable names to try (e.g., ["DEEPSEEK_API_KEY"])

    Returns:
        API key string or None if not found anywhere.
    """
    # First, check environment variables
    for env_var in env_vars:
        key = os.environ.get(env_var)
        if key and key.strip():
            return key

    # Then, try models.yaml api_keys
    for alias in key_aliases:
        key = get_api_key_from_models_yaml(alias)
        if key and key.strip():
            return key

    return None


# =============================================================================
# Convenience functions for specific providers
# =============================================================================


def get_deepseek_api_key() -> str | None:
    """Get DeepSeek API key.

    Checks (in order):
    1. DEEPSEEK_API_KEY environment variable
    2. models.yaml api_keys.deepseek
    """
    return get_api_key(
        key_aliases=["deepseek"],
        env_vars=["DEEPSEEK_API_KEY"],
    )


def get_kimi_openai_api_key() -> str | None:
    """Get Kimi OpenAI-compatible API key.

    Checks (in order):
    1. KIMI_API_KEY environment variable
    2. MOONSHOT_API_KEY environment variable
    3. models.yaml api_keys.moonshot
    4. models.yaml api_keys.kimi
    """
    return get_api_key(
        key_aliases=["moonshot", "kimi", "moonshot-bao"],
        env_vars=["KIMI_API_KEY", "MOONSHOT_API_KEY"],
    )


def get_kimi_anthropic_api_key() -> str | None:
    """Get Kimi Anthropic-compatible API key.

    Checks (in order):
    1. KIMI_ANTHROPIC_API_KEY environment variable
    2. KIMI_API_KEY environment variable
    3. models.yaml api_keys.kimi
    4. models.yaml api_keys.moonshot
    """
    return get_api_key(
        key_aliases=["kimi", "moonshot", "moonshot-bao"],
        env_vars=["KIMI_ANTHROPIC_API_KEY", "KIMI_API_KEY"],
    )


def get_minimax_api_key() -> str | None:
    """Get MiniMax API key.

    Checks (in order):
    1. MINIMAX_API_KEY environment variable
    2. models.yaml api_keys.minimax
    """
    return get_api_key(
        key_aliases=["minimax"],
        env_vars=["MINIMAX_API_KEY"],
    )


def get_glm_api_key() -> str | None:
    """Get GLM (Zhipu AI) API key.

    Checks (in order):
    1. GLM_API_KEY environment variable
    2. ZHIPU_API_KEY environment variable
    3. models.yaml api_keys.glm
    4. models.yaml api_keys.zhipu
    """
    return get_api_key(
        key_aliases=["glm", "zhipu"],
        env_vars=["GLM_API_KEY", "ZHIPU_API_KEY"],
    )


# Backwards compatibility alias
get_moonshot_api_key = get_kimi_openai_api_key
