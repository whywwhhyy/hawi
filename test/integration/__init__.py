"""
Integration test utilities for loading API keys.

If apikey.yaml exists in the project root, API keys will be loaded from it.
Otherwise, falls back to environment variables.

Structure matches main.py implementation:
- key: provider identifier
- apikey: the API key string
- model: list of model configurations
"""

import os
from pathlib import Path
from typing import Any

import yaml


def load_apikey_yaml() -> list[dict[str, Any]]:
    """Load apikey.yaml from project root if it exists."""
    # Find project root (where apikey.yaml is located)
    current_file = Path(__file__).resolve()
    # Go up from test/integration/__init__.py to project root
    project_root = current_file.parent.parent.parent

    apikey_path = project_root / "apikey.yaml"

    if apikey_path.exists():
        with open(apikey_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or []

    return []


def get_api_key(provider_key: str, env_var: str | None = None) -> str | None:
    """
    Get API key from apikey.yaml or environment variable.

    Priority:
    1. Environment variable (if set)
    2. apikey.yaml (if file exists and contains matching entry)

    Args:
        provider_key: Provider key in apikey.yaml (e.g., "deepseek", "moonshot", "kimi-code")
        env_var: Environment variable name to check first

    Returns:
        API key string or None if not found
    """
    # First, check environment variable
    if env_var:
        key = os.environ.get(env_var)
        if key and key.strip():
            return key

    # Then, try apikey.yaml - match by 'key' field (not 'name')
    configs = load_apikey_yaml()
    for config in configs:
        if config.get("key") == provider_key:
            return config.get("apikey")

    return None


def find_model_config(provider_key: str, model_key: str | None = None) -> dict[str, Any] | None:
    """
    Find a specific model configuration from apikey.yaml.

    Args:
        provider_key: Provider key in apikey.yaml
        model_key: Optional model key to select specific model config

    Returns:
        Model configuration dict or None if not found
    """
    configs = load_apikey_yaml()
    for config in configs:
        if config.get("key") == provider_key:
            models = config.get("model", [])
            if not models:
                return None
            # If model_key specified, find matching model
            if model_key:
                for model in models:
                    if model.get("key") == model_key:
                        return model
                return None
            # Otherwise return first model
            return models[0]
    return None


# Convenience functions for specific providers
def get_deepseek_api_key() -> str | None:
    """Get DeepSeek API key."""
    return get_api_key("deepseek", env_var="DEEPSEEK_API_KEY")


def get_kimi_openai_api_key() -> str | None:
    """Get Kimi OpenAI-compatible API key (from moonshot or kimi-code provider)."""
    # First check env var
    key = os.environ.get("KIMI_API_KEY")
    if key and key.strip():
        return key

    # Try moonshot provider (OpenAI compatible)
    key = get_api_key("moonshot")
    if key:
        return key

    # Try kimi-code provider
    key = get_api_key("kimi-code")
    if key:
        return key

    # Try siliconflow provider with kimi model
    configs = load_apikey_yaml()
    for config in configs:
        if config.get("key") == "siliconflow":
            return config.get("apikey")

    return None


def get_kimi_anthropic_api_key() -> str | None:
    """Get Kimi Anthropic-compatible API key."""
    # Check env var first
    key = os.environ.get("KIMI_ANTHROPIC_API_KEY") or os.environ.get("KIMI_API_KEY")
    if key and key.strip():
        return key

    # Try kimi-code provider (uses anthropic API)
    key = get_api_key("kimi-code")
    if key:
        return key

    # Try siliconflow
    configs = load_apikey_yaml()
    for config in configs:
        if config.get("key") == "siliconflow":
            return config.get("apikey")

    return None


def get_minimax_api_key() -> str | None:
    """Get MiniMax API key."""
    # Check env var first
    key = os.environ.get("MINIMAX_API_KEY")
    if key and key.strip():
        return key

    # Try minimax provider from apikey.yaml
    return get_api_key("minimax")
