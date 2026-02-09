"""
Integration test utilities for loading API keys.

If apikey.yaml exists in the project root, API keys will be loaded from it.
Otherwise, falls back to environment variables.
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


def get_api_key(name: str, env_var: str | None = None, base_url: str | None = None) -> str | None:
    """
    Get API key from apikey.yaml or environment variable.
    
    Priority:
    1. Environment variable (if set)
    2. apikey.yaml (if file exists and contains matching entry)
    
    Args:
        name: Provider name in apikey.yaml (e.g., "deepseek", "kimi-openai")
        env_var: Environment variable name to check first
        base_url: Optional base_url to match (for providers with multiple endpoints)
    
    Returns:
        API key string or None if not found
    """
    # First, check environment variable
    if env_var:
        key = os.environ.get(env_var)
        if key and key.strip():
            return key
    
    # Then, try apikey.yaml
    configs = load_apikey_yaml()
    for config in configs:
        if config.get("name") == name:
            # If base_url is specified, match it
            if base_url is None or config.get("base_url") == base_url:
                return config.get("apikey")
    
    return None


# Convenience functions for specific providers
def get_deepseek_api_key() -> str | None:
    """Get DeepSeek API key."""
    return get_api_key("deepseek", env_var="DEEPSEEK_API_KEY")


def get_kimi_openai_api_key() -> str | None:
    """Get Kimi OpenAI-compatible API key."""
    return get_api_key("kimi-openai", env_var="KIMI_API_KEY")


def get_kimi_anthropic_api_key() -> str | None:
    """Get Kimi Anthropic-compatible API key."""
    # Check env var first
    key = os.environ.get("KIMI_ANTHROPIC_API_KEY") or os.environ.get("KIMI_API_KEY")
    if key and key.strip():
        return key
    # Then check apikey.yaml
    return get_api_key("kimi-anthropic", base_url="https://api.kimi.com/coding/")
