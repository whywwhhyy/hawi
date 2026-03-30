"""
Models integration tests for Hawi Agent.

Uses Model Registry for all model creation and configuration.
All tests should use model names from models.yaml.
"""

import functools

import pytest

from hawi.models.registry import model_registry

__all__ = [
    "model_registry",
    "has_model",
    "list_models",
    "create_model",
    "is_rate_limit_error",
    "skip_on_rate_limit",
    "async_skip_on_rate_limit",
]


def has_model(name: str) -> bool:
    """Check if a model exists in the registry.
    
    Args:
        name: Factory name to check
        
    Returns:
        True if model exists, False otherwise
    """
    return model_registry.has_model(name)


def list_models() -> list[str]:
    """List all available model names.
    
    Returns:
        List of model names
    """
    return model_registry.list_models()


def create_model(name: str, **overrides):
    """Create a model instance from model name.
    
    Args:
        name: Factory name
        **overrides: Optional parameter overrides
        
    Returns:
        Model instance
    """
    return model_registry.create_model(name, overrides=overrides or None)


def is_rate_limit_error(e: Exception) -> bool:
    """Check if exception is a rate limit error (429).
    
    Checks for:
    - HTTP 429 status code
    - "rate limit" in error message
    - Provider-specific rate limit error codes (GLM: 1302)
    """
    error_msg = str(e).lower()
    if "429" in error_msg or "rate limit" in error_msg:
        return True
    # Provider-specific error codes
    if "1302" in error_msg:  # GLM rate limit
        return True
    return False


def skip_on_rate_limit(func):
    """Decorator that skips test on rate limit errors."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if is_rate_limit_error(e):
                pytest.skip(f"Rate limit exceeded: {e}")
            raise
    return wrapper


def async_skip_on_rate_limit(func):
    """Decorator that skips async test on rate limit errors."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if is_rate_limit_error(e):
                pytest.skip(f"Rate limit exceeded: {e}")
            raise
    return wrapper
