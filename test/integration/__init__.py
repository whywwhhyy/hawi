"""
Integration test utilities using Model Registry.

All integration tests should use model_registry to create model instances
instead of manually parsing models.yaml.

Usage:
    from test.integration import model_registry, has_factory
    
    # Check if factory exists
    if has_factory("deepseek-chat-openai"):
        model = model_registry.create_model("deepseek-chat-openai")
"""

from hawi.models.registry import model_registry, ModelRegistry

__all__ = [
    "model_registry",
    "ModelRegistry",
    "has_factory",
    "list_factories",
    "get_factory_config",
]


def has_factory(name: str) -> bool:
    """Check if a factory exists in the registry.
    
    Args:
        name: Factory name to check
        
    Returns:
        True if factory exists, False otherwise
    """
    return model_registry.has_factory(name)


def list_factories() -> list[str]:
    """List all available factory names.
    
    Returns:
        List of factory names
    """
    return model_registry.list_factories()


def get_factory_config(name: str) -> dict | None:
    """Get factory configuration.
    
    Args:
        name: Factory name
        
    Returns:
        Factory config dict or None if not found
    """
    return model_registry.get_factory_config(name)
