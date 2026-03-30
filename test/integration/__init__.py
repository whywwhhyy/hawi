"""
	Integration test utilities using Model Registry.

	All integration tests should use model_registry to create model instances
	instead of manually parsing models.yaml.

	Usage:
	    from test.integration import model_registry, has_model

	    # Check if model exists
	    if has_model("deepseek-chat-openai"):
	        model = model_registry.create_model("deepseek-chat-openai")
	"""

from hawi.models.registry import model_registry, ModelRegistry

__all__ = [
    "model_registry",
    "ModelRegistry",
    "has_model",
    "list_models",
    "get_model_config",
]

def has_model(name: str) -> bool:
    """Check if a model exists in registry.

    Args:
        name: Model name to check

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

def get_model_config(name: str) -> dict | None:
    """Get model configuration.

    Args:
        name: Model name

    Returns:
        Model config dict or None if not found
    """
    return model_registry.get_model_config(name)
