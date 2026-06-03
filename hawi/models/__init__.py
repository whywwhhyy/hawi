"""
Hawi Agent 模型实现

提供各 LLM 提供商的具体实现。

Example:
    from hawi.models import OpenAIModel
    from hawi.models import ModelOverrideConfig

    model = OpenAIModel(config=ModelOverrideConfig(
        model_id="gpt-4",
        api_key="..."
    ))
"""

from .model import (
    BalanceDetails,
    BalanceInfo,
    DelegateModel,
    Model,
    ModelParams,
    ProviderRequest,
    ProviderResponse,
    TokenEstimate,
    TokenEstimateConfidence,
    TokenEstimateMethod,
)
from .message import (
    AudioPart,
    AudioSource,
    BLOB_URI_SCHEME,
    BlobDirection,
    BlobSource,
    CacheControl,
    CacheControlPart,
    CachePoint,
    CachePointPart,
    CachePointTTL,
    CitationCharLocation,
    CitationContentBlockLocation,
    CitationLocation,
    CitationPageLocation,
    CitationPart,
    CitationsSearchResultLocation,
    CitationsWebSearchResultLocation,
    ContentPart,
    ContentPartType,
    DeltaFinishPart,
    DeltaMetadataPart,
    DeltaPart,
    DeltaPartType,
    DeltaSignaturePart,
    DeltaTextPart,
    DeltaThinkingPart,
    DeltaToolCallPart,
    DocumentPart,
    DocumentSource,
    FilePart,
    FileSource,
    GuardContentPart,
    ImagePart,
    ImageSource,
    MediaSource,
    Message,
    MessageMetadata,
    MessageRequest,
    MessageResponse,
    ReasoningPart,
    RefusalPart,
    SteerMergeMode,
    SteerPart,
    TextPart,
    TokenUsage,
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
    VideoPart,
    VideoSource,
    blob_source,
    blob_uri,
    cache_control_part,
    cache_point_part,
    get_content_cache_point,
    normalize_cache_point,
)
from .usage import (
    merge_token_usage,
    normalize_anthropic_usage,
    normalize_openai_usage,
    normalize_strands_usage,
    normalize_token_usage,
    usage_context_tokens,
    usage_total,
)
from .openai import OpenAIModel
from .anthropic import AnthropicModel
from .deepseek import DeepSeekModel
from .kimi import KimiModel
from .minimax import MiniMaxModel
from .strands import StrandsModel
from .registry import (
    CircularDependencyError,
    ModelConfig,
    ModelOverrideConfig,
    InvalidInheritanceError,
    ModelProviderConfig,
    ModelRegistry,
    UnknownModelError,
    UnknownTemplateError,
    apply_provider_config_overrides,
    model_registry,
    refresh_provider_models,
)
from .config_persistence import (
    ModelProviderConfigPersistenceError,
    persist_provider_properties,
)

# Convenience functions
def create_model(name: str, **overrides):
    """Create a model instance using the global registry."""
    return model_registry.create_model(name, **overrides)

def get_model_adapter(name: str):
    """Get a model adapter class using the global registry."""
    return model_registry.get_model_adapter(name)

def get_model_config(name: str):
    """Get model configuration using the global registry."""
    return model_registry.get_model_config(name)

def load_config(path, quiet: bool = False):
    """Load configuration from a YAML file using the global registry."""
    return model_registry.load_config(path, quiet=quiet)

def list_models():
    """List all available models using the global registry."""
    return model_registry.list_models()

def list_providers():
    """List all registered providers using the global registry."""
    return model_registry.list_providers()

def refresh_models(provider: str):
    """Refresh one provider's model list using the global registry."""
    return model_registry.refresh_provider_models(provider)

def apply_provider_overrides(overrides: dict[str, dict]):
    """Apply temporary provider property overrides using the global registry."""
    return model_registry.apply_provider_config_overrides(overrides)

def persist_provider_config(provider: str, properties: dict, config_paths):
    """Persist provider property updates to models.yaml."""
    return persist_provider_properties(provider, properties, config_paths)

__all__ = [
    # Base classes
    "Model",
    "DelegateModel",
    "BalanceInfo",
    "ProviderRequest",
    "ProviderResponse",
    "ModelParams",
    "BalanceDetails",
    "TokenEstimate",
    "TokenEstimateConfidence",
    "TokenEstimateMethod",
    # Message types
    "Message",
    "MessageRequest",
    "MessageResponse",
    "ContentPartType",
    "DeltaPartType",
    "TokenUsage",
    "ContentPart",
    "TextPart",
    "ImagePart",
    "ImageSource",
    "DocumentPart",
    "DocumentSource",
    "ToolCallPart",
    "ToolResultPart",
    "SteerPart",
    "SteerMergeMode",
    "ReasoningPart",
    "CacheControlPart",
    "CacheControl",
    "AudioPart",
    "AudioSource",
    "BLOB_URI_SCHEME",
    "BlobDirection",
    "BlobSource",
    "VideoPart",
    "VideoSource",
    "FilePart",
    "FileSource",
    "MediaSource",
    "RefusalPart",
    "GuardContentPart",
    "CitationPart",
    "CitationLocation",
    "CitationCharLocation",
    "CitationPageLocation",
    "CitationContentBlockLocation",
    "CitationsWebSearchResultLocation",
    "CitationsSearchResultLocation",
    "DeltaPart",
    "DeltaTextPart",
    "DeltaThinkingPart",
    "DeltaSignaturePart",
    "DeltaMetadataPart",
    "DeltaToolCallPart",
    "DeltaFinishPart",
    "MessageMetadata",
    "ToolDefinition",
    "ToolChoice",
    "blob_source",
    "blob_uri",
    "merge_token_usage",
    "normalize_anthropic_usage",
    "normalize_openai_usage",
    "normalize_strands_usage",
    "normalize_token_usage",
    "usage_context_tokens",
    "usage_total",
    # Model implementations
    "OpenAIModel",
    "AnthropicModel",
    "DeepSeekModel",
    "KimiModel",
    "MiniMaxModel",
    "StrandsModel",
    # Registry
    "ModelRegistry",
    "model_registry",
    "ModelOverrideConfig",
    "CircularDependencyError",
    "UnknownModelError",
    "UnknownTemplateError",

    "InvalidInheritanceError",
    # Convenience functions
    "create_model",
    "get_model_adapter",
    "load_config",
    "list_models",
    "list_providers",
    "apply_provider_config_overrides",
    "apply_provider_overrides",
    "persist_provider_config",
    "persist_provider_properties",
    "ModelProviderConfigPersistenceError",
    "refresh_provider_models",
    "refresh_models",
]
