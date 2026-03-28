"""
Hawi Agent 模型实现

提供各 LLM 提供商的具体实现。

Example:
    from hawi.models import OpenAIModel
    from hawi.models import ModelConfig

    model = OpenAIModel(config=ModelConfig(
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
)
from .message import (
    AudioPart,
    AudioSource,
    CacheControl,
    CacheControlPart,
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
    Message,
    MessageMetadata,
    MessageRequest,
    MessageResponse,
    ReasoningPart,
    RefusalPart,
    TextPart,
    TokenUsage,
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
    VideoPart,
    VideoSource,
)
from .openai import OpenAIModel
from .anthropic import AnthropicModel
from .deepseek import DeepSeekModel
from .kimi import KimiModel
from .minimax import MiniMaxModel
from .strands import StrandsModel
from .registry import (
    CircularDependencyError,
    FactoryConfig,
    InvalidInheritanceError,
    ModelRegistry,
    TemplateConfig,
    UnknownFactoryError,
    UnknownTemplateError,
    create_model,
    get_model_class,
    list_factories,
    list_templates,
    load_config,
    model_registry,
)

__all__ = [
    # Base classes
    "Model",
    "DelegateModel",
    "BalanceInfo",
    "ProviderRequest",
    "ProviderResponse",
    "ModelParams",
    "BalanceDetails",
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
    "ReasoningPart",
    "CacheControlPart",
    "CacheControl",
    "AudioPart",
    "AudioSource",
    "VideoPart",
    "VideoSource",
    "FilePart",
    "FileSource",
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
    "FactoryConfig",
    "TemplateConfig",
    "CircularDependencyError",
    "UnknownFactoryError",
    "UnknownTemplateError",

    "InvalidInheritanceError",
    # Convenience functions
    "create_model",
    "get_model_class",
    "load_config",
    "list_factories",
    "list_templates",
]
