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

from typing import Optional

from .model import (
    Model,
    BalanceInfo,
    ProviderRequest,
    ProviderResponse,
    ModelParams,
    BalanceDetails,
)
from .message import (
    Message,
    MessageRequest,
    MessageResponse,
    ContentPartType,
    DeltaPartType,
    TokenUsage,
    ContentPart,
    TextPart,
    ImagePart,
    ImageSource,
    DocumentPart,
    DocumentSource,
    ToolCallPart,
    ToolResultPart,
    ReasoningPart,
    CacheControlPart,
    CacheControl,
    AudioPart,
    AudioSource,
    VideoPart,
    VideoSource,
    FilePart,
    FileSource,
    RefusalPart,
    GuardContentPart,
    CitationPart,
    CitationLocation,
    CitationCharLocation,
    CitationPageLocation,
    CitationContentBlockLocation,
    CitationsWebSearchResultLocation,
    CitationsSearchResultLocation,
    DeltaPart,
    DeltaTextPart,
    DeltaThinkingPart,
    DeltaSignaturePart,
    DeltaMetadataPart,
    DeltaToolCallPart,
    DeltaFinishPart,
    MessageMetadata,
    ToolDefinition,
    ToolChoice,
)
from .openai import OpenAIModel
from .anthropic import AnthropicModel
from .deepseek import DeepSeekModel
from .kimi import KimiModel
from .minimax import MiniMaxModel
from .strands import StrandsModel

def get_model_class(name:str) -> Optional[type]:
    return {
        "OpenAIModel": OpenAIModel,
        "AnthropicModel": AnthropicModel,
        "DeepSeekModel": DeepSeekModel,
        "KimiModel": KimiModel,
        "MiniMaxModel": MiniMaxModel,
        "StrandsModel": StrandsModel,
    }.get(name)

__all__ = [
    # Base classes
    "Model",
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
    "get_model_class",
]
