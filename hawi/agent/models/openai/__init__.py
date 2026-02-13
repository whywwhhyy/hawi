"""
OpenAI API 兼容模型实现

提供 OpenAI 官方 API 及兼容 OpenAI 格式的第三方 API 支持。

Example:
    from hawi.agent.models.openai import OpenAIModel

    model = OpenAIModel(
        model_id="gpt-4",
        api_key="sk-...",
    )
    response = model.invoke(messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}], "name": None, "tool_calls": None, "tool_call_id": None, "metadata": None}])
"""

from ._model import OpenAIModel

__all__ = ["OpenAIModel"]
