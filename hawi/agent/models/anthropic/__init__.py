"""
Anthropic API 兼容模型实现

修复内容：
1. Document 格式改为 base64 格式
2. Image URL 自动下载转为 base64
3. 正确处理 tool 消息的 tool_use_id
4. 支持 Prompt Caching (cache_control)

使用示例:
    from hawi.agent.models.anthropic import AnthropicModel

    model = AnthropicModel(
        model_id="claude-3-5-sonnet-20241022",
        api_key="sk-ant-...",
    )
    response = model.invoke(messages=[create_user_message("Hello")])
"""

from ._model import AnthropicModel

__all__ = ["AnthropicModel"]
