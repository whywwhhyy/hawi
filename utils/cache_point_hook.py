"""Cache point hook for Strands Agent.

This module provides a hook that dynamically adds cache point before model invocation
and removes it after invocation completes.
"""

from typing import Any

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import BeforeInvocationEvent, AfterInvocationEvent
from strands.types.content import ContentBlock


class CachePointHook(HookProvider):
    """Hook provider that dynamically adds cache point before invoke and removes it after.

    This hook adds a cachePoint content block to the last assistant message before model invocation,
    and removes it from agent.messages after invocation completes.

    Note: Per Bedrock API specification, cache points should only be added to messages
    with role="assistant", not user messages.
    """

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeInvocationEvent, self._on_before_invocation)
        registry.add_callback(AfterInvocationEvent, self._on_after_invocation)

    def _on_before_invocation(self, event: BeforeInvocationEvent) -> None:
        """Add cache point to the last assistant message before invocation."""
        if not event.messages:
            return

        # Find the last assistant message with non-empty content
        # Note: cache point should only be added to assistant messages per Bedrock spec
        for msg in reversed(event.messages):
            if msg.get("role") == "assistant" and "content" in msg:
                content = msg["content"]
                # Check if content is non-empty
                if isinstance(content, list) and len(content) > 0:
                    content.append(ContentBlock(cachePoint={"type": "default"}))
                    break
                elif isinstance(content, str) and content.strip():
                    # If content is a string, convert to list format first
                    msg["content"] = [
                        ContentBlock(text=content),
                        ContentBlock(cachePoint={"type": "default"})
                    ]
                    break

    def _on_after_invocation(self, event: AfterInvocationEvent) -> None:
        """Remove cache point from the last assistant message after invocation."""
        agent = event.agent
        if not agent.messages:
            return

        # Find the last assistant message and remove cachePoint from it
        # Note: cache point should only exist in assistant messages per Bedrock spec
        for msg in reversed(agent.messages):
            if msg.get("role") == "assistant" and "content" in msg:
                content = msg["content"]
                if isinstance(content, list) and len(content) > 0:
                    # Remove cachePoint content blocks from this message
                    msg["content"] = [
                        block for block in content
                        if not (isinstance(block, dict) and "cachePoint" in block)
                    ]
                    break
