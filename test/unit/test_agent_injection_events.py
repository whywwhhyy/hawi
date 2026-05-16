from __future__ import annotations

from typing import Any

import pytest

from hawi.agent import HawiAgent
from hawi.models.model import Model
from hawi.models.message import MessageRequest, MessageResponse
from hawi.plugin import HawiPlugin, HookResult, before_conversation, before_model_call


class OneShotModel(Model):
    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[MessageRequest] = []

    @property
    def model_id(self) -> str:
        return "one-shot"

    def _prepare_request_impl(self, request: MessageRequest) -> dict[str, Any]:
        return {"messages": request.messages, "system": request.system}

    def _parse_response_impl(self, response: dict[str, Any]) -> MessageResponse:
        return MessageResponse(id="one-shot", content=[])

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        return MessageResponse(id="one-shot", content=[])

    async def _astream_impl(self, request: MessageRequest):
        self.requests.append(request)
        yield {
            "type": "text_delta",
            "index": 0,
            "delta": "ok",
            "is_start": True,
            "is_end": False,
        }
        yield {
            "type": "finish",
            "index": 0,
            "stop_reason": "end_turn",
        }

    async def _ainvoke_impl(self, request: MessageRequest):
        async for delta in self._astream_impl(request):
            yield delta


class PromptInjectionPlugin(HawiPlugin):
    @before_conversation(system_prompt_variability="hardcoded")
    def inject_prompt_material(self, agent: HawiAgent, ctx: Any) -> None:
        system_prompt = list(agent.context.system_prompt or [])
        system_prompt.append({"type": "text", "text": "plugin system"})
        agent.context.system_prompt = system_prompt

        insert_at = max(0, len(agent.context.messages) - 1)
        agent.context.inject(
            {
                "role": "user",
                "content": [{"type": "text", "text": "framework env"}],
                "name": None,
                "metadata": {"source": "prompt_injection_test"},
            },
            position=insert_at,
        )


class ReinvokePlugin(HawiPlugin):
    def __init__(self) -> None:
        self.used = False

    @before_model_call
    def reinvoke_once(self, agent: HawiAgent, model: Model, ctx: Any) -> HookResult | None:
        if self.used:
            return None
        self.used = True
        return HookResult.reinvoke("hook supplied follow-up")


@pytest.mark.asyncio
async def test_agent_emits_system_prompt_and_context_injected_events() -> None:
    model = OneShotModel()
    agent = HawiAgent(
        model=model,
        plugins=[PromptInjectionPlugin()],
        system_prompt="base system",
    )
    events = []
    agent.subscribe_blocking(
        events.append,
        ["agent.system_prompt", "agent.context_injected"],
    )

    await agent.arun("hello")

    system_events = [event for event in events if event.type == "agent.system_prompt"]
    context_events = [event for event in events if event.type == "agent.context_injected"]

    assert len(system_events) == 2
    segment_event = next(
        event for event in system_events
        if event.metadata == {
            "content_scope": "injected_segment",
            "change_type": "append",
        }
    )
    snapshot_event = next(
        event for event in system_events
        if event.metadata == {"content_scope": "full_prompt"}
    )
    assert segment_event.content == [{"type": "text", "text": "plugin system"}]
    assert segment_event.plugin_id == "PromptInjectionPlugin"
    assert segment_event.plugin_name == "PromptInjectionPlugin"
    assert segment_event.plugin_role == "plugin"
    assert segment_event.injection_name == "inject_prompt_material"
    assert [part["text"] for part in snapshot_event.content] == [
        "base system",
        "plugin system",
    ]
    assert snapshot_event.origin == "model_input"

    assert len(context_events) == 1
    injected = context_events[0]
    assert injected.role == "user"
    assert injected.hook_type == "before_conversation"
    assert injected.content == [{"type": "text", "text": "framework env"}]
    assert injected.metadata == {"source": "prompt_injection_test"}
    assert injected.plugin_id == "PromptInjectionPlugin"
    assert injected.plugin_name == "PromptInjectionPlugin"
    assert injected.plugin_role == "plugin"
    assert injected.injection_name == "inject_prompt_material"
    assert injected.merge_target == "user_message"
    assert injected.merge_position == "before"
    assert injected.position == 0
    assert model.requests[-1].messages[0]["content"] == [
        {"type": "text", "text": "framework env"}
    ]
    assert model.requests[-1].messages[1]["content"] == [
        {"type": "text", "text": "hello"}
    ]
    assert injected.target_message_index == 0


@pytest.mark.asyncio
async def test_hook_reinvoke_message_emits_context_injected_event() -> None:
    agent = HawiAgent(model=OneShotModel(), plugins=[ReinvokePlugin()])
    events = []
    agent.subscribe_blocking(events.append, ["agent.context_injected"])

    await agent.arun("hello")

    reinvoke_events = [
        event
        for event in events
        if event.hook_type == "before_model_call.reinvoke"
    ]
    assert len(reinvoke_events) == 1
    assert reinvoke_events[0].role == "user"
    assert reinvoke_events[0].content == [
        {"type": "text", "text": "hook supplied follow-up"}
    ]
