"""Unit tests for HawiAgent steer behavior."""
from unittest.mock import MagicMock

import pytest

from hawi.agent import HawiAgent, SteerPartMergeMode
from hawi.errors import ConfigurationError
from hawi.models.deepseek.deepseek_openai import DeepSeekOpenAIModel
from hawi.models.model import Model
from hawi.models.message import MessageRequest, MessageResponse


class DummyModel(Model):
    default_steer_merge_mode = "tool_result_assistant_template_and_user_message"

    @property
    def model_id(self) -> str:
        return "dummy"

    def _prepare_request_impl(self, request: MessageRequest) -> dict:
        return {"messages": request.messages}

    def _parse_response_impl(self, response: dict) -> MessageResponse:
        return MessageResponse(id="dummy", content=[])

    def _invoke_impl(self, request: MessageRequest) -> MessageResponse:
        return MessageResponse(id="dummy", content=[])


class TestHawiAgentSteer:
    def test_real_model_without_declared_steer_mode_fails_early(self):
        class UndeclaredModel(DummyModel):
            default_steer_merge_mode = None

        with pytest.raises(ConfigurationError, match="does not declare default_steer_merge_mode"):
            HawiAgent(model=UndeclaredModel())

    def test_active_tool_steer_materializes_as_raw_steer_part(self):
        agent = HawiAgent(model=MagicMock())
        agent._current_tool_calls.append({"id": "call_1", "name": "tool", "arguments": {}})

        steer_id = agent.steer("Please also consider the user's new message.")

        agent._add_tool_result_with_pending_steer("call_1", "tool output")

        assert steer_id
        assert len(agent.context.messages) == 2

        tool_message, steer_message = agent.context.messages
        assert tool_message["role"] == "tool"
        tool_result_part = tool_message["content"][0]
        nested_content = tool_result_part["content"]
        assert nested_content[0]["text"] == "tool output"
        assert steer_message["role"] == "user"
        steer_part = steer_message["content"][0]
        assert steer_part["type"] == "steer"
        assert steer_part["tool_call_id"] == "call_1"
        assert steer_part["preferred_merge_mode"] is None
        assert steer_part["content"] == [{
            "type": "text",
            "text": "Please also consider the user's new message.",
        }]

    def test_pending_input_messages_expose_high_priority_previews(self):
        agent = HawiAgent(model=MagicMock())
        agent._current_tool_calls.append({"id": "call_1", "name": "tool", "arguments": {}})

        steer_id = agent.steer("new high priority steer")

        messages = agent.get_pending_input_messages()
        assert len(messages) == 1
        message = messages[0]
        assert message["id"] == steer_id
        assert message["queue"] == "high_prio"
        assert message["content_preview"] == "new high priority steer"
        assert message["created_at"] > 0
        assert message["metadata"] == {
            "candidate_tool_call_ids": ["call_1"],
            "merge_mode": None,
        }

    def test_pending_input_messages_remain_high_priority_without_tool_candidates(self):
        agent = HawiAgent(model=MagicMock())
        agent._session_active = True

        steer_id = agent.steer("plain follow-up")

        messages = agent.get_pending_input_messages()
        assert len(messages) == 1
        message = messages[0]
        assert message["id"] == steer_id
        assert message["queue"] == "high_prio"
        assert message["content_preview"] == "plain follow-up"
        assert message["metadata"] == {
            "candidate_tool_call_ids": [],
            "merge_mode": None,
        }

    @pytest.mark.asyncio
    async def test_drained_pending_input_event_is_marked_as_normal_queue(self):
        agent = HawiAgent(model=MagicMock())
        agent._session_active = True
        events = []
        agent.subscribe_blocking(events.append, ["agent.message_added"])

        steer_id = agent.steer("plain follow-up")
        drained = await agent._drain_pending_inputs_to_context("run-plain", None)

        assert drained is True
        assert len(events) == 1
        assert events[0].metadata == {
            "message_id": steer_id,
            "queue": "normal",
            "display_message_type": "normal",
            "source_queue": "high_prio",
            "materialized_as": "plain_user_message",
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "merge_mode",
        [
            SteerPartMergeMode.APPEND_TO_TOOL_RESULT,
            SteerPartMergeMode.USER_MESSAGE_TEMPLATE,
            SteerPartMergeMode.TOOL_RESULT_ASSISTANT_TEMPLATE_AND_USER_MESSAGE,
        ],
    )
    async def test_materialized_steer_event_is_emitted_for_each_merge_mode(
        self,
        merge_mode: SteerPartMergeMode,
    ):
        agent = HawiAgent(model=MagicMock())
        agent._current_tool_calls.append({"id": "call_1", "name": "tool", "arguments": {}})
        events = []
        agent.subscribe_blocking(events.append, ["agent.message_added"])

        steer_id = agent.steer("steer follow-up", merge_mode=merge_mode)
        materialized = agent._add_tool_result_with_pending_steer("call_1", "tool output")
        await agent._emit_materialized_steer_events("run-steer", materialized, None)

        assert len(events) == 1
        event = events[0]
        assert event.run_id == "run-steer"
        assert event.content[0]["type"] == "steer"
        assert event.content[0]["preferred_merge_mode"] == merge_mode.value
        assert event.metadata == {
            "message_id": steer_id,
            "queue": "high_prio",
            "display_message_type": "steer",
            "source_queue": "high_prio",
            "materialized_as": "steer",
            "tool_call_id": "call_1",
            "merge_mode": merge_mode.value,
        }

    def test_user_message_template_lowering_combines_tool_and_steer(self):
        agent = HawiAgent(model=MagicMock())
        agent._current_tool_calls.append({"id": "call_2", "name": "tool", "arguments": {}})

        agent.steer(
            "用户补充：优先回答最新的问题",
            merge_mode=SteerPartMergeMode.USER_MESSAGE_TEMPLATE,
        )
        agent._add_tool_result_with_pending_steer("call_2", "weather: sunny")

        lowered = DummyModel().lower_messages(agent.context.messages)

        assert len(lowered) == 1
        message = lowered[-1]
        assert message["role"] == "user"
        content = message["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert "[system] 用户发送了新的消息：用户补充：优先回答最新的问题" in content[0]["text"]
        assert "[tool result] 另外，工具返回结果为：" in content[0]["text"]
        assert "weather: sunny" in content[0]["text"]

    def test_assistant_template_lowering_adds_tool_assistant_and_user_messages(self):
        agent = HawiAgent(model=MagicMock())
        agent._current_tool_calls.append({"id": "call_3", "name": "tool", "arguments": {}})

        agent.steer(
            "用户补充：处理完当前任务后回复我",
            merge_mode=(
                SteerPartMergeMode.TOOL_RESULT_ASSISTANT_TEMPLATE_AND_USER_MESSAGE
            ),
        )
        agent._add_tool_result_with_pending_steer("call_3", "weather: sunny")

        lowered = DummyModel().lower_messages(agent.context.messages)

        assert len(lowered) == 3

        tool_message, assistant_message, user_message = lowered
        assert tool_message["role"] == "tool"
        tool_result_part = tool_message["content"][0]
        nested_content = tool_result_part["content"]
        assert nested_content[0]["text"] == "weather: sunny"

        assert assistant_message["role"] == "assistant"
        assert assistant_message["content"] == [{
            "type": "text",
            "text": (
                "The user is sending a new steering message, I'll reply to it "
                "and continue the ongoing task with it"
            ),
        }]

        assert user_message["role"] == "user"
        assert user_message["content"] == [{
            "type": "text",
            "text": "用户补充：处理完当前任务后回复我",
        }]

    def test_model_default_lowering_uses_assistant_template_strategy(self):
        messages = [
            {
                "role": "tool",
                "content": [{
                    "type": "tool_result",
                    "tool_call_id": "call_9",
                    "content": [{"type": "text", "text": "weather: sunny"}],
                    "is_error": False,
                }],
                "name": None,
                "metadata": None,
            },
            {
                "role": "user",
                "content": [{
                    "type": "steer",
                    "content": [{"type": "text", "text": "请继续处理并回复我"}],
                    "tool_call_id": "call_9",
                }],
                "name": None,
                "metadata": None,
            },
        ]

        lowered = DummyModel().lower_messages(messages)

        assert len(lowered) == 3
        assert lowered[0]["role"] == "tool"
        assert lowered[1]["role"] == "assistant"
        assert lowered[2]["role"] == "user"

    def test_clear_interrupt_state_preserves_pending_steer(self):
        agent = HawiAgent(model=MagicMock())
        agent._session_active = True

        agent.steer("this should be preserved")
        agent.clear_interrupt_state()

        assert len(agent._pending_inputs) == 1
        assert agent._pending_inputs[0].content[0]["text"] == "this should be preserved"

    def test_idle_steer_queues_pending_turn_and_requests_new_loop(self):
        agent = HawiAgent(model=MagicMock())
        agent._ensure_pending_turn_loop = MagicMock()

        steer_id = agent.steer("follow up immediately")

        assert steer_id
        assert len(agent._pending_inputs) == 1
        assert agent._pending_inputs[0].content[0]["text"] == "follow up immediately"
        assert agent._pending_inputs[0].candidate_tool_call_ids == ()
        agent._ensure_pending_turn_loop.assert_called_once()

    def test_active_session_steer_queues_pending_turn_without_new_loop(self):
        agent = HawiAgent(model=MagicMock())
        agent._ensure_pending_turn_loop = MagicMock()
        agent._session_active = True

        agent.steer("continue current loop")

        assert len(agent._pending_inputs) == 1
        assert agent._pending_inputs[0].candidate_tool_call_ids == ()
        agent._ensure_pending_turn_loop.assert_not_called()

    def test_tool_result_consumes_first_matching_pending_input(self):
        agent = HawiAgent(model=MagicMock())
        agent._current_tool_calls.extend([
            {"id": "call_1", "name": "tool_a", "arguments": {}},
            {"id": "call_2", "name": "tool_b", "arguments": {}},
        ])

        agent.steer("优先处理先返回的工具结果")
        agent._add_tool_result_with_pending_steer("call_2", "result for call 2")

        assert len(agent._pending_inputs) == 0
        steer_message = agent.context.messages[-1]
        assert steer_message["content"][0]["type"] == "steer"
        assert steer_message["content"][0]["tool_call_id"] == "call_2"

    def test_tool_result_consumes_pending_input_created_before_tool_call_ids_exist(self):
        agent = HawiAgent(model=MagicMock())
        agent._session_active = True

        agent.steer("用户补充：请优先处理这个新要求")
        agent._add_tool_result_with_pending_steer("call_late", "tool output")

        assert len(agent._pending_inputs) == 0
        steer_message = agent.context.messages[-1]
        assert steer_message["content"][0]["type"] == "steer"
        assert steer_message["content"][0]["tool_call_id"] == "call_late"

    def test_deepseek_interrupt_and_continue_with_assistant_template(self):
        """DeepSeek 使用打断并继续策略：tool 结果 + assistant 模板 + user steer 消息."""
        agent = HawiAgent(model=MagicMock())
        agent._session_active = True

        agent.steer("用户补充：请优先处理这个新要求")
        agent._add_tool_result_with_pending_steer("call_late", "tool output")

        model = DeepSeekOpenAIModel(api_key="test-key")
        lowered = model.lower_messages(agent.context.messages)
        tool_msg = model._convert_message_to_openai(lowered[0])[0]

        # 3 条消息：tool 结果 + assistant 确认模板 + user steer
        assert len(lowered) == 3
        assert lowered[0]["role"] == "tool"
        assert lowered[1]["role"] == "assistant"
        assert lowered[2]["role"] == "user"

        # tool 消息仍然保留原始内容
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_late"
        assert "tool output" in tool_msg["content"]

        # assistant 确认消息包含打断模板
        assistant_msg = model._convert_message_to_openai(lowered[1])[0]
        assert "steering message" in assistant_msg["content"]

        # user steer 消息包含用户的新要求
        user_msg = model._convert_message_to_openai(lowered[2])[0]
        assert user_msg["role"] == "user"
        assert "用户补充：请优先处理这个新要求" in user_msg["content"]

    def test_batch_tool_results_materialize_steer_after_all_results(self):
        agent = HawiAgent(model=MagicMock())
        agent._current_tool_calls.extend([
            {"id": "call_1", "name": "tool_a", "arguments": {}},
            {"id": "call_2", "name": "tool_b", "arguments": {}},
        ])

        agent.steer("请优先处理新的用户消息")
        agent._add_tool_result_with_pending_steer(
            "call_1",
            "result for call 1",
            materialize_pending_steer=False,
        )
        agent._add_tool_result_with_pending_steer(
            "call_2",
            "result for call 2",
            materialize_pending_steer=False,
        )
        agent._materialize_pending_steer_for_tool_results(["call_1", "call_2"])

        assert [message["role"] for message in agent.context.messages] == [
            "tool",
            "tool",
            "user",
        ]
        steer_message = agent.context.messages[-1]
        assert steer_message["content"][0]["type"] == "steer"
        assert steer_message["content"][0]["tool_call_id"] == "call_1"

    def test_run_only_accepts_message_argument(self):
        agent = HawiAgent(model=MagicMock())

        with pytest.raises(TypeError):
            agent.run("hello", event_bus=MagicMock())  # type: ignore[call-arg]
