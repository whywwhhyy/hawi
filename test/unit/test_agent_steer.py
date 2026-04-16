"""Unit tests for HawiAgent steer behavior."""
from unittest.mock import MagicMock

import pytest

from hawi.agent import HawiAgent, SteerPartMergeMode


class TestHawiAgentSteer:
    def test_append_mode_merges_steer_into_next_tool_result(self):
        agent = HawiAgent(model=MagicMock())
        agent._current_tool_calls.append({"id": "call_1", "name": "tool", "arguments": {}})

        steer_id = agent.steer("Please also consider the user's new message.")

        agent._add_tool_result_with_pending_steer("call_1", "tool output")

        assert steer_id
        assert len(agent.context.messages) == 1
        message = agent.context.messages[-1]
        assert message["role"] == "tool"
        tool_result_part = message["content"][0]
        nested_content = tool_result_part["content"]
        assert nested_content[0]["text"] == "tool output"
        assert nested_content[1]["text"] == "Please also consider the user's new message."

    def test_template_mode_converts_tool_result_into_user_message(self):
        agent = HawiAgent(model=MagicMock())
        agent._current_tool_calls.append({"id": "call_2", "name": "tool", "arguments": {}})

        agent.steer(
            "用户补充：优先回答最新的问题",
            merge_mode=SteerPartMergeMode.USER_MESSAGE_TEMPLATE,
        )
        agent._add_tool_result_with_pending_steer("call_2", "weather: sunny")

        assert len(agent.context.messages) == 1
        message = agent.context.messages[-1]
        assert message["role"] == "user"
        content = message["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert "[system] 用户发送了新的消息：用户补充：优先回答最新的问题" in content[0]["text"]
        assert "[tool result] 另外，工具返回结果为：" in content[0]["text"]
        assert "weather: sunny" in content[0]["text"]

    def test_clear_interrupt_state_discards_pending_steer(self):
        agent = HawiAgent(model=MagicMock())

        agent.steer("this should be cleared")
        agent.clear_interrupt_state()
        agent._add_tool_result_with_pending_steer("call_3", "tool output")

        message = agent.context.messages[-1]
        assert message["role"] == "tool"
        tool_result_part = message["content"][0]
        nested_content = tool_result_part["content"]
        assert len(nested_content) == 1
        assert nested_content[0]["text"] == "tool output"

    def test_idle_steer_queues_pending_turn_and_requests_new_loop(self):
        agent = HawiAgent(model=MagicMock())
        agent._ensure_pending_turn_loop = MagicMock()

        steer_id = agent.steer("follow up immediately")

        assert steer_id
        assert len(agent._pending_turns) == 1
        assert agent._pending_turns[0].content[0]["text"] == "follow up immediately"
        agent._ensure_pending_turn_loop.assert_called_once()

    def test_active_session_steer_queues_pending_turn_without_new_loop(self):
        agent = HawiAgent(model=MagicMock())
        agent._ensure_pending_turn_loop = MagicMock()
        agent._session_active = True

        agent.steer("continue current loop")

        assert len(agent._pending_turns) == 1
        agent._ensure_pending_turn_loop.assert_not_called()

    def test_run_only_accepts_message_argument(self):
        agent = HawiAgent(model=MagicMock())

        with pytest.raises(TypeError):
            agent.run("hello", event_bus=MagicMock())  # type: ignore[call-arg]
