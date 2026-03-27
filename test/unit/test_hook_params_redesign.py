"""Exploratory tests for hook params redesign bugfix.

These tests run on the CURRENT (unfixed) code to confirm bugs exist.
Tests 1.1 and 1.2 PASS because they assert AttributeError IS raised (confirming missing methods).
Test 1.3 PASSES because it asserts the redundant context arg IS present (confirming the bug).
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import Any, AsyncIterator

from hawi.plugin.hook_context import HookResult, HookContext
from hawi.plugin.plugin import HawiPlugin
from hawi.plugin.decorators import before_model_call
from hawi.agent import HawiAgent
from hawi.models.model import Model
from hawi.models.message import MessageRequest


# ---------------------------------------------------------------------------
# Minimal mock model that returns a single text response without real API calls
# ---------------------------------------------------------------------------

class MockModel(Model):
    """Minimal model stub that returns a fixed streaming response."""

    @property
    def model_id(self) -> str:
        return "mock-model"

    def _get_params(self) -> dict:
        return {}

    def _prepare_request_impl(self, request: MessageRequest) -> dict:
        return {}

    def _parse_response_impl(self, response: Any) -> Any:
        return response

    def _invoke_impl(self, request: MessageRequest) -> Any:
        raise NotImplementedError

    async def _astream_impl(self, request: MessageRequest) -> AsyncIterator[Any]:  # type: ignore[override]
        yield {"type": "text_delta", "index": 0, "delta": "hello"}
        yield {"type": "finish", "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}


# ---------------------------------------------------------------------------
# Task 1.1 — HookResult.replace_model does not exist → AttributeError
# ---------------------------------------------------------------------------

class TestHookResultReplaceModelMissing:
    def test_replace_model_raises_attribute_error(self):
        """1.1 HookResult.replace_model doesn't exist on unfixed code — should raise AttributeError."""
        with pytest.raises(AttributeError):
            HookResult.replace_model(model=MagicMock())


# ---------------------------------------------------------------------------
# Task 1.2 — HookResult.reinvoke does not exist → AttributeError
# ---------------------------------------------------------------------------

class TestHookResultReinvokeMissing:
    def test_reinvoke_raises_attribute_error(self):
        """1.2 HookResult.reinvoke doesn't exist on unfixed code — should raise AttributeError."""
        with pytest.raises(AttributeError):
            HookResult.reinvoke(message="test message")


# ---------------------------------------------------------------------------
# Task 1.3 — before_model_call hook receives redundant context arg at runtime
# ---------------------------------------------------------------------------

class TestBeforeModelCallRedundantContext:
    """Verify that before_model_call currently receives (agent, context, model, hook_ctx).

    On unfixed code the call site is:
        _ainvoke_hook("before_model_call", self, self._context, m, HookContext(...))
    So the bound hook method receives 4 positional args: agent, context, model, hook_ctx.
    This test confirms the redundant `context` parameter is present.
    """

    @pytest.mark.asyncio
    async def test_before_model_call_receives_context_arg(self):
        """1.3 before_model_call hook is called with redundant context arg on unfixed code."""
        captured_args: list[tuple] = []

        class CapturingPlugin(HawiPlugin):
            @before_model_call
            def before_model_call(self, *args):  # type: ignore[override]
                captured_args.append(args)
                return None  # continue normally

        agent = HawiAgent(model=MockModel(), plugins=[CapturingPlugin()])
        await agent.arun("hi")

        assert len(captured_args) >= 1, "before_model_call hook was never called"

        args = captured_args[0]
        # On unfixed code: (agent, context, model, hook_ctx) = 4 args
        # On fixed code:   (agent, model, hook_ctx)           = 3 args
        assert len(args) == 4, (
            f"Expected 4 args (agent, context, model, hook_ctx) on unfixed code, got {len(args)}: {args}"
        )

        from hawi.agent.context import AgentContext
        # Second arg should be AgentContext (the redundant context parameter)
        assert isinstance(args[1], AgentContext), (
            f"Expected args[1] to be AgentContext (redundant context), got {type(args[1])}"
        )


# ===========================================================================
# Task 6 — Fix-checking tests (run on FIXED code, all should PASS)
# ===========================================================================

from hawi.plugin.decorators import after_model_call
from hawi.models.message import MessageResponse


# ---------------------------------------------------------------------------
# 6.1 — HookResult.replace_model(model) creates correct action and model field
# ---------------------------------------------------------------------------

class TestHookResultReplaceModel:
    def test_replace_model_creates_correct_result(self):
        """6.1 HookResult.replace_model(model) sets action='replace_model' and model field."""
        from unittest.mock import MagicMock
        m = MagicMock()
        hr = HookResult.replace_model(m)
        assert hr.action == "replace_model"
        assert hr.model is m


# ---------------------------------------------------------------------------
# 6.2 — HookResult.reinvoke(message) creates correct action and message field
# ---------------------------------------------------------------------------

class TestHookResultReinvoke:
    def test_reinvoke_creates_correct_result(self):
        """6.2 HookResult.reinvoke(message) sets action='reinvoke' and message field."""
        hr = HookResult.reinvoke("injected message")
        assert hr.action == "reinvoke"
        assert hr.message == "injected message"


# ---------------------------------------------------------------------------
# 6.3 — HookResult.restart_turn() creates correct action
# ---------------------------------------------------------------------------

class TestHookResultRestartTurn:
    def test_restart_turn_creates_correct_result(self):
        """6.3 HookResult.restart_turn() sets action='restart_turn'."""
        hr = HookResult.restart_turn()
        assert hr.action == "restart_turn"


# ---------------------------------------------------------------------------
# 6.4 — before_model_call hook is called with 3 args: (agent, model, hook_ctx)
# ---------------------------------------------------------------------------

class TestBeforeModelCallArgs:
    @pytest.mark.asyncio
    async def test_before_model_call_receives_3_args(self):
        """6.4 before_model_call hook receives (agent, model, hook_ctx) — 3 args, no context."""
        from hawi.models.model import Model as ModelBase
        captured_args: list[tuple] = []

        class CapturingPlugin(HawiPlugin):
            @before_model_call
            def hook(self, *args):
                captured_args.append(args)
                return None

        agent = HawiAgent(model=MockModel(), plugins=[CapturingPlugin()])
        await agent.arun("hi")

        assert len(captured_args) >= 1, "before_model_call hook was never called"
        args = captured_args[0]
        assert len(args) == 3, (
            f"Expected 3 args (agent, model, hook_ctx) on fixed code, got {len(args)}: {args}"
        )
        assert isinstance(args[1], ModelBase), (
            f"Expected args[1] to be a Model instance, got {type(args[1])}"
        )
        assert isinstance(args[2], HookContext), (
            f"Expected args[2] to be HookContext, got {type(args[2])}"
        )


# ---------------------------------------------------------------------------
# 6.5 — after_model_call hook is called with 3 args: (agent, response, hook_ctx)
# ---------------------------------------------------------------------------

class TestAfterModelCallArgs:
    @pytest.mark.asyncio
    async def test_after_model_call_receives_3_args(self):
        """6.5 after_model_call hook receives (agent, response, hook_ctx) — 3 args."""
        captured_args: list[tuple] = []

        class CapturingPlugin(HawiPlugin):
            @after_model_call
            def hook(self, *args):
                captured_args.append(args)
                return None

        agent = HawiAgent(model=MockModel(), plugins=[CapturingPlugin()])
        await agent.arun("hi")

        assert len(captured_args) >= 1, "after_model_call hook was never called"
        args = captured_args[0]
        assert len(args) == 3, (
            f"Expected 3 args (agent, response, hook_ctx) on fixed code, got {len(args)}: {args}"
        )
        assert isinstance(args[1], MessageResponse), (
            f"Expected args[1] to be MessageResponse, got {type(args[1])}"
        )
        assert isinstance(args[2], HookContext), (
            f"Expected args[2] to be HookContext, got {type(args[2])}"
        )


# ---------------------------------------------------------------------------
# 6.6 — after_model_call HookContext does not contain stop_reason or usage
# ---------------------------------------------------------------------------

class TestAfterModelCallHookContextFields:
    @pytest.mark.asyncio
    async def test_hook_context_has_no_stop_reason_or_usage(self):
        """6.6 HookContext passed to after_model_call has no stop_reason or usage fields."""
        captured_ctx: list[HookContext] = []

        class CapturingPlugin(HawiPlugin):
            @after_model_call
            def hook(self, agent, response, ctx):
                captured_ctx.append(ctx)
                return None

        agent = HawiAgent(model=MockModel(), plugins=[CapturingPlugin()])
        await agent.arun("hi")

        assert len(captured_ctx) >= 1, "after_model_call hook was never called"
        ctx = captured_ctx[0]
        assert not hasattr(ctx, "stop_reason"), "HookContext should not have stop_reason field"
        assert not hasattr(ctx, "usage"), "HookContext should not have usage field"
        # duration_ms should still be present
        assert hasattr(ctx, "duration_ms"), "HookContext should still have duration_ms field"


# ---------------------------------------------------------------------------
# 6.7 — agent uses hr.model when replace_model action is returned
# ---------------------------------------------------------------------------

class TestReplaceModelUsed:
    @pytest.mark.asyncio
    async def test_replace_model_uses_replacement_model(self):
        """6.7 When before_model_call returns HookResult.replace_model(m2), agent calls m2."""
        from typing import AsyncIterator as _AsyncIterator

        class TrackingModel(MockModel):
            def __init__(self):
                super().__init__()
                self.called = False

            async def _astream_impl(self, request) -> _AsyncIterator[Any]:  # type: ignore[override]
                self.called = True
                yield {"type": "text_delta", "index": 0, "delta": "from tracking model"}
                yield {"type": "finish", "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}

        replacement = TrackingModel()

        class ReplacePlugin(HawiPlugin):
            @before_model_call
            def hook(self, agent, model, ctx):
                return HookResult.replace_model(replacement)

        agent = HawiAgent(model=MockModel(), plugins=[ReplacePlugin()])
        await agent.arun("hi")

        assert replacement.called, "Replacement model's ainvoke was not called"


# ---------------------------------------------------------------------------
# 6.8 — agent skips model call and continues loop when restart_turn is returned
# ---------------------------------------------------------------------------

class TestRestartTurnSkipsModelCall:
    @pytest.mark.asyncio
    async def test_restart_turn_skips_model_call_first_iteration(self):
        """6.8 restart_turn from before_model_call skips model call; model called only once."""
        call_count = 0
        hook_call_count = 0

        from typing import AsyncIterator as _AsyncIterator

        class CountingModel(MockModel):
            async def _astream_impl(self, request) -> _AsyncIterator[Any]:  # type: ignore[override]
                nonlocal call_count
                call_count += 1
                yield {"type": "text_delta", "index": 0, "delta": "hello"}
                yield {"type": "finish", "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}

        class RestartOncePlugin(HawiPlugin):
            @before_model_call
            def hook(self, agent, model, ctx):
                nonlocal hook_call_count
                hook_call_count += 1
                if hook_call_count == 1:
                    return HookResult.restart_turn()
                return None

        agent = HawiAgent(model=CountingModel(), plugins=[RestartOncePlugin()])
        result = await agent.arun("hi")

        # Hook called twice: first returns restart_turn, second returns None
        assert hook_call_count == 2, f"Expected hook called 2 times, got {hook_call_count}"
        # Model called only once (second iteration)
        assert call_count == 1, f"Expected model called 1 time, got {call_count}"


# ---------------------------------------------------------------------------
# 6.9 — agent appends message and re-invokes when reinvoke is returned
# ---------------------------------------------------------------------------

class TestReinvokeAppendsMessage:
    @pytest.mark.asyncio
    async def test_reinvoke_appends_message_to_context(self):
        """6.9 reinvoke from after_model_call appends message to context and re-invokes agent."""
        reinvoke_count = 0

        class ReinvokeOncePlugin(HawiPlugin):
            @after_model_call
            def hook(self, agent, response, ctx):
                nonlocal reinvoke_count
                if reinvoke_count == 0:
                    reinvoke_count += 1
                    return HookResult.reinvoke("injected by hook")
                return None

        agent = HawiAgent(model=MockModel(), plugins=[ReinvokeOncePlugin()])
        await agent.arun("hi")

        # The injected message should appear in the context
        messages = agent.context.messages
        message_texts = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        message_texts.append(part.get("text", ""))
            elif isinstance(content, str):
                message_texts.append(content)

        assert any("injected by hook" in t for t in message_texts), (
            f"Expected 'injected by hook' in context messages, got: {message_texts}"
        )


# ===========================================================================
# Task 7 — Preservation checking tests (run on FIXED code, all should PASS)
# ===========================================================================

import asyncio as _asyncio
from hawi.plugin.decorators import before_tool_calling, after_session, before_session
from hawi.tool.types import ToolResult


# ---------------------------------------------------------------------------
# 7.1 — HookResult.abort() from before_model_call stops agent (model never called)
# ---------------------------------------------------------------------------

class TestAbortPreservation:
    @pytest.mark.asyncio
    async def test_abort_from_before_model_call_stops_agent(self):
        """7.1 abort() from before_model_call terminates agent; model is never called."""
        call_count = 0

        class CountingModel(MockModel):
            async def _astream_impl(self, request):
                nonlocal call_count
                call_count += 1
                yield {"type": "finish", "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}

        class AbortPlugin(HawiPlugin):
            @before_model_call
            def hook(self, agent, model, ctx):
                return HookResult.abort("test abort")

        agent = HawiAgent(model=CountingModel(), plugins=[AbortPlugin()])
        await agent.arun("hi")
        assert call_count == 0, f"Model should never be called when abort is returned; got call_count={call_count}"


# ---------------------------------------------------------------------------
# 7.2 — before_tool_calling skip bypasses tool execution
# ---------------------------------------------------------------------------

class TestSkipPreservation:
    @pytest.mark.asyncio
    async def test_skip_bypasses_tool_execution(self):
        """7.2 HookResult.skip(result) from before_tool_calling bypasses actual tool execution."""
        class SkipPlugin(HawiPlugin):
            @before_tool_calling
            def hook(self, agent, tool_name, arguments, ctx):
                return HookResult.skip(ToolResult(success=True, output="synthetic"))

        agent = HawiAgent(model=MockModel(), plugins=[SkipPlugin()])
        ctx = HookContext(run_id="test", iteration=1, tool_call_id="tc1")
        result = await agent._invoke_before_tool_calling("my_tool", {}, ctx)

        assert result is not None
        assert result.action == "skip"
        assert result.tool_result is not None
        assert result.tool_result.output == "synthetic"


# ---------------------------------------------------------------------------
# 7.3 — hook chain stops at first non-None result, in registration order
# ---------------------------------------------------------------------------

class TestChainStopPreservation:
    @pytest.mark.asyncio
    async def test_chain_stops_at_first_result(self):
        """7.3 Hook chain stops at first non-None result; subsequent hooks are not called."""
        calls = []

        class PluginA(HawiPlugin):
            @before_session
            def hook(self, agent, ctx):
                calls.append("A")
                return HookResult.abort("stop")

        class PluginB(HawiPlugin):
            @before_session
            def hook(self, agent, ctx):
                calls.append("B")

        agent = HawiAgent(model=MockModel(), plugins=[PluginA(), PluginB()])
        result = await agent._invoke_session_hook(
            "before_session", HookContext(run_id="t", iteration=0)
        )
        assert calls == ["A"], f"Expected only ['A'] to be called, got {calls}"
        assert result is not None and result.action == "abort"


# ---------------------------------------------------------------------------
# 7.4 — async hook functions are correctly awaited
# ---------------------------------------------------------------------------

class TestAsyncHookPreservation:
    @pytest.mark.asyncio
    async def test_async_hook_is_awaited(self):
        """7.4 Async hook functions are properly awaited before continuing."""
        awaited = []

        class AsyncPlugin(HawiPlugin):
            @before_session
            async def hook(self, agent, ctx):
                await _asyncio.sleep(0)
                awaited.append(True)

        agent = HawiAgent(model=MockModel(), plugins=[AsyncPlugin()])
        await agent._invoke_session_hook(
            "before_session", HookContext(run_id="t", iteration=0)
        )
        assert awaited == [True], "Async hook was not awaited"


# ---------------------------------------------------------------------------
# 7.5 — property-based test: random hook chains preserve chain-stop semantics
#        (Property 2 — Validates: Requirements 3.1, 3.2, 3.3)
#        hypothesis not available; using parametrized approach instead
# ---------------------------------------------------------------------------

class TestChainStopSemantics:
    """**Validates: Requirements 3.1, 3.2, 3.3**

    Property 2: For any sequence of hooks where one returns a HookResult,
    the chain stops at the first non-None result and returns it.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stop_at", [0, 1, 2, 3])
    async def test_chain_stops_at_index(self, stop_at):
        """7.5 Chain stops at the first hook that returns a non-None result (index stop_at)."""
        n_hooks = 4
        calls = []

        plugins = []
        for i in range(n_hooks):
            idx = i

            class P(HawiPlugin):
                @before_session
                def hook(self, agent, ctx, _i=idx):
                    calls.append(_i)
                    if _i == stop_at:
                        return HookResult.abort(f"stop at {_i}")
                    return None

            plugins.append(P())

        agent = HawiAgent(model=MockModel(), plugins=plugins)
        result = await agent._invoke_session_hook(
            "before_session", HookContext(run_id="t", iteration=0)
        )

        assert calls == list(range(stop_at + 1)), (
            f"Expected hooks 0..{stop_at} to be called, got {calls}"
        )
        assert result is not None, "Expected a non-None HookResult"
        assert result.action == "abort"


# ---------------------------------------------------------------------------
# 7.6 — before_session / after_session called with (agent, HookContext) signature
# ---------------------------------------------------------------------------

class TestSessionHookSignaturePreservation:
    @pytest.mark.asyncio
    async def test_session_hooks_receive_agent_and_ctx(self):
        """7.6 before_session and after_session hooks receive exactly (agent, HookContext)."""
        captured: list[tuple] = []

        class SessionPlugin(HawiPlugin):
            @before_session
            def hook(self, *args):
                captured.append(("before_session", args))

            @after_session
            def hook2(self, *args):
                captured.append(("after_session", args))

        agent = HawiAgent(model=MockModel(), plugins=[SessionPlugin()])
        await agent.arun("hi")

        before = [c for c in captured if c[0] == "before_session"]
        after = [c for c in captured if c[0] == "after_session"]

        assert len(before) >= 1, "before_session hook was never called"
        assert len(after) >= 1, "after_session hook was never called"

        # Each should receive (agent, HookContext) = 2 positional args
        assert len(before[0][1]) == 2, (
            f"before_session expected 2 args (agent, ctx), got {len(before[0][1])}"
        )
        assert isinstance(before[0][1][0], HawiAgent), (
            f"before_session args[0] should be HawiAgent, got {type(before[0][1][0])}"
        )
        assert isinstance(before[0][1][1], HookContext), (
            f"before_session args[1] should be HookContext, got {type(before[0][1][1])}"
        )

        assert len(after[0][1]) == 2, (
            f"after_session expected 2 args (agent, ctx), got {len(after[0][1])}"
        )
        assert isinstance(after[0][1][0], HawiAgent), (
            f"after_session args[0] should be HawiAgent, got {type(after[0][1][0])}"
        )
        assert isinstance(after[0][1][1], HookContext), (
            f"after_session args[1] should be HookContext, got {type(after[0][1][1])}"
        )
