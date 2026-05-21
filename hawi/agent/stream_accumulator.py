"""StreamBlockAccumulator - 流式内容块累积器。

管理单个流式内容块（text/reasoning/tool_use）的生命周期：
start → delta 累积 → end，并创建对应的事件。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from hawi.models import (
    ContentPart,
    DeltaPart,
    TextPart,
    ToolCallPart,
)

from hawi.events import (
    Event,
    ModelContentBlockDeltaEvent,
    ModelContentBlockStartEvent,
    ModelContentBlockStopEvent,
    ModelToolCallBlockDeltaEvent,
    ModelToolCallBlockStartEvent,
    ModelToolCallBlockStopEvent,
)


@dataclass
class StreamBlockAccumulator:
    """流式内容块累积器，统一处理 text/thinking/tool_call 等块类型。

    管理单个流式块的完整生命周期：接收 start chunk 初始化状态，
    累积 delta chunk 内容，在 end chunk 时构建完整的 ContentPart。

    本类只负责创建事件，不发布事件。调用者应使用 _emit_event() 统一发布，
    以确保 DumpManager 能正确记录所有事件。

    Example:
        # 文本块累积器
        text_acc = StreamBlockAccumulator.create_text_handler()

        # 工具调用累积器
        tool_acc = StreamBlockAccumulator.create_tool_handler()

        # 使用
        part, events = acc.handle(chunk, request_id)
        for event in events:
            await agent._emit_event(event, event_bus)
        if part is not None:
            content_parts.append(part)
    """

    # 块类型配置
    stream_part_type: str  # "text_delta", "reasoning_delta", "tool_call_delta"
    block_type: Literal["text", "reasoning", "tool_use", "redacted_thinking"]

    # 当前块状态
    _current_block_index: int = field(default=-1, repr=False)
    _accumulator: Any = field(default=None, repr=False)
    _signature_accumulator: list[str] | None = field(default=None, repr=False)

    # 排序/校验状态
    _pending: dict[int, list[DeltaPart]] = field(default_factory=dict, repr=False)
    _finished_indices: set[int] = field(default_factory=set, repr=False)
    _tool_accumulators: dict[int, dict[str, Any]] = field(default_factory=dict, repr=False)
    # Per-block-index buffer for tool_use blocks whose tool_call_id was empty
    # at is_start. The StartEvent is held back until a delta or stop chunk
    # reveals the real id, so downstream consumers (event_mapper, GUI) never
    # see an empty tool_call_id in a block_start event.
    _pending_tool_start: dict[int, dict[str, Any]] = field(default_factory=dict, repr=False)

    @classmethod
    def create_text_handler(cls) -> StreamBlockAccumulator:
        """创建文本块累积器"""
        return cls(
            stream_part_type="text_delta",
            block_type="text",
        )

    @classmethod
    def create_thinking_handler(cls) -> StreamBlockAccumulator:
        """创建推理块累积器"""
        return cls(
            stream_part_type="reasoning_delta",
            block_type="reasoning",
        )

    @classmethod
    def create_tool_handler(cls) -> StreamBlockAccumulator:
        """创建工具调用块累积器"""
        return cls(
            stream_part_type="tool_call_delta",
            block_type="tool_use",
        )

    def _create_accumulator(self) -> Any:
        """创建累积器"""
        if self.block_type == "text":
            return []
        elif self.block_type == "reasoning":
            return []
        elif self.block_type == "tool_use":
            return {"id": "", "name": "", "arguments": ""}
        return None

    def _add_delta(self, chunk: DeltaPart) -> None:
        """添加 delta 到累积器"""
        if self._accumulator is None:
            return

        if self.block_type == "reasoning" and chunk.get("type") == "signature_delta":
            if self._signature_accumulator is not None:
                delta = chunk.get("delta", "")
                if delta:
                    self._signature_accumulator.append(delta)
            return

        if self.block_type in ("text", "reasoning"):
            # list[str] accumulator
            delta = chunk.get("delta", "")
            if delta:
                self._accumulator.append(delta)
        elif self.block_type == "tool_use":
            # dict accumulator
            acc = self._accumulator
            chunk_id = chunk.get("id")
            chunk_name = chunk.get("name")
            chunk_args = chunk.get("arguments_delta")
            if chunk_id:
                acc["id"] = chunk_id
            if chunk_name:
                acc["name"] = chunk_name
            if chunk_args:
                acc["arguments"] += chunk_args

    @staticmethod
    def _add_tool_delta(acc: dict[str, Any], chunk: DeltaPart) -> None:
        chunk_id = chunk.get("id")
        chunk_name = chunk.get("name")
        chunk_args = chunk.get("arguments_delta")
        if chunk_id:
            acc["id"] = chunk_id
        if chunk_name:
            acc["name"] = chunk_name
        if chunk_args:
            acc["arguments"] += chunk_args

    def _tool_call_id_for_delta(
        self,
        idx: int,
        chunk: DeltaPart,
        acc: dict[str, Any] | None = None,
    ) -> str:
        chunk_id = chunk.get("id") or ""
        if chunk_id:
            return str(chunk_id)

        if acc is not None:
            acc_id = acc.get("id") or ""
            if acc_id:
                return str(acc_id)

        indexed_acc = self._tool_accumulators.get(idx)
        if indexed_acc is not None:
            acc_id = indexed_acc.get("id") or ""
            if acc_id:
                return str(acc_id)

        if isinstance(self._accumulator, dict):
            acc_id = self._accumulator.get("id") or ""
            if acc_id:
                return str(acc_id)

        return ""

    def _build_tool_part(self, acc: dict[str, Any]) -> ToolCallPart:
        return ToolCallPart(
            type="tool_call",
            id=acc["id"],
            name=acc["name"],
            arguments=self._parse_tool_arguments(acc["arguments"]),
        )

    def _build_part(self) -> ContentPart:
        """从累积器构建 ContentPart"""
        if self._accumulator is None:
            raise ValueError("No accumulator to build part from")

        if self.block_type == "text":
            return TextPart(type="text", text="".join(self._accumulator))
        elif self.block_type == "reasoning":
            from hawi.models.message import ReasoningPart
            signature = (
                "".join(self._signature_accumulator)
                if self._signature_accumulator
                else None
            )
            return ReasoningPart(
                type="reasoning",
                reasoning="".join(self._accumulator),
                signature=signature,
                redacted_content=None,
            )
        elif self.block_type == "tool_use":
            acc = self._accumulator
            return ToolCallPart(
                type="tool_call",
                id=acc["id"],
                name=acc["name"],
                arguments=self._parse_tool_arguments(acc["arguments"]),
            )
        raise ValueError(f"Unknown block type: {self.block_type}")

    def partial_content(self) -> tuple[int, ContentPart] | None:
        """Return the currently open text/reasoning block, if it has content."""
        if self.block_type == "tool_use":
            return None
        if self._current_block_index < 0 or self._accumulator is None:
            return None

        if self.block_type == "text":
            text = "".join(self._accumulator)
            if not text.strip():
                return None
            return self._current_block_index, TextPart(type="text", text=text)

        if self.block_type == "reasoning":
            from hawi.models.message import ReasoningPart

            reasoning = "".join(self._accumulator)
            signature = (
                "".join(self._signature_accumulator)
                if self._signature_accumulator
                else None
            )
            if not reasoning and not signature:
                return None
            return self._current_block_index, ReasoningPart(
                type="reasoning",
                reasoning=reasoning,
                signature=signature,
                redacted_content=None,
            )

        return None

    def _is_empty(self) -> bool:
        """检查累积器是否为空"""
        if self._accumulator is None:
            return True
        if self.block_type == "reasoning":
            # DeepSeek adaptive thinking can intentionally produce an empty
            # reasoning block; keep it as part of the assistant message.
            return False
        if self.block_type in ("text", "reasoning"):
            return not "".join(self._accumulator).strip()
        elif self.block_type == "tool_use":
            return not self._accumulator.get("name")
        return False

    @staticmethod
    def _parse_tool_arguments(args_str: str) -> dict[str, Any]:
        """解析工具参数 JSON"""
        try:
            return json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            return {}

    def _process_chunk(
        self,
        chunk: DeltaPart,
        request_id: str,
        is_streaming: bool = True,
    ) -> tuple[ContentPart | None, list[Event]]:
        """处理单个 chunk 的核心逻辑（不含排序/校验）。

        本方法只创建事件，不发布。调用者应使用 _emit_event() 统一发布事件，
        以确保 DumpManager 能正确记录所有事件。

        Args:
            chunk: DeltaPart（必须是匹配的 type）
            request_id: 请求 ID
            is_streaming: 是否来自流式接口（默认 True）

        Returns:
            tuple[ContentPart | None, list[Event]]:
                - Part 在块完成时返回，否则为 None
                - 事件列表需要由调用者发布
        """
        idx = chunk.get("index", 0)
        events: list[Event] = []

        # is_start: 初始化新块，创建 StartEvent
        if chunk.get("is_start"):
            self._current_block_index = idx
            self._accumulator = self._create_accumulator()
            self._signature_accumulator = [] if self.block_type == "reasoning" else None

            if self.block_type == "tool_use":
                chunk_id = chunk.get("id") or ""
                if chunk_id:
                    events.append(ModelToolCallBlockStartEvent.create(
                        request_id=request_id,
                        block_index=idx,
                        tool_call_id=chunk_id,
                        tool_name=chunk.get("name") or "",
                    ))
                else:
                    # Defer StartEvent until id is known. Some OpenAI-compatible
                    # providers stream the id in a later chunk.
                    self._pending_tool_start[idx] = {
                        "tool_name": chunk.get("name") or "",
                    }
            else:
                events.append(ModelContentBlockStartEvent.create(
                    request_id=request_id,
                    block_index=idx,
                    block_type=self.block_type,
                ))

        # If a tool block's StartEvent is pending and this chunk now reveals
        # the id, flush the StartEvent before emitting the DeltaEvent.
        if self.block_type == "tool_use" and idx in self._pending_tool_start:
            chunk_id = chunk.get("id") or ""
            if chunk_id:
                meta = self._pending_tool_start.pop(idx)
                events.append(ModelToolCallBlockStartEvent.create(
                    request_id=request_id,
                    block_index=idx,
                    tool_call_id=chunk_id,
                    tool_name=meta.get("tool_name") or chunk.get("name") or "",
                ))

        # 创建 DeltaEvent
        if self.block_type == "tool_use":
            events.append(ModelToolCallBlockDeltaEvent.create(
                request_id=request_id,
                block_index=idx,
                tool_call_id=self._tool_call_id_for_delta(idx, chunk),
                arguments_delta=chunk.get("arguments_delta", ""),
                is_streaming=is_streaming,
            ))
        else:
            events.append(ModelContentBlockDeltaEvent.create(
                request_id=request_id,
                part=chunk,
                is_streaming=is_streaming,
            ))

        # 累积内容
        self._add_delta(chunk)

        # is_end: 构建 Part，创建 StopEvent
        part: ContentPart | None = None
        if chunk.get("is_end") and self._accumulator is not None:
            part = self._build_part()

            if self.block_type == "tool_use":
                acc = self._accumulator
                # Last chance to flush a deferred StartEvent — the accumulator
                # has the canonical id by now (set via _add_delta above).
                if idx in self._pending_tool_start:
                    acc_id = acc.get("id") or ""
                    if acc_id:
                        meta = self._pending_tool_start.pop(idx)
                        events.append(ModelToolCallBlockStartEvent.create(
                            request_id=request_id,
                            block_index=idx,
                            tool_call_id=acc_id,
                            tool_name=meta.get("tool_name") or acc.get("name") or "",
                        ))
                    else:
                        # No id ever arrived — drop the pending start.
                        self._pending_tool_start.pop(idx, None)
                events.append(ModelToolCallBlockStopEvent.create(
                    request_id=request_id,
                    block_index=idx,
                    tool_call_id=acc.get("id") or "",
                    tool_name=acc.get("name") or "",
                    arguments=self._parse_tool_arguments(acc.get("arguments", "")),
                ))
            else:
                assert part is not None
                events.append(ModelContentBlockStopEvent.create(
                    request_id=request_id,
                    block_index=idx,
                    content=[part],
                ))

            # 在重置前检查是否为空
            is_empty = self._is_empty()

            # 记录已完成的 idx，重置状态
            self._finished_indices.add(self._current_block_index)
            self._current_block_index = -1
            self._accumulator = None
            self._signature_accumulator = None

            if is_empty:
                part = None

        return part, events

    def _process_tool_chunk(
        self,
        chunk: DeltaPart,
        request_id: str,
        is_streaming: bool,
    ) -> tuple[ContentPart | None, list[Event]]:
        idx = chunk.get("index", 0)
        events: list[Event] = []

        if chunk.get("is_start"):
            if idx in self._tool_accumulators:
                raise ValueError(
                    f"Block index {idx} already started "
                    f"(block_type={self.block_type})"
                )
            acc = {"id": "", "name": "", "arguments": ""}
            self._tool_accumulators[idx] = acc
            chunk_id = chunk.get("id") or ""
            if chunk_id:
                events.append(ModelToolCallBlockStartEvent.create(
                    request_id=request_id,
                    block_index=idx,
                    tool_call_id=chunk_id,
                    tool_name=chunk.get("name") or "",
                ))
            else:
                # Defer StartEvent until id is known.
                self._pending_tool_start[idx] = {
                    "tool_name": chunk.get("name") or "",
                }
        else:
            acc = self._tool_accumulators[idx]

        acc = self._tool_accumulators[idx]

        # Flush a deferred StartEvent if the id arrived in this delta chunk.
        if idx in self._pending_tool_start:
            chunk_id = chunk.get("id") or ""
            if chunk_id:
                meta = self._pending_tool_start.pop(idx)
                events.append(ModelToolCallBlockStartEvent.create(
                    request_id=request_id,
                    block_index=idx,
                    tool_call_id=chunk_id,
                    tool_name=meta.get("tool_name") or chunk.get("name") or "",
                ))

        events.append(ModelToolCallBlockDeltaEvent.create(
            request_id=request_id,
            block_index=idx,
            tool_call_id=self._tool_call_id_for_delta(idx, chunk, acc),
            arguments_delta=chunk.get("arguments_delta", ""),
            is_streaming=is_streaming,
        ))
        self._add_tool_delta(acc, chunk)

        part: ContentPart | None = None
        if chunk.get("is_end"):
            # Last chance to flush a deferred StartEvent — the accumulator
            # has the canonical id by now (set via _add_tool_delta above).
            if idx in self._pending_tool_start:
                acc_id = acc.get("id") or ""
                if acc_id:
                    meta = self._pending_tool_start.pop(idx)
                    events.append(ModelToolCallBlockStartEvent.create(
                        request_id=request_id,
                        block_index=idx,
                        tool_call_id=acc_id,
                        tool_name=meta.get("tool_name") or acc.get("name") or "",
                    ))
                else:
                    # No id ever arrived — drop the pending start; downstream
                    # would have nothing meaningful to key on. Logged via the
                    # absence of any block_start event.
                    self._pending_tool_start.pop(idx, None)

            part = self._build_tool_part(acc)
            events.append(ModelToolCallBlockStopEvent.create(
                request_id=request_id,
                block_index=idx,
                tool_call_id=acc.get("id") or "",
                tool_name=acc.get("name") or "",
                arguments=self._parse_tool_arguments(acc.get("arguments", "")),
            ))
            del self._tool_accumulators[idx]
            self._finished_indices.add(idx)
            if not acc.get("name"):
                part = None

        return part, events

    def _handle_tool_chunk(
        self,
        chunk: DeltaPart,
        request_id: str,
        is_streaming: bool,
    ) -> list[tuple[ContentPart | None, list[Event]]]:
        idx = chunk.get("index", 0)

        if chunk.get("is_start") and idx in self._finished_indices:
            raise ValueError(
                f"Block index {idx} already completed "
                f"(block_type={self.block_type})"
            )

        if idx in self._finished_indices:
            return []

        if not chunk.get("is_start") and idx not in self._tool_accumulators:
            self._pending.setdefault(idx, []).append(chunk)
            return []

        results = [self._process_tool_chunk(chunk, request_id, is_streaming)]

        if chunk.get("is_start") and idx in self._pending:
            pending_chunks = self._pending.pop(idx)
            for pending_chunk in pending_chunks:
                if idx in self._finished_indices:
                    break
                results.append(
                    self._process_tool_chunk(pending_chunk, request_id, is_streaming)
                )

        return results

    def handle(
        self,
        chunk: DeltaPart,
        request_id: str,
        is_streaming: bool = True,
    ) -> list[tuple[ContentPart | None, list[Event]]]:
        """处理单个 chunk，按 idx 校验与排序后返回结果列表。

        正常情况下列表只含一个元素；当缓冲的乱序 chunk 在当前块结束后
        被一并刷新时，可能返回多个元素（每个完成的块对应一个元素）。

        本方法只创建事件，不发布。调用者应使用 _emit_event() 统一发布事件，
        以确保 DumpManager 能正确记录所有事件。

        Args:
            chunk: DeltaPart（必须是匹配的 type）
            request_id: 请求 ID
            is_streaming: 是否来自流式接口（默认 True）

        Returns:
            list[tuple[ContentPart | None, list[Event]]]:
                每个元素对应一个已处理的 chunk，包含可选的完整 Part 和事件列表。

        Raises:
            ValueError: 收到已完成块的重复 chunk（idx 校验失败）
        """
        idx = chunk.get("index", 0)

        if self.block_type == "tool_use":
            return self._handle_tool_chunk(chunk, request_id, is_streaming)

        # 校验：同一 idx 不能被 start 两次（协议错误）
        # 注意：只在 is_start 时检查，迟到的 delta/end 应静默丢弃而非报错——
        # 乱序流中 is_end 可能早于部分 delta 到达，此时 idx 已在 _finished_indices
        # 中，但 delta 本身并非协议错误。
        if chunk.get("is_start") and idx in self._finished_indices:
            raise ValueError(
                f"Block index {idx} already completed "
                f"(block_type={self.block_type})"
            )

        # 已完成块的迟到 chunk（is_end 先于部分 delta 到达的乱序情况）：静默丢弃。
        # 块已构建完毕，无法再合并内容，继续处理只会产生无意义的事件。
        if idx in self._finished_indices:
            return []

        # 排序：当前有未完成的块时，缓冲不属于该块的 chunk，待当前块结束后按序刷新。
        if self._current_block_index >= 0 and idx != self._current_block_index:
            self._pending.setdefault(idx, []).append(chunk)
            return []

        # 处理当前 chunk
        part, events = self._process_chunk(chunk, request_id, is_streaming)
        results: list[tuple[ContentPart | None, list[Event]]] = [(part, events)]

        # 块结束后，按 idx 升序刷新缓冲区
        if chunk.get("is_end") and self._pending:
            for pending_idx in sorted(self._pending.keys()):
                for pending_chunk in self._pending.pop(pending_idx):
                    p, ev = self._process_chunk(pending_chunk, request_id, is_streaming)
                    results.append((p, ev))

        return results
