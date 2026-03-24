"""
Event base class for Hawi Event System.

Events are:
- Read-only and immutable
- Produced by Model and Agent
- Non-blocking, multi-consumer
"""


from __future__ import annotations

import logging
import time
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


EventSource = Literal['model', 'agent']

ModelEventType = Literal[
    'model.stream_start',
    'model.content_block_start',
    'model.content_block_delta',
    'model.content_block_stop',
    'model.tool_call_block_start',
    'model.tool_call_block_delta',
    'model.tool_call_block_stop',
    'model.content_metadata',
    'model.metadata',
    'model.retry',
    'model.stream_stop',
    'model.error',
]

AgentEventType = Literal[
    'agent.run_start',
    'agent.message_added',
    'agent.tool_call',
    'agent.tool_result_part',
    'agent.tool_result',
    'agent.run_stop',
    'agent.error',
]

EventType = ModelEventType | AgentEventType


class Event(BaseModel):
    """
    统一事件基类。只读、不可变。

    特性：
    - frozen=True: 不可变，防止意外修改
    - extra="forbid": 禁止额外字段，确保类型安全
    - arbitrary_types_allowed=True: 允许任意类型（如 AgentError/ModelError）
    - 抽象基类: 不能直接实例化，必须使用子类
    - 异步处理: 消费者不能阻塞主流程
    - 多播: 可被多个消费者同时监听
    """
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    type: EventType
    source: EventSource
    timestamp: float = Field(default_factory=time.time)

    def __init__(self, **data):
        # 防止直接实例化 Event 基类
        if self.__class__ is Event:
            raise TypeError(
                "Event is an abstract base class and cannot be instantiated directly. "
                "Use concrete subclasses like ModelStreamStartEvent, AgentToolCallEvent, etc."
            )
        super().__init__(**data)


