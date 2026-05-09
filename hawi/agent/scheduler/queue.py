
"""Message queue management for HawiScheduler."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from hawi.events import EventBus
from hawi.models.message import ContentPart

QueueMessageSnapshot = dict[str, Any]


class QueueType(Enum):
    """Queue type enumeration with priority semantics."""

    NORMAL = auto()
    HIGH_PRIO = auto()
    URGENT = auto()


@dataclass
class QueuedMessage:
    """Message in the scheduler queue."""

    id: str
    content: str | list[ContentPart]
    queue_type: QueueType
    created_at: float
    event_bus: EventBus | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    merged_tool_call_ids: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        content: str | list[ContentPart],
        queue_type: QueueType,
        metadata: dict[str, Any] | None = None,
        *,
        event_bus: EventBus | None = None,
    ) -> QueuedMessage:
        return cls(
            id=str(uuid.uuid4())[:8],
            content=content,
            queue_type=queue_type,
            created_at=time.time(),
            event_bus=event_bus,
            metadata=metadata or {},
        )

    def get_content_preview(self, max_length: int = 100) -> str:
        if isinstance(self.content, str):
            text = self.content
        else:
            texts = []
            for part in self.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            text = " ".join(texts)

        if len(text) > max_length:
            return text[: max_length - 3] + "..."
        return text


class MessageQueueManager:
    """Manages the three-tier message queue system."""

    def __init__(self) -> None:
        self._normal_queue: list[QueuedMessage] = []
        self._high_prio_queue: list[QueuedMessage] = []
        self._pending_urgent: QueuedMessage | None = None

    def enqueue_normal(
        self,
        content: str | list[ContentPart],
        metadata: dict[str, Any] | None = None,
        *,
        event_bus: EventBus | None = None,
    ) -> QueuedMessage:
        msg = QueuedMessage.create(
            content=content,
            queue_type=QueueType.NORMAL,
            event_bus=event_bus,
            metadata=metadata,
        )
        self._normal_queue.append(msg)
        return msg

    def enqueue_high_prio(
        self,
        content: str | list[ContentPart],
        metadata: dict[str, Any] | None = None,
        *,
        event_bus: EventBus | None = None,
    ) -> QueuedMessage:
        msg = QueuedMessage.create(
            content=content,
            queue_type=QueueType.HIGH_PRIO,
            event_bus=event_bus,
            metadata=metadata,
        )
        self._high_prio_queue.append(msg)
        return msg

    def enqueue_urgent(
        self,
        content: str | list[ContentPart],
        metadata: dict[str, Any] | None = None,
        *,
        event_bus: EventBus | None = None,
    ) -> QueuedMessage:
        msg = QueuedMessage.create(
            content=content,
            queue_type=QueueType.URGENT,
            event_bus=event_bus,
            metadata=metadata,
        )
        self._pending_urgent = msg
        return msg

    def dequeue_urgent(self) -> QueuedMessage | None:
        msg = self._pending_urgent
        self._pending_urgent = None
        return msg

    def dequeue_high_prio(self) -> QueuedMessage | None:
        if self._high_prio_queue:
            return self._high_prio_queue.pop(0)
        return None

    def peek_high_prio(self) -> QueuedMessage | None:
        if self._high_prio_queue:
            return self._high_prio_queue[0]
        return None

    def dequeue_normal(self) -> QueuedMessage | None:
        if self._normal_queue:
            return self._normal_queue.pop(0)
        return None

    def insert_front_normal(self, msg: QueuedMessage) -> None:
        self._normal_queue.insert(0, msg)

    def has_urgent(self) -> bool:
        return self._pending_urgent is not None

    def has_high_prio(self) -> bool:
        return len(self._high_prio_queue) > 0

    def has_normal(self) -> bool:
        return len(self._normal_queue) > 0

    def get_queue_lengths(self) -> dict[str, int]:
        return {
            "urgent": 1 if self._pending_urgent else 0,
            "high_prio": len(self._high_prio_queue),
            "normal": len(self._normal_queue),
        }

    def get_queue_messages(self) -> dict[str, list[QueueMessageSnapshot]]:
        return {
            "urgent": (
                [self._message_snapshot(self._pending_urgent)]
                if self._pending_urgent
                else []
            ),
            "high_prio": [
                self._message_snapshot(msg)
                for msg in self._high_prio_queue
            ],
            "normal": [
                self._message_snapshot(msg)
                for msg in self._normal_queue
            ],
        }

    @staticmethod
    def _message_snapshot(msg: QueuedMessage) -> QueueMessageSnapshot:
        return {
            "id": msg.id,
            "queue": msg.queue_type.name.lower(),
            "content_preview": msg.get_content_preview(240),
            "created_at": msg.created_at,
            "metadata": dict(msg.metadata),
        }

    def remove_message(self, message_id: str) -> bool:
        if self._pending_urgent and self._pending_urgent.id == message_id:
            self._pending_urgent = None
            return True
        for i, msg in enumerate(self._high_prio_queue):
            if msg.id == message_id:
                self._high_prio_queue.pop(i)
                return True
        for i, msg in enumerate(self._normal_queue):
            if msg.id == message_id:
                self._normal_queue.pop(i)
                return True
        return False

    def remove_messages(self, filter_fn: Callable[[QueuedMessage], bool]) -> list[str]:
        removed: list[str] = []
        if self._pending_urgent and filter_fn(self._pending_urgent):
            removed.append(self._pending_urgent.id)
            self._pending_urgent = None
        high_prio_remaining = []
        for msg in self._high_prio_queue:
            if filter_fn(msg):
                removed.append(msg.id)
            else:
                high_prio_remaining.append(msg)
        self._high_prio_queue = high_prio_remaining
        normal_remaining = []
        for msg in self._normal_queue:
            if filter_fn(msg):
                removed.append(msg.id)
            else:
                normal_remaining.append(msg)
        self._normal_queue = normal_remaining
        return removed

    def clear_queue(self, queue_type: QueueType) -> int:
        if queue_type == QueueType.URGENT:
            if self._pending_urgent:
                self._pending_urgent = None
                return 1
            return 0
        elif queue_type == QueueType.HIGH_PRIO:
            count = len(self._high_prio_queue)
            self._high_prio_queue = []
            return count
        else:
            count = len(self._normal_queue)
            self._normal_queue = []
            return count

    def clear_all_queues(self) -> dict[str, int]:
        return {
            "urgent": self.clear_queue(QueueType.URGENT),
            "high_prio": self.clear_queue(QueueType.HIGH_PRIO),
            "normal": self.clear_queue(QueueType.NORMAL),
        }

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of all queues for session persistence.

        ``event_bus`` references on ``QueuedMessage`` are stripped — callers must
        re-attach a live bus via :py:meth:`rebind_event_bus` after load.
        """
        return {
            "version": 1,
            "urgent": (
                self._queued_to_dict(self._pending_urgent)
                if self._pending_urgent is not None
                else None
            ),
            "high_prio": [self._queued_to_dict(m) for m in self._high_prio_queue],
            "normal": [self._queued_to_dict(m) for m in self._normal_queue],
        }

    def load_snapshot(self, data: dict[str, Any]) -> None:
        """Replace queue contents from a snapshot produced by :py:meth:`snapshot`."""
        version = data.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported queue snapshot version: {version}")
        urgent = data.get("urgent")
        self._pending_urgent = (
            self._queued_from_dict(urgent) if urgent is not None else None
        )
        self._high_prio_queue = [
            self._queued_from_dict(d) for d in data.get("high_prio", [])
        ]
        self._normal_queue = [
            self._queued_from_dict(d) for d in data.get("normal", [])
        ]

    def rebind_event_bus(self, event_bus: EventBus | None) -> None:
        """Re-attach a live ``EventBus`` to every restored queued message."""
        if self._pending_urgent is not None:
            self._pending_urgent.event_bus = event_bus
        for msg in self._high_prio_queue:
            msg.event_bus = event_bus
        for msg in self._normal_queue:
            msg.event_bus = event_bus

    @staticmethod
    def _queued_to_dict(msg: QueuedMessage) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for key, value in msg.metadata.items():
            if isinstance(value, Enum):
                # str+Enum members serialize cleanly already, but also handle
                # plain Enums by storing the member name.
                metadata[key] = value.value if isinstance(value, str) else value.name
            else:
                metadata[key] = value
        return {
            "id": msg.id,
            "content": msg.content,
            "queue_type": msg.queue_type.name,
            "created_at": msg.created_at,
            "metadata": metadata,
            "merged_tool_call_ids": list(msg.merged_tool_call_ids),
        }

    @staticmethod
    def _queued_from_dict(data: dict[str, Any]) -> QueuedMessage:
        return QueuedMessage(
            id=data["id"],
            content=data["content"],
            queue_type=QueueType[data["queue_type"]],
            created_at=data["created_at"],
            event_bus=None,
            metadata=dict(data.get("metadata", {})),
            merged_tool_call_ids=list(data.get("merged_tool_call_ids", [])),
        )
