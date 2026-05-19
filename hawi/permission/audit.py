"""Audit sink and serialization for permission decisions.

The :class:`PermissionAuditSink` is a lightweight collector attached to
:class:`HawiAgent` that gathers :class:`PermissionAuditRecord` instances
during agent runs.  GUI and observability layers can query or subscribe
to audit events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import PermissionAuditRecord


@dataclass
class PermissionAuditSink:
    """Collects permission audit records and provides query access.

    Usage::

        sink = PermissionAuditSink()
        sink.record(record1)
        sink.record(record2)
        for record in sink.recent():
            print(record.decision)
    """

    _records: list[PermissionAuditRecord] = field(default_factory=list)
    max_records: int = 1000
    """Maximum number of records to keep in memory."""

    def record(self, audit_record: PermissionAuditRecord) -> None:
        """Add a record, trimming if over capacity."""
        self._records.append(audit_record)
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]

    def extend(self, records: list[PermissionAuditRecord]) -> None:
        """Add multiple records at once."""
        for r in records:
            self.record(r)

    def recent(self, limit: int = 50) -> list[PermissionAuditRecord]:
        """Return the most recent *limit* records."""
        if limit <= 0:
            return []
        return self._records[-limit:]

    def all(self) -> list[PermissionAuditRecord]:
        """Return a copy of all records."""
        return list(self._records)

    def clear(self) -> None:
        """Remove all records."""
        self._records.clear()

    def to_json(self) -> list[dict[str, Any]]:
        """Return all records as a list of JSON-safe dicts."""
        return [r.to_dict() for r in self._records]

    def __len__(self) -> int:
        return len(self._records)
