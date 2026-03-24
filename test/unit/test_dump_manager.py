"""Tests for DumpManager."""

import json
import re
import tempfile
from pathlib import Path

import pytest

from hawi.events import DumpManager, AgentRunStartEvent


def _parse_jsonl_records(content: str) -> list[dict]:
    """Parse JSONL content with pretty-printed JSON objects."""
    # Pattern to find boundaries between JSON objects: } followed by whitespace/newlines followed by {
    pattern = r'(?<=\})\s*(?=\{)'
    parts = re.split(pattern, content.strip())

    records = []
    for part in parts:
        part = part.strip()
        if part:
            records.append(json.loads(part))

    return records


class TestDumpManager:
    """Test cases for DumpManager."""

    def test_init_without_dump_file(self):
        """Test DumpManager without dump file is disabled."""
        dm = DumpManager(None)
        assert not dm.is_enabled()
        assert dm.get_dump_path() is None

    def test_init_with_dump_file(self):
        """Test DumpManager with dump file creates initial structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_file = Path(tmpdir) / "test_events.json"
            dm = DumpManager(str(dump_file))

            assert dm.is_enabled()
            assert dm.get_dump_path() == dump_file
            assert dump_file.exists()

            # Check initial structure
            with open(dump_file, "r") as f:
                content = f.read()

            records = _parse_jsonl_records(content)
            assert len(records) == 1
            assert records[0]["type"] == "session_start"
            assert "session_start" in records[0]

    def test_dump_event(self):
        """Test dumping an event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_file = Path(tmpdir) / "test_events.json"
            dm = DumpManager(str(dump_file))

            event = AgentRunStartEvent.create(run_id="test-123")
            dm.dump(event)

            # Verify event was dumped
            with open(dump_file, "r") as f:
                content = f.read()

            records = _parse_jsonl_records(content)
            assert len(records) == 2  # session_start + event
            assert records[1]["type"] == "agent.run_start"
            assert records[1]["source"] == "agent"
            assert records[1]["run_id"] == "test-123"

    def test_dump_multiple_events(self):
        """Test dumping multiple events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_file = Path(tmpdir) / "test_events.json"
            dm = DumpManager(str(dump_file))

            event1 = AgentRunStartEvent.create(run_id="test-1")
            event2 = AgentRunStartEvent.create(run_id="test-2")

            dm.dump(event1)
            dm.dump(event2)

            with open(dump_file, "r") as f:
                content = f.read()

            records = _parse_jsonl_records(content)
            assert len(records) == 3  # session_start + 2 events
            assert records[1]["run_id"] == "test-1"
            assert records[2]["run_id"] == "test-2"

    def test_dump_without_init(self):
        """Test dump does nothing when disabled."""
        dm = DumpManager(None)
        event = AgentRunStartEvent.create(run_id="test")

        # Should not raise
        dm.dump(event)

    def test_creates_parent_directory(self):
        """Test DumpManager creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "level1" / "level2" / "events.json"
            dm = DumpManager(str(nested_path))

            assert dm.is_enabled()
            assert nested_path.exists()

    def test_dump_raw_dict(self):
        """Test dumping raw dict data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_file = Path(tmpdir) / "test_events.json"
            dm = DumpManager(str(dump_file))

            dm.dump({"test": "data", "number": 42})

            with open(dump_file, "r") as f:
                content = f.read()

            records = _parse_jsonl_records(content)
            assert len(records) == 2  # session_start + raw data
            assert records[1]["test"] == "data"
            assert records[1]["number"] == 42
