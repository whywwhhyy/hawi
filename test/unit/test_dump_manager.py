"""Tests for DumpManager."""

import json
import os
import tempfile
import shutil
from pathlib import Path

import pytest

from hawi.utils import DumpManager
from hawi.events import AgentRunStartEvent


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
                data = json.load(f)

            assert "session_start" in data
            assert "session_start_iso" in data
            assert "events" in data
            assert data["events"] == []

    def test_dump_event(self):
        """Test dumping an event."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_file = Path(tmpdir) / "test_events.json"
            dm = DumpManager(str(dump_file))

            event = AgentRunStartEvent.create(run_id="test-123", message_preview="Hello")
            dm.dump(event)

            # Verify event was dumped
            with open(dump_file, "r") as f:
                data = json.load(f)

            assert len(data["events"]) == 1
            assert data["events"][0]["type"] == "agent.run_start"
            assert data["events"][0]["source"] == "agent"
            assert data["events"][0]["data"]["run_id"] == "test-123"

    def test_dump_multiple_events(self):
        """Test dumping multiple events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_file = Path(tmpdir) / "test_events.json"
            dm = DumpManager(str(dump_file))

            event1 = AgentRunStartEvent.create(run_id="test-1", message_preview="Hello")
            event2 = AgentRunStartEvent.create(run_id="test-2", message_preview="World")

            dm.dump(event1)
            dm.dump(event2)

            with open(dump_file, "r") as f:
                data = json.load(f)

            assert len(data["events"]) == 2
            assert data["events"][0]["data"]["run_id"] == "test-1"
            assert data["events"][1]["data"]["run_id"] == "test-2"

    def test_dump_without_init(self):
        """Test dump does nothing when disabled."""
        dm = DumpManager(None)
        event = AgentRunStartEvent.create(run_id="test", message_preview="Hello")

        # Should not raise
        dm.dump(event)

    def test_creates_parent_directory(self):
        """Test DumpManager creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "level1" / "level2" / "events.json"
            dm = DumpManager(str(nested_path))

            assert dm.is_enabled()
            assert nested_path.exists()

    def test_dump_raw(self):
        """Test dumping raw data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dump_file = Path(tmpdir) / "test_events.json"
            dm = DumpManager(str(dump_file))

            dm.dump_raw({"test": "data", "number": 42})

            with open(dump_file, "r") as f:
                data = json.load(f)

            assert "raw_dumps" in data
            assert len(data["raw_dumps"]) == 1
            assert data["raw_dumps"][0]["data"]["test"] == "data"
