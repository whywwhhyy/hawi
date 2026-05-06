from __future__ import annotations

import json

import pytest

from hawi_core_cli.protocol import VERSION, ProtocolError, json_dumps, make_ack, parse_frame


def test_parse_valid_command_frame() -> None:
    command = parse_frame(
        json.dumps(
            {
                "version": VERSION,
                "type": "enqueue",
                "id": "req-1",
                "payload": {"content": "hi", "queue": "normal"},
            }
        )
    )

    assert command.type == "enqueue"
    assert command.id == "req-1"
    assert command.payload["content"] == "hi"


def test_parse_rejects_unknown_command() -> None:
    with pytest.raises(ProtocolError, match="Unknown command"):
        parse_frame(json.dumps({"version": VERSION, "type": "wat"}))


def test_parse_rejects_wrong_version() -> None:
    with pytest.raises(ProtocolError) as exc:
        parse_frame(json.dumps({"version": "hawi.core.v0", "type": "ping"}))

    assert exc.value.code == "unsupported_version"


def test_ack_frame_serializes_to_ndjson_safe_json() -> None:
    frame = make_ack("hello", request_id="req-2", payload={"message": "你好"})
    encoded = json_dumps(frame)

    assert "\n" not in encoded
    assert json.loads(encoded)["payload"]["message"] == "你好"
