"""Tests for the hello command's capability negotiation."""

from __future__ import annotations

import json

import pytest

from hawi_engine.protocol import VERSION, parse_frame
from hawi_engine.runtime import SERVER_CAPS, CoreRuntime


class _FakeClient:
    def __init__(self) -> None:
        self.id = "test-client"
        self.authenticated = False
        self.negotiated_caps: set[str] = set()
        self.sent: list[dict] = []

    async def send(self, frame: dict) -> None:
        self.sent.append(frame)

    async def close(self) -> None:
        return None


@pytest.fixture
def runtime():
    return CoreRuntime(model_name="dummy/dummy", token=None)


async def test_hello_without_caps_yields_empty_negotiated(runtime):
    client = _FakeClient()
    await runtime.register_client(client)
    frame = {"version": VERSION, "type": "hello", "id": "h1", "payload": {}}
    await runtime.handle_frame(client, json.dumps(frame))
    ack = next(f for f in client.sent if f["type"] == "ack")
    assert ack["payload"]["negotiated"] == []
    assert ack["payload"]["server_caps"] == sorted(SERVER_CAPS)
    assert client.negotiated_caps == set()


async def test_hello_with_known_caps_intersects(runtime, monkeypatch):
    # Force the server caps to a known value for the test
    monkeypatch.setattr("hawi_engine.runtime.SERVER_CAPS", frozenset({"alpha", "beta"}))
    client = _FakeClient()
    await runtime.register_client(client)
    frame = {
        "version": VERSION,
        "type": "hello",
        "id": "h1",
        "payload": {"client_caps": ["alpha", "gamma"]},
    }
    await runtime.handle_frame(client, json.dumps(frame))
    ack = next(f for f in client.sent if f["type"] == "ack")
    assert ack["payload"]["negotiated"] == ["alpha"]
    assert ack["payload"]["server_caps"] == ["alpha", "beta"]
    assert client.negotiated_caps == {"alpha"}


async def test_hello_rejects_non_list_caps(runtime):
    client = _FakeClient()
    await runtime.register_client(client)
    frame = {
        "version": VERSION,
        "type": "hello",
        "id": "h1",
        "payload": {"client_caps": "not-a-list"},
    }
    await runtime.handle_frame(client, json.dumps(frame))
    err = next(f for f in client.sent if f["type"] == "error")
    assert err["payload"]["code"] == "bad_request"
