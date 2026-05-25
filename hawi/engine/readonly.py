"""Read-only Hawi engine runtime for session browsing."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Protocol

from hawi.session.reader import ReadOnlySessionBrowser

from .protocol import (
    CoreCommand,
    ProtocolError,
    make_ack,
    make_error,
    make_frame,
    parse_frame,
)

logger = logging.getLogger(__name__)

READONLY_SERVER_CAPS: frozenset[str] = frozenset({
    "readonly_v1",
    "session_search_v1",
})


class ReadOnlyRuntimeClient(Protocol):
    id: str
    authenticated: bool
    negotiated_caps: set[str]

    async def send(self, frame: dict[str, Any]) -> None:
        """Queue a frame for this client."""

    async def close(self) -> None:
        """Close the client transport."""


class ReadOnlyRuntime:
    """A lightweight command runtime for searching persisted chat history."""

    def __init__(
        self,
        *,
        session_root: Path | str | None = None,
        token: str | None = None,
    ) -> None:
        self._browser = ReadOnlySessionBrowser(session_root)
        self._token = token
        self._clients: set[ReadOnlyRuntimeClient] = set()
        self._shutdown_requested = asyncio.Event()

    @property
    def is_shutdown_requested(self) -> bool:
        return self._shutdown_requested.is_set()

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        if self._shutdown_requested.is_set():
            return
        self._shutdown_requested.set()
        for client in list(self._clients):
            await client.close()

    async def wait_shutdown(self) -> None:
        await self._shutdown_requested.wait()

    async def register_client(self, client: ReadOnlyRuntimeClient) -> None:
        self._clients.add(client)
        if self._token is None:
            client.authenticated = True
            await client.send(make_frame("core.ready", self._ready_payload()))

    async def unregister_client(self, client: ReadOnlyRuntimeClient) -> None:
        self._clients.discard(client)

    async def handle_frame(
        self,
        client: ReadOnlyRuntimeClient,
        raw: str | bytes,
    ) -> None:
        try:
            command = parse_frame(raw)
        except ProtocolError as exc:
            await client.send(make_error(str(exc), code=exc.code))
            return
        await self.handle_command(client, command)

    async def handle_command(
        self,
        client: ReadOnlyRuntimeClient,
        command: CoreCommand,
    ) -> None:
        try:
            if command.type == "hello":
                await self._handle_hello(client, command)
                return
            if not client.authenticated:
                await client.send(
                    make_error(
                        "Client must send a successful hello command first.",
                        request_id=command.id,
                        code="unauthenticated",
                    )
                )
                return

            if command.type == "ping":
                await client.send(
                    make_frame("pong", {"ok": True}, request_id=command.id)
                )
            elif command.type == "session_list":
                await self._handle_session_list(client, command)
            elif command.type == "session_history":
                await self._handle_session_history(client, command)
            elif command.type == "session_search":
                await self._handle_session_search(client, command)
            elif command.type == "shutdown":
                await client.send(make_ack("shutdown", request_id=command.id))
                await self.stop()
            else:
                await client.send(
                    make_error(
                        f"Command is not available in read-only mode: {command.type}",
                        request_id=command.id,
                        code="read_only_mode",
                    )
                )
        except Exception as exc:
            logger.exception("Read-only command failed: %s", command.type)
            await client.send(
                make_error(
                    str(exc),
                    request_id=command.id,
                    code="command_failed",
                    details={"command": command.type, "class": exc.__class__.__name__},
                )
            )

    async def _handle_hello(
        self,
        client: ReadOnlyRuntimeClient,
        command: CoreCommand,
    ) -> None:
        if self._token is not None and command.payload.get("token") != self._token:
            await client.send(
                make_error(
                    "Invalid authentication token.",
                    request_id=command.id,
                    code="unauthorized",
                )
            )
            return

        client_caps_raw = command.payload.get("client_caps", [])
        if not isinstance(client_caps_raw, list) or not all(
            isinstance(cap, str) for cap in client_caps_raw
        ):
            await client.send(
                make_error(
                    "'hello.payload.client_caps' must be a list of strings.",
                    request_id=command.id,
                    code="bad_request",
                )
            )
            return
        negotiated = set(client_caps_raw) & set(READONLY_SERVER_CAPS)
        client.negotiated_caps = negotiated
        was_authenticated = client.authenticated
        client.authenticated = True
        await client.send(
            make_ack(
                "hello",
                request_id=command.id,
                payload={
                    "authenticated": True,
                    "server_caps": sorted(READONLY_SERVER_CAPS),
                    "negotiated": sorted(negotiated),
                },
            )
        )
        if not was_authenticated:
            await client.send(make_frame("core.ready", self._ready_payload()))

    async def _handle_session_list(
        self,
        client: ReadOnlyRuntimeClient,
        command: CoreCommand,
    ) -> None:
        await client.send(
            make_ack(
                "session_list",
                request_id=command.id,
                payload={
                    "sessions": self._browser.list_sessions(),
                    "current_session_id": None,
                    "read_only": True,
                },
            )
        )

    async def _handle_session_history(
        self,
        client: ReadOnlyRuntimeClient,
        command: CoreCommand,
    ) -> None:
        session_id = command.payload.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("'session_history.payload.session_id' is required in read-only mode")
        await client.send(
            make_ack(
                "session_history",
                request_id=command.id,
                payload={
                    "session_id": session_id,
                    "message_history": self._browser.read_message_history(session_id),
                    "read_only": True,
                },
            )
        )

    async def _handle_session_search(
        self,
        client: ReadOnlyRuntimeClient,
        command: CoreCommand,
    ) -> None:
        query = command.payload.get("query")
        if not isinstance(query, str):
            raise ValueError("'session_search.payload.query' must be a string")
        limit_raw = command.payload.get("limit", 100)
        if isinstance(limit_raw, bool) or not isinstance(limit_raw, int):
            raise ValueError("'session_search.payload.limit' must be an integer")
        session_id_raw = command.payload.get("session_id")
        if session_id_raw is not None and (
            not isinstance(session_id_raw, str) or not session_id_raw
        ):
            raise ValueError(
                "'session_search.payload.session_id' must be a non-empty string when present"
            )
        case_sensitive = command.payload.get("case_sensitive", False)
        if not isinstance(case_sensitive, bool):
            raise ValueError("'session_search.payload.case_sensitive' must be a boolean")
        whole_word = command.payload.get("whole_word", False)
        if not isinstance(whole_word, bool):
            raise ValueError("'session_search.payload.whole_word' must be a boolean")
        result = self._browser.search(
            query,
            limit=limit_raw,
            session_id=session_id_raw,
            case_sensitive=case_sensitive,
            whole_word=whole_word,
        )
        await client.send(
            make_ack(
                "session_search",
                request_id=command.id,
                payload={"read_only": True, **result},
            )
        )

    def _ready_payload(self) -> dict[str, Any]:
        return {
            "mode": "readonly",
            "server_caps": sorted(READONLY_SERVER_CAPS),
            "status": {
                "agent_state": "READONLY",
                "runner_state": "READONLY",
                "mode": "readonly",
            },
        }
