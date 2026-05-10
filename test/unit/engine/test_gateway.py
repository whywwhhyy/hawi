"""Tests for hawi_engine.gateway: ABC contract, registry, discovery."""

from __future__ import annotations

import argparse

import pytest

from hawi_engine.gateway import (
    GATEWAY_REGISTRY,
    Gateway,
    register_gateway,
    unregister_gateway,
)


class _NoopGateway(Gateway):
    name = "noop_test"

    def register_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--noop-flag", action="store_true")

    async def serve(self, runtime, args) -> None:
        return None


def test_gateway_abc_requires_name():
    with pytest.raises(TypeError):
        Gateway()  # type: ignore[abstract]


def test_register_gateway_adds_to_registry():
    gw = _NoopGateway()
    try:
        register_gateway(gw)
        assert "noop_test" in GATEWAY_REGISTRY
        assert GATEWAY_REGISTRY["noop_test"] is gw
    finally:
        unregister_gateway("noop_test")


def test_register_gateway_rejects_duplicate_name():
    gw1 = _NoopGateway()
    gw2 = _NoopGateway()
    register_gateway(gw1)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_gateway(gw2)
    finally:
        unregister_gateway("noop_test")


def test_unregister_gateway_removes_from_registry():
    gw = _NoopGateway()
    register_gateway(gw)
    assert "noop_test" in GATEWAY_REGISTRY
    unregister_gateway("noop_test")
    assert "noop_test" not in GATEWAY_REGISTRY


def test_unregister_unknown_gateway_is_noop():
    unregister_gateway("does_not_exist")  # must not raise


def test_discover_gateways_idempotent():
    """discover_gateways() can be called multiple times without raising."""
    from hawi_engine.gateway import discover_gateways

    # Plan 4 replaced the standalone WebSocket gateway with HTTP+WS-upgrade.
    # Side-effect import is required so http is registered before assertions.
    from hawi_engine import builtin_gateways  # noqa: F401
    from hawi_engine import http_gateway  # noqa: F401

    discover_gateways()
    discover_gateways()  # second call must not raise
    # Built-ins are still registered.
    assert "stdio" in GATEWAY_REGISTRY
    assert "tcp" in GATEWAY_REGISTRY
    assert "http" in GATEWAY_REGISTRY
    assert "websocket" not in GATEWAY_REGISTRY


def test_discover_gateways_finds_builtins_via_entry_points():
    """The pyproject.toml entry_points should resolve the 3 built-in gateways."""
    from importlib.metadata import entry_points
    eps = entry_points(group="hawi_engine.gateways")
    names = {ep.name for ep in eps}
    # Plan 4: websocket entry-point removed, http added.
    assert {"stdio", "tcp", "http"} <= names
    assert "websocket" not in names
