"""Gateway abstraction for the Hawi engine.

A Gateway is a long-running listener that accepts client connections, wraps
each in a QueuedJsonClient subclass, and dispatches incoming JSON frames to
the CoreRuntime. Built-in gateways live in builtin_gateways.py; third-party
gateways register via the `hawi_engine.gateways` entry-point group.
"""

from __future__ import annotations

import argparse
import logging
from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import CoreRuntime

logger = logging.getLogger(__name__)


class Gateway(ABC):
    """Per-listener gateway contract.

    Subclasses must define:
      - `name`: a unique identifier exposed via `--gateway <name>`.
      - `register_args(parser)`: add gateway-specific argparse args.
      - `serve(runtime, args)`: run the listener until shutdown.
    """

    name: str

    @abstractmethod
    def register_args(self, parser: argparse.ArgumentParser) -> None:
        """Add gateway-specific options to the argparse parser."""

    @abstractmethod
    async def serve(self, runtime: "CoreRuntime", args: argparse.Namespace) -> None:
        """Run the gateway until the runtime is shutdown."""


GATEWAY_REGISTRY: dict[str, Gateway] = {}


def register_gateway(gateway: Gateway) -> None:
    """Register a gateway instance under its `name`."""
    if gateway.name in GATEWAY_REGISTRY:
        raise ValueError(f"Gateway {gateway.name!r} already registered")
    GATEWAY_REGISTRY[gateway.name] = gateway


def unregister_gateway(name: str) -> None:
    """Remove a gateway from the registry. No-op if not present."""
    GATEWAY_REGISTRY.pop(name, None)


def discover_gateways() -> None:
    """Load third-party gateways from the `hawi_engine.gateways` entry-point group.

    Each entry-point must resolve to a Gateway instance (or a zero-arg callable
    returning one). Errors are logged but do not abort startup — a missing
    optional gateway should not break the engine.
    """
    try:
        eps = entry_points(group="hawi_engine.gateways")
    except TypeError:
        # Older API: entry_points() returns a dict
        eps = entry_points().get("hawi_engine.gateways", [])

    for ep in eps:
        try:
            obj = ep.load()
            gateway = obj() if callable(obj) and not isinstance(obj, Gateway) else obj
            if not isinstance(gateway, Gateway):
                logger.warning(
                    "Entry point %s did not resolve to a Gateway instance; got %r",
                    ep.name,
                    type(gateway).__name__,
                )
                continue
            if gateway.name in GATEWAY_REGISTRY:
                logger.debug("Gateway %r already registered; skipping entry point %s",
                             gateway.name, ep.name)
                continue
            register_gateway(gateway)
            logger.info("Discovered gateway %r from entry point %s", gateway.name, ep.name)
        except Exception:
            logger.exception("Failed to load gateway entry point %s", ep.name)
