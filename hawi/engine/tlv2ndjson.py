"""Compatibility alias for the older ``tlv2ndjson`` debug CLI name."""

from __future__ import annotations

import asyncio
import sys

from .tlv2json import _amain, translate_stream


def main() -> None:
    sys.exit(asyncio.run(_amain(sys.argv[1:], prog="tlv2ndjson")))


if __name__ == "__main__":
    main()
