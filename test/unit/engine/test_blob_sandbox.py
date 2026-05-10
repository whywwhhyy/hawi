"""Tests for blob path sandbox primitives."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hawi_engine.blob.sandbox import (
    Direction,
    SandboxViolation,
    blob_id_to_relpath,
    resolve_blob_path,
    validate_blob_id,
)


def test_validate_blob_id_accepts_hex_64():
    bid = "a" * 64
    assert validate_blob_id(bid) == bid


@pytest.mark.parametrize("bad", [
    "",
    "abc",                        # too short
    "a" * 65,                     # too long
    "g" * 64,                     # non-hex
    "../etc/passwd",
    "ab/cd",
    "ab\\cd",
    "AB" + "0" * 62,              # mixed case rejected (we lowercase normalize)
    "../" + "a" * 64,
])
def test_validate_blob_id_rejects_bad(bad):
    with pytest.raises(SandboxViolation):
        validate_blob_id(bad)


def test_blob_id_to_relpath_layout():
    bid = "ab" + "0" * 62
    rp = blob_id_to_relpath(bid)
    # Layout: <first2>/<remaining62>
    assert rp == Path("ab") / ("0" * 62)


def test_resolve_blob_path_under_sandbox(tmp_path):
    sandbox_root = tmp_path / ".hawi" / "blobs"
    bid = "ab" + "0" * 62
    resolved = resolve_blob_path(sandbox_root, "inbound", bid)
    assert resolved.is_relative_to(sandbox_root / "inbound")
    assert resolved == sandbox_root / "inbound" / "ab" / ("0" * 62)


def test_resolve_blob_path_rejects_escape(tmp_path):
    sandbox_root = tmp_path / ".hawi" / "blobs"
    # Even with a "valid-looking" id but a relpath that escapes — our id-to-path
    # function never produces .., so this should be impossible. Test the defensive
    # check anyway:
    with pytest.raises(SandboxViolation):
        resolve_blob_path(sandbox_root, "inbound", "../" + "a" * 64)


def test_resolve_blob_path_rejects_unknown_direction(tmp_path):
    sandbox_root = tmp_path / ".hawi" / "blobs"
    with pytest.raises(SandboxViolation):
        resolve_blob_path(sandbox_root, "sideways", "a" * 64)  # type: ignore[arg-type]


def test_direction_literal_values():
    assert Direction.__args__ == ("inbound", "outbound")  # type: ignore[attr-defined]
