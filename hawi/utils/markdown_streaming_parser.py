"""Markdown streaming parser for LLM delta streams.

Converts streaming markdown delta fragments into structured RenderEvent sequences,
enabling O(n) rendering without FOIM flicker or terminal height jitter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable


# ---------------------------------------------------------------------------
# Public API: BlockType
# ---------------------------------------------------------------------------

class BlockType(Enum):
    paragraph      = "paragraph"
    heading        = "heading"
    code           = "code"
    list_item      = "list_item"
    blockquote     = "blockquote"
    math           = "math"
    thematic_break = "thematic_break"
    table          = "table"


# ---------------------------------------------------------------------------
# Public API: RenderEvent hierarchy
# ---------------------------------------------------------------------------

@dataclass
class RenderEvent:
    block_id:   str
    block_type: BlockType


@dataclass
class BlockUpdate(RenderEvent):
    content: str  # cumulative snapshot
    delta:   str  # newly added fragment in this call


@dataclass
class BlockCommit(RenderEvent):
    content: str  # complete block text


# ---------------------------------------------------------------------------
# Internal: parser state and block buffer
# ---------------------------------------------------------------------------

class _ParserState(Enum):
    NORMAL = auto()
    FENCED = auto()
    TABLE  = auto()


@dataclass
class _BlockBuffer:
    block_id:       str
    block_type:     BlockType
    content:        str = ""
    fence_marker:   str | None = None  # closing marker expected in FENCED mode
    type_finalized: bool = False       # True once the first \n has been seen


# ---------------------------------------------------------------------------
# Public API: MarkdownStreamingParser
# ---------------------------------------------------------------------------

class MarkdownStreamingParser:
    """State-machine parser that converts markdown delta streams to RenderEvents.

    Architecture notes:
    - _line_buf accumulates the current incomplete line across multiple stream() calls.
      It is only cleared when a complete line (ending with \\n) is processed.
    - _committed_len tracks how many chars of _line_buf have been "committed" to
      _current_block.content via partial-line BlockUpdate previews in NORMAL mode.
    - In FENCED/TABLE modes, _line_buf is never previewed mid-line; events are only
      emitted when a complete line arrives.
    """

    def __init__(self) -> None:
        self._block_counter: int = 0
        self._state: _ParserState = _ParserState.NORMAL
        self._current_block: _BlockBuffer | None = None
        # Current incomplete line buffer — persists across stream() calls
        self._line_buf: str = ""
        # How many chars of _line_buf have already been added to _current_block.content
        # (used in NORMAL mode for partial-line previews)
        self._committed_len: int = 0

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _new_block_id(self) -> str:
        bid = str(self._block_counter)
        self._block_counter += 1
        return bid

    def _detect_block_type(self, first_line: str) -> BlockType:
        """Identify block type from the first line (prefix matching)."""
        stripped = first_line.rstrip("\n")

        if re.match(r"^#{1,6}(\s|$)", stripped):
            return BlockType.heading
        if stripped.startswith("```") or stripped.startswith("~~~"):
            return BlockType.code
        if stripped == "$" or stripped == "$$":
            return BlockType.math
        if re.match(r"^[-*+] ", stripped) or re.match(r"^\d+\. ", stripped):
            return BlockType.list_item
        if stripped.startswith(">"):
            return BlockType.blockquote
        if re.match(r"^(-{3,}|\*{3,}|_{3,})\s*$", stripped):
            return BlockType.thematic_break
        if stripped.startswith("|"):
            return BlockType.table
        return BlockType.paragraph

    def _detect_fence_open(self, line: str) -> str | None:
        """If line opens a fenced block, return the expected closing marker; else None."""
        stripped = line.rstrip("\n")
        if stripped.startswith("```"):
            return "```"
        if stripped.startswith("~~~"):
            return "~~~"
        if stripped == "$" or stripped == "$$":
            return stripped
        return None

    def _is_fence_close(self, line: str, marker: str) -> bool:
        """Return True if line is the matching closing marker for a fenced block."""
        stripped = line.rstrip("\n")
        return stripped == marker

    def _is_table_line(self, line: str) -> bool:
        return line.startswith("|")

    def _ensure_block(self, block_type: BlockType = BlockType.paragraph) -> _BlockBuffer:
        if self._current_block is None:
            self._current_block = _BlockBuffer(
                block_id=self._new_block_id(),
                block_type=block_type,
            )
        return self._current_block

    def _commit_block(self) -> BlockCommit:
        assert self._current_block is not None
        commit = BlockCommit(
            block_id=self._current_block.block_id,
            block_type=self._current_block.block_type,
            content=self._current_block.content,
        )
        self._current_block = None
        self._state = _ParserState.NORMAL
        self._committed_len = 0
        return commit

    # -----------------------------------------------------------------------
    # Line processing
    # -----------------------------------------------------------------------

    def _process_complete_line(self, line: str, prev_committed: int = 0) -> list[RenderEvent]:
        """Process a complete line (ending with \\n) according to current state."""
        if self._state == _ParserState.FENCED:
            return self._process_fenced_line(line)
        if self._state == _ParserState.TABLE:
            return self._process_table_line(line)
        return self._process_normal_line(line, prev_committed)

    def _process_fenced_line(self, line: str) -> list[RenderEvent]:
        events: list[RenderEvent] = []
        blk = self._ensure_block()
        blk.content += line
        if self._is_fence_close(line, blk.fence_marker):  # type: ignore[arg-type]
            events.append(self._commit_block())
        else:
            events.append(BlockUpdate(
                block_id=blk.block_id,
                block_type=blk.block_type,
                content=blk.content,
                delta=line,
            ))
        return events

    def _process_table_line(self, line: str) -> list[RenderEvent]:
        events: list[RenderEvent] = []
        blk = self._ensure_block(BlockType.table)

        if line == "\n":
            # blank line → commit table (include the blank line in content)
            blk.content += line
            events.append(self._commit_block())
        elif self._is_table_line(line):
            blk.content += line
            events.append(BlockUpdate(
                block_id=blk.block_id,
                block_type=blk.block_type,
                content=blk.content,
                delta=line,
            ))
        else:
            # non-'|' line → commit table, then process line as new block
            events.append(self._commit_block())
            events.extend(self._process_normal_line(line))
        return events

    def _process_normal_line(self, line: str, prev_committed: int = 0) -> list[RenderEvent]:
        """Process a complete line in NORMAL state.

        prev_committed: how many chars of this line were already previewed via
        partial BlockUpdate events (so delta = line[prev_committed:]).
        """
        events: list[RenderEvent] = []

        # blank line → commit current block if any
        if line == "\n":
            if self._current_block is not None:
                # Block content was already set to partial content via previews.
                # The blank line itself is not part of the block content in this design.
                events.append(self._commit_block())
            return events

        fence_marker = self._detect_fence_open(line)

        if fence_marker is not None:
            # Commit any existing finalized block first
            if self._current_block is not None and self._current_block.type_finalized:
                events.append(self._commit_block())
            # Start (or reuse) a fenced block
            blk = self._ensure_block(self._detect_block_type(line))
            blk.block_type = self._detect_block_type(line)
            blk.fence_marker = fence_marker
            blk.content = line  # full first line
            blk.type_finalized = True
            self._state = _ParserState.FENCED
            delta = line[prev_committed:]
            events.append(BlockUpdate(
                block_id=blk.block_id,
                block_type=blk.block_type,
                content=blk.content,
                delta=delta,
            ))
            return events

        if self._is_table_line(line):
            # Commit any existing finalized block first
            if self._current_block is not None and self._current_block.type_finalized:
                events.append(self._commit_block())
            blk = self._ensure_block(BlockType.table)
            blk.block_type = BlockType.table
            blk.content = line  # full first line
            blk.type_finalized = True
            self._state = _ParserState.TABLE
            delta = line[prev_committed:]
            events.append(BlockUpdate(
                block_id=blk.block_id,
                block_type=blk.block_type,
                content=blk.content,
                delta=delta,
            ))
            return events

        # Regular line
        blk = self._ensure_block(self._detect_block_type(line))
        if not blk.type_finalized:
            blk.block_type = self._detect_block_type(line)
        blk.content = line  # full line
        blk.type_finalized = True
        delta = line[prev_committed:]
        events.append(BlockUpdate(
            block_id=blk.block_id,
            block_type=blk.block_type,
            content=blk.content,
            delta=delta,
        ))
        events.append(self._commit_block())
        return events

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def stream(self, part: str) -> Iterable[RenderEvent]:
        """Feed a delta fragment; yields zero or more RenderEvents."""
        if not part:
            return

        events: list[RenderEvent] = []

        for ch in part:
            self._line_buf += ch

            if ch == "\n":
                line = self._line_buf
                prev_committed = self._committed_len
                self._line_buf = ""
                self._committed_len = 0
                events.extend(self._process_complete_line(line, prev_committed))
            elif self._state == _ParserState.NORMAL:
                # Emit a partial-line BlockUpdate preview in NORMAL mode.
                new_chars = self._line_buf[self._committed_len:]
                if new_chars:
                    blk = self._ensure_block(self._detect_block_type(self._line_buf))
                    if not blk.type_finalized:
                        blk.block_type = self._detect_block_type(self._line_buf)
                    blk.content = self._line_buf
                    self._committed_len = len(self._line_buf)
                    events.append(BlockUpdate(
                        block_id=blk.block_id,
                        block_type=blk.block_type,
                        content=blk.content,
                        delta=new_chars,
                    ))
            # In FENCED/TABLE modes, no partial-line events — wait for \n

        yield from events

    def flush(self) -> Iterable[RenderEvent]:
        """Commit any in-progress block as a final BlockCommit."""
        events: list[RenderEvent] = []

        # If there's a partial line in _line_buf, finalize it into the block
        if self._line_buf:
            partial = self._line_buf
            self._line_buf = ""
            self._committed_len = 0

            if self._state == _ParserState.FENCED:
                blk = self._ensure_block()
                blk.content += partial
                events.append(BlockUpdate(
                    block_id=blk.block_id,
                    block_type=blk.block_type,
                    content=blk.content,
                    delta=partial,
                ))
            elif self._state == _ParserState.TABLE:
                blk = self._ensure_block(BlockType.table)
                blk.content += partial
                events.append(BlockUpdate(
                    block_id=blk.block_id,
                    block_type=blk.block_type,
                    content=blk.content,
                    delta=partial,
                ))
            else:
                # NORMAL: blk.content already has the partial content from stream() previews.
                # Just ensure the block exists with the right type.
                blk = self._ensure_block(self._detect_block_type(partial))
                if not blk.type_finalized:
                    blk.block_type = self._detect_block_type(partial)
                # blk.content is already set to partial from the last stream() preview.
                # If no previews were emitted (e.g., first stream() call had no partial),
                # set it now.
                if blk.content != partial:
                    blk.content = partial
                # Emit a final BlockUpdate only if content changed since last preview
                # (i.e., if there's new content not yet previewed)
                # Since _committed_len was reset to 0 above, we check if blk.content == partial
                # which it should be. No new delta to emit here.

        if self._current_block is not None:
            events.append(self._commit_block())

        return events

    def reset(self) -> None:
        """Reset all internal state; equivalent to creating a new instance."""
        self._block_counter = 0
        self._state = _ParserState.NORMAL
        self._current_block = None
        self._line_buf = ""
        self._committed_len = 0


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "BlockType",
    "RenderEvent",
    "BlockUpdate",
    "BlockCommit",
    "MarkdownStreamingParser",
]
