"""Unit and property-based tests for MarkdownStreamingParser."""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from hawi.utils.markdown_streaming_parser import (
    MarkdownStreamingParser,
    BlockType,
    RenderEvent,
    BlockUpdate,
    BlockCommit,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def collect_events(text: str, chunk_size: int = 1) -> list[RenderEvent]:
    parser = MarkdownStreamingParser()
    events = []
    for i in range(0, len(text), chunk_size):
        events.extend(parser.stream(text[i : i + chunk_size]))
    events.extend(parser.flush())
    return events


# ---------------------------------------------------------------------------
# Task 4: Unit Tests
# ---------------------------------------------------------------------------

class TestStreamEmpty:
    """4.1 stream("") returns empty sequence."""

    def test_empty_string_returns_no_events(self):
        parser = MarkdownStreamingParser()
        result = list(parser.stream(""))
        assert result == []

    def test_empty_string_does_not_change_state(self):
        parser = MarkdownStreamingParser()
        list(parser.stream(""))
        # flush should also return nothing
        assert list(parser.flush()) == []


class TestSingleLineParagraph:
    """4.2 Single-line paragraph produces BlockUpdate + BlockCommit."""

    def test_complete_line_produces_update_then_commit(self):
        parser = MarkdownStreamingParser()
        events = list(parser.stream("hello\n"))
        # Should have at least one BlockUpdate followed by a BlockCommit
        updates = [e for e in events if isinstance(e, BlockUpdate)]
        commits = [e for e in events if isinstance(e, BlockCommit)]
        assert len(updates) >= 1
        assert len(commits) == 1
        # The last update should come before the commit
        last_update_idx = max(i for i, e in enumerate(events) if isinstance(e, BlockUpdate))
        commit_idx = next(i for i, e in enumerate(events) if isinstance(e, BlockCommit))
        assert last_update_idx < commit_idx

    def test_block_update_content_matches(self):
        parser = MarkdownStreamingParser()
        events = list(parser.stream("hello\n"))
        # The last BlockUpdate before commit should have full content
        updates = [e for e in events if isinstance(e, BlockUpdate)]
        last_update = updates[-1]
        assert last_update.content == "hello\n"

    def test_block_commit_content_matches(self):
        parser = MarkdownStreamingParser()
        events = list(parser.stream("hello\n"))
        commit = next(e for e in events if isinstance(e, BlockCommit))
        assert commit.content == "hello\n"

    def test_same_block_id_in_update_and_commit(self):
        parser = MarkdownStreamingParser()
        events = list(parser.stream("hello\n"))
        update = next(e for e in events if isinstance(e, BlockUpdate))
        commit = next(e for e in events if isinstance(e, BlockCommit))
        assert update.block_id == commit.block_id

    def test_block_type_is_paragraph(self):
        parser = MarkdownStreamingParser()
        events = list(parser.stream("hello\n"))
        commit = next(e for e in events if isinstance(e, BlockCommit))
        assert commit.block_type == BlockType.paragraph


class TestBlockTypeDetection:
    """4.3 Block type prefix detection for all types."""

    def test_heading_h1(self):
        events = collect_events("# Title\n")
        assert any(e.block_type == BlockType.heading for e in events)

    def test_heading_h2(self):
        events = collect_events("## Section\n")
        assert any(e.block_type == BlockType.heading for e in events)

    def test_heading_h6(self):
        events = collect_events("###### Deep\n")
        assert any(e.block_type == BlockType.heading for e in events)

    def test_code_backtick(self):
        events = collect_events("```python\ncode\n```\n")
        assert any(e.block_type == BlockType.code for e in events)

    def test_code_tilde(self):
        events = collect_events("~~~\ncode\n~~~\n")
        assert any(e.block_type == BlockType.code for e in events)

    def test_list_item_dash(self):
        events = collect_events("- item\n")
        assert any(e.block_type == BlockType.list_item for e in events)

    def test_list_item_star(self):
        events = collect_events("* item\n")
        assert any(e.block_type == BlockType.list_item for e in events)

    def test_list_item_plus(self):
        events = collect_events("+ item\n")
        assert any(e.block_type == BlockType.list_item for e in events)

    def test_list_item_ordered(self):
        events = collect_events("1. item\n")
        assert any(e.block_type == BlockType.list_item for e in events)

    def test_blockquote(self):
        events = collect_events("> quote\n")
        assert any(e.block_type == BlockType.blockquote for e in events)

    def test_thematic_break_dashes(self):
        events = collect_events("---\n")
        assert any(e.block_type == BlockType.thematic_break for e in events)

    def test_thematic_break_stars(self):
        events = collect_events("***\n")
        assert any(e.block_type == BlockType.thematic_break for e in events)

    def test_thematic_break_underscores(self):
        events = collect_events("___\n")
        assert any(e.block_type == BlockType.thematic_break for e in events)

    def test_table(self):
        events = collect_events("| a | b |\n| - | - |\n\n")
        assert any(e.block_type == BlockType.table for e in events)

    def test_paragraph_fallback(self):
        events = collect_events("plain text\n")
        assert any(e.block_type == BlockType.paragraph for e in events)


class TestFencedBlockDetection:
    """4.4 Fenced code block open/close marker detection."""

    def test_backtick_fence_open_and_close(self):
        text = "```python\nprint('hi')\n```\n"
        events = collect_events(text)
        commits = [e for e in events if isinstance(e, BlockCommit)]
        assert len(commits) == 1
        assert commits[0].content == text

    def test_tilde_fence_open_and_close(self):
        text = "~~~\nsome code\n~~~\n"
        events = collect_events(text)
        commits = [e for e in events if isinstance(e, BlockCommit)]
        assert len(commits) == 1
        assert commits[0].content == text

    def test_fenced_block_only_updates_until_close(self):
        text = "```\nline1\nline2\n```\n"
        events = collect_events(text)
        # All events before the final commit should be BlockUpdate
        commit_idx = next(i for i, e in enumerate(events) if isinstance(e, BlockCommit))
        for e in events[:commit_idx]:
            assert isinstance(e, BlockUpdate)

    def test_fenced_block_type_is_code(self):
        text = "```\ncode\n```\n"
        events = collect_events(text)
        # The BlockCommit should have code type
        commits = [e for e in events if isinstance(e, BlockCommit)]
        assert len(commits) == 1
        assert commits[0].block_type == BlockType.code
        # BlockUpdates after the first line is determined should also be code
        code_updates = [e for e in events if isinstance(e, BlockUpdate) and e.block_type == BlockType.code]
        assert len(code_updates) >= 1


class TestTableDelayedCommit:
    """4.5 Table multi-line delayed commit: blank line triggers commit."""

    def test_table_commits_on_blank_line(self):
        text = "| a | b |\n| - | - |\n| 1 | 2 |\n\n"
        events = collect_events(text)
        commits = [e for e in events if isinstance(e, BlockCommit)]
        assert len(commits) == 1
        assert commits[0].block_type == BlockType.table

    def test_table_commit_content_includes_all_rows(self):
        text = "| a | b |\n| - | - |\n| 1 | 2 |\n\n"
        events = collect_events(text)
        commit = next(e for e in events if isinstance(e, BlockCommit))
        assert "| a | b |" in commit.content
        assert "| - | - |" in commit.content
        assert "| 1 | 2 |" in commit.content

    def test_table_only_updates_before_commit(self):
        text = "| a |\n| b |\n\n"
        events = collect_events(text)
        commit_idx = next(i for i, e in enumerate(events) if isinstance(e, BlockCommit))
        for e in events[:commit_idx]:
            assert isinstance(e, BlockUpdate)


class TestTableFollowedByNonTableLine:
    """4.6 Table followed by non-| line: that line starts a new block."""

    def test_non_table_line_triggers_commit(self):
        text = "| a |\n| b |\nsome paragraph\n"
        events = collect_events(text)
        commits = [e for e in events if isinstance(e, BlockCommit)]
        # Should have at least 2 commits: table + paragraph
        assert len(commits) >= 2

    def test_non_table_line_not_discarded(self):
        text = "| a |\n| b |\nsome paragraph\n"
        events = collect_events(text)
        commits = [e for e in events if isinstance(e, BlockCommit)]
        # The paragraph commit should contain the non-table line
        para_commits = [c for c in commits if c.block_type == BlockType.paragraph]
        assert len(para_commits) >= 1
        assert "some paragraph" in para_commits[0].content

    def test_table_commit_does_not_include_non_table_line(self):
        text = "| a |\n| b |\nsome paragraph\n"
        events = collect_events(text)
        table_commits = [e for e in events if isinstance(e, BlockCommit) and e.block_type == BlockType.table]
        assert len(table_commits) == 1
        assert "some paragraph" not in table_commits[0].content


class TestFlushFencedMode:
    """4.7 flush() in FENCED mode emits unclosed content as BlockCommit."""

    def test_flush_unclosed_fenced_block(self):
        parser = MarkdownStreamingParser()
        list(parser.stream("```python\nprint('hi')\n"))
        flush_events = list(parser.flush())
        assert len(flush_events) == 1
        assert isinstance(flush_events[0], BlockCommit)

    def test_flush_fenced_content_is_complete(self):
        parser = MarkdownStreamingParser()
        list(parser.stream("```python\nprint('hi')\n"))
        flush_events = list(parser.flush())
        commit = flush_events[0]
        assert "```python" in commit.content
        assert "print('hi')" in commit.content

    def test_flush_fenced_block_type_is_code(self):
        parser = MarkdownStreamingParser()
        list(parser.stream("```\ncode\n"))
        flush_events = list(parser.flush())
        assert flush_events[0].block_type == BlockType.code


class TestFlushTableMode:
    """4.8 flush() in TABLE mode emits current table content as BlockCommit."""

    def test_flush_table_without_blank_line(self):
        parser = MarkdownStreamingParser()
        list(parser.stream("| a |\n| b |\n"))
        flush_events = list(parser.flush())
        assert len(flush_events) == 1
        assert isinstance(flush_events[0], BlockCommit)

    def test_flush_table_content_complete(self):
        parser = MarkdownStreamingParser()
        list(parser.stream("| a |\n| b |\n"))
        flush_events = list(parser.flush())
        commit = flush_events[0]
        assert "| a |" in commit.content
        assert "| b |" in commit.content

    def test_flush_table_block_type(self):
        parser = MarkdownStreamingParser()
        list(parser.stream("| a |\n"))
        flush_events = list(parser.flush())
        assert flush_events[0].block_type == BlockType.table


class TestReset:
    """4.9 reset() clears state without emitting events."""

    def test_reset_returns_none(self):
        parser = MarkdownStreamingParser()
        result = parser.reset()
        assert result is None

    def test_reset_emits_no_events(self):
        parser = MarkdownStreamingParser()
        list(parser.stream("hello"))
        parser.reset()
        # After reset, flush should return nothing
        assert list(parser.flush()) == []

    def test_reset_clears_block_counter(self):
        parser = MarkdownStreamingParser()
        list(parser.stream("line1\n"))
        parser.reset()
        events = list(parser.stream("line2\n"))
        # block_id should restart from "0"
        assert events[0].block_id == "0"

    def test_reset_mid_fenced_block(self):
        parser = MarkdownStreamingParser()
        list(parser.stream("```\ncode"))
        parser.reset()
        assert list(parser.flush()) == []

    def test_reset_mid_table(self):
        parser = MarkdownStreamingParser()
        list(parser.stream("| a |\n"))
        parser.reset()
        assert list(parser.flush()) == []


class TestFlushEmpty:
    """4.10 flush() with no in-progress block returns empty sequence."""

    def test_flush_on_fresh_parser(self):
        parser = MarkdownStreamingParser()
        assert list(parser.flush()) == []

    def test_flush_after_complete_block(self):
        parser = MarkdownStreamingParser()
        list(parser.stream("hello\n"))
        assert list(parser.flush()) == []

    def test_flush_after_previous_flush(self):
        parser = MarkdownStreamingParser()
        list(parser.stream("hello"))
        list(parser.flush())
        assert list(parser.flush()) == []


class TestInlineMarkdownPreserved:
    """4.11 Inline markdown syntax is preserved in paragraph content."""

    def test_inline_link_preserved(self):
        """Test that [text](url) link syntax is preserved in paragraph."""
        events = collect_events("Check [Google](https://google.com) for more\n")
        commit = next(e for e in events if isinstance(e, BlockCommit))
        assert "[Google](https://google.com)" in commit.content

    def test_inline_bold_preserved(self):
        """Test that **bold** syntax is preserved."""
        events = collect_events("This is **bold** text\n")
        commit = next(e for e in events if isinstance(e, BlockCommit))
        assert "**bold**" in commit.content

    def test_inline_italic_preserved(self):
        """Test that *italic* syntax is preserved."""
        events = collect_events("This is *italic* text\n")
        commit = next(e for e in events if isinstance(e, BlockCommit))
        assert "*italic*" in commit.content

    def test_inline_code_preserved(self):
        """Test that `code` syntax is preserved."""
        events = collect_events("Use `print()` function\n")
        commit = next(e for e in events if isinstance(e, BlockCommit))
        assert "`print()`" in commit.content

    def test_multiple_inline_elements(self):
        """Test multiple inline elements in one paragraph."""
        text = "See [docs](https://example.com) for **bold** and *italic* and `code` examples\n"
        events = collect_events(text)
        commit = next(e for e in events if isinstance(e, BlockCommit))
        assert "[docs](https://example.com)" in commit.content
        assert "**bold**" in commit.content
        assert "*italic*" in commit.content
        assert "`code`" in commit.content


# ---------------------------------------------------------------------------
# Task 5: Property-Based Tests (hypothesis)
# ---------------------------------------------------------------------------

# Strategy: printable text that may contain markdown-like structures
markdown_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd", "Zs"),
        whitelist_characters="# `~|>-*+_\n$.",
    ),
    min_size=0,
    max_size=200,
)

# Strategy for chunk sizes
chunk_size_st = st.integers(min_value=1, max_value=20)


class TestProperty1RoundTrip:
    """5.1 Property 1: Content completeness round-trip."""

    # Feature: markdown-streaming-parser, Property 1: 内容完整性 Round-Trip — 随机文本任意分割后 BlockCommit.content 拼接等于原文
    @given(text=markdown_text, chunk_size=chunk_size_st)
    @settings(max_examples=100)
    def test_round_trip(self, text: str, chunk_size: int):
        # Skip pure whitespace-only text (e.g., just newlines) as it doesn't produce blocks
        assume(text.strip())
        events = collect_events(text, chunk_size=chunk_size)
        commits = [e for e in events if isinstance(e, BlockCommit)]
        reconstructed = "".join(c.content for c in commits)
        assert reconstructed == text


class TestProperty2MonotonicContent:
    """5.2 Property 2: BlockUpdate content monotonically non-decreasing."""

    # Feature: markdown-streaming-parser, Property 2: BlockUpdate content 单调递增 — 同一 block 的 content 长度单调不减
    @given(text=markdown_text, chunk_size=chunk_size_st)
    @settings(max_examples=100)
    def test_monotonic_content(self, text: str, chunk_size: int):
        events = collect_events(text, chunk_size=chunk_size)
        # Group BlockUpdates by block_id
        updates_by_block: dict[str, list[BlockUpdate]] = {}
        for e in events:
            if isinstance(e, BlockUpdate):
                updates_by_block.setdefault(e.block_id, []).append(e)
        for block_id, updates in updates_by_block.items():
            for i in range(1, len(updates)):
                assert len(updates[i].content) >= len(updates[i - 1].content), (
                    f"block {block_id}: content shrank at index {i}"
                )


class TestProperty3DeltaEqualsContent:
    """5.3 Property 3: Cumulative deltas equal final content snapshot."""

    # Feature: markdown-streaming-parser, Property 3: delta 累加等价于 content 快照 — 所有 delta 拼接等于最后一个 content
    @given(text=markdown_text, chunk_size=chunk_size_st)
    @settings(max_examples=100)
    def test_delta_sum_equals_content(self, text: str, chunk_size: int):
        events = collect_events(text, chunk_size=chunk_size)
        updates_by_block: dict[str, list[BlockUpdate]] = {}
        for e in events:
            if isinstance(e, BlockUpdate):
                updates_by_block.setdefault(e.block_id, []).append(e)
        for block_id, updates in updates_by_block.items():
            delta_sum = "".join(u.delta for u in updates)
            last_content = updates[-1].content
            assert delta_sum == last_content, (
                f"block {block_id}: delta sum '{delta_sum}' != last content '{last_content}'"
            )


class TestProperty4BlockIdConsistency:
    """5.4 Property 4: block_id consistent within a block."""

    # Feature: markdown-streaming-parser, Property 4: block_id 在同一 block 内保持一致 — 所有事件的 block_id 相同
    @given(text=markdown_text, chunk_size=chunk_size_st)
    @settings(max_examples=100)
    def test_block_id_consistent(self, text: str, chunk_size: int):
        events = collect_events(text, chunk_size=chunk_size)
        # Each BlockCommit should have a matching block_id with its preceding BlockUpdates
        # Collect all block_ids seen in updates, then verify commit uses same id
        updates_by_block: dict[str, list[BlockUpdate]] = {}
        commits_by_block: dict[str, list[BlockCommit]] = {}
        for e in events:
            if isinstance(e, BlockUpdate):
                updates_by_block.setdefault(e.block_id, []).append(e)
            elif isinstance(e, BlockCommit):
                commits_by_block.setdefault(e.block_id, []).append(e)
        # Every block_id in commits should also appear in updates (or be a standalone commit)
        for block_id, commits in commits_by_block.items():
            assert len(commits) == 1, f"block {block_id} committed more than once"


class TestProperty5NoUpdateAfterCommit:
    """5.5 Property 5: No BlockUpdate after BlockCommit for same block_id."""

    # Feature: markdown-streaming-parser, Property 5: BlockCommit 后不再发出 BlockUpdate — 相同 block_id 不再出现 BlockUpdate
    @given(text=markdown_text, chunk_size=chunk_size_st)
    @settings(max_examples=100)
    def test_no_update_after_commit(self, text: str, chunk_size: int):
        events = collect_events(text, chunk_size=chunk_size)
        committed_ids: set[str] = set()
        for e in events:
            if isinstance(e, BlockCommit):
                committed_ids.add(e.block_id)
            elif isinstance(e, BlockUpdate):
                assert e.block_id not in committed_ids, (
                    f"BlockUpdate for block {e.block_id} after BlockCommit"
                )


class TestProperty6FencedTableOnlyUpdatesBeforeCommit:
    """5.6 Property 6: Fenced/Table blocks only emit BlockUpdate before completion."""

    # Feature: markdown-streaming-parser, Property 6: Fenced/Table Block 在完成前只发 BlockUpdate — 完成前无 BlockCommit
    @given(
        body=st.text(
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"),
                whitelist_characters=" \n",
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_fenced_block_no_commit_before_close(self, body: str):
        # Build a complete fenced block
        text = f"```\n{body}\n```\n"
        events = collect_events(text)
        # Find the code block events
        code_events = [e for e in events if e.block_type == BlockType.code]
        if not code_events:
            return
        commit_idx = next(
            (i for i, e in enumerate(code_events) if isinstance(e, BlockCommit)), None
        )
        if commit_idx is not None:
            for e in code_events[:commit_idx]:
                assert isinstance(e, BlockUpdate)

    @given(
        rows=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=" "),
                min_size=1,
                max_size=10,
            ),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=100)
    def test_table_block_no_commit_before_blank_line(self, rows: list[str]):
        table_text = "".join(f"| {row} |\n" for row in rows) + "\n"
        events = collect_events(table_text)
        table_events = [e for e in events if e.block_type == BlockType.table]
        if not table_events:
            return
        commit_idx = next(
            (i for i, e in enumerate(table_events) if isinstance(e, BlockCommit)), None
        )
        if commit_idx is not None:
            for e in table_events[:commit_idx]:
                assert isinstance(e, BlockUpdate)


class TestProperty7ResetEquivalence:
    """5.7 Property 7: After reset(), behavior equals a fresh instance."""

    # Feature: markdown-streaming-parser, Property 7: reset 后状态等价于新建实例 — 随机时刻 reset 后行为与新建实例一致
    @given(
        pre_text=markdown_text,
        post_text=markdown_text,
        chunk_size=chunk_size_st,
    )
    @settings(max_examples=100)
    def test_reset_equivalence(self, pre_text: str, post_text: str, chunk_size: int):
        # Parser that processes pre_text then resets
        parser_reset = MarkdownStreamingParser()
        for i in range(0, len(pre_text), chunk_size):
            list(parser_reset.stream(pre_text[i : i + chunk_size]))
        parser_reset.reset()

        # Fresh parser
        parser_fresh = MarkdownStreamingParser()

        # Both should produce identical events for post_text
        events_reset = []
        events_fresh = []
        for i in range(0, len(post_text), chunk_size):
            events_reset.extend(parser_reset.stream(post_text[i : i + chunk_size]))
            events_fresh.extend(parser_fresh.stream(post_text[i : i + chunk_size]))
        events_reset.extend(parser_reset.flush())
        events_fresh.extend(parser_fresh.flush())

        # Compare event types, block_types, and contents (not block_ids which may differ)
        assert len(events_reset) == len(events_fresh), (
            f"Event count mismatch: {len(events_reset)} vs {len(events_fresh)}"
        )
        for r, f in zip(events_reset, events_fresh):
            assert type(r) == type(f)
            assert r.block_type == f.block_type
            if isinstance(r, BlockCommit):
                assert r.content == f.content
            elif isinstance(r, BlockUpdate):
                assert r.content == f.content
                assert r.delta == f.delta


class TestProperty8BlockTypeStable:
    """5.8 Property 8: block_type stable after first line determined."""

    # Feature: markdown-streaming-parser, Property 8: block_type 在首行确定后不变 — 首行确定后所有事件的 block_type 相同
    @given(text=markdown_text, chunk_size=chunk_size_st)
    @settings(max_examples=100)
    def test_block_type_stable(self, text: str, chunk_size: int):
        events = collect_events(text, chunk_size=chunk_size)
        # Group all events by block_id
        events_by_block: dict[str, list[RenderEvent]] = {}
        for e in events:
            events_by_block.setdefault(e.block_id, []).append(e)

        for block_id, block_events in events_by_block.items():
            # Find the first event after a newline has been seen (first BlockUpdate with \n in content)
            first_newline_idx = None
            for i, e in enumerate(block_events):
                if isinstance(e, BlockUpdate) and "\n" in e.content:
                    first_newline_idx = i
                    break
            if first_newline_idx is None:
                continue
            # All events from first_newline_idx onward should have same block_type
            determined_type = block_events[first_newline_idx].block_type
            for e in block_events[first_newline_idx:]:
                assert e.block_type == determined_type, (
                    f"block {block_id}: type changed from {determined_type} to {e.block_type}"
                )


class TestProperty9TableContentIntegrity:
    """5.9 Property 9: Table BlockCommit.content contains all table rows."""

    # Feature: markdown-streaming-parser, Property 9: Table Block 内容完整性 — BlockCommit.content 包含所有表格行，无行丢失
    @given(
        rows=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=" "),
                min_size=1,
                max_size=15,
            ),
            min_size=1,
            max_size=6,
        ),
        chunk_size=chunk_size_st,
    )
    @settings(max_examples=100)
    def test_table_all_rows_in_commit(self, rows: list[str], chunk_size: int):
        table_lines = [f"| {row} |\n" for row in rows]
        text = "".join(table_lines) + "\n"
        events = collect_events(text, chunk_size=chunk_size)
        table_commits = [
            e for e in events
            if isinstance(e, BlockCommit) and e.block_type == BlockType.table
        ]
        assert len(table_commits) == 1
        commit_content = table_commits[0].content
        for line in table_lines:
            assert line in commit_content, (
                f"Table row '{line.strip()}' missing from commit content"
            )
