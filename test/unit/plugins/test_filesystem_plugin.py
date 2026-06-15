import json
import os
import tempfile
import shutil
import pytest
from hawi.builtin_plugins.filesystem_plugin.plugin import FileSystemPlugin


def output_text(result) -> str:
    assert isinstance(result.output, str)
    return result.output


class TestFileSystemPlugin:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for file operations."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp, ignore_errors=True)

    @pytest.fixture
    def plugin(self):
        """Create a FileSystemPlugin instance."""
        return FileSystemPlugin()

    def test_read_write_file(self, plugin, temp_dir):
        """Test file read and write tools."""
        file_path = os.path.join(temp_dir, "test.txt")
        content = "Hello, World!"

        # Write new file
        result = plugin.write_file(file_path, content)
        assert result.success is True
        write_output = output_text(result)
        assert write_output.startswith(f"Created file: {os.path.abspath(file_path)}")
        assert "Changed +13/-0 chars, +1/-0 lines." in write_output
        assert os.path.exists(file_path)

        # Clear cache to simulate fresh read (write_file populates cache)
        abs_path = os.path.abspath(file_path)
        plugin._read_state_cache.pop(abs_path, None)

        # Read file
        read_result = plugin.read_file(file_path)
        assert read_result.success is True
        # read_file now returns line-numbered content
        assert "Hello, World!" in output_text(read_result)

    def test_read_file_with_start_line_line_count(self, plugin, temp_dir):
        """Test read_file with start_line and line_count."""
        file_path = os.path.join(temp_dir, "test.txt")
        lines = ["line1\n", "line2\n", "line3\n", "line4\n", "line5\n"]
        with open(file_path, "w") as f:
            f.writelines(lines)

        result = plugin.read_file(file_path, start_line=2, line_count=2)
        assert result.success is True
        content = output_text(result)
        assert "   2|line2" in content
        assert "   3|line3" in content
        assert "line1" not in content.replace("[Lines", "")
        assert "line4" not in content
        assert content.startswith("[Lines 2-3 of 5]")

    def test_read_file_clamps_start_line_and_line_count(self, plugin, temp_dir):
        """read_file should clamp invalid start_line and over-large line_count."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("line1\nline2\nline3\n")

        result = plugin.read_file(file_path, start_line=-10, line_count=50)
        assert result.success is True
        content = output_text(result)
        assert "   1|line1" in content
        assert "   3|line3" in content
        assert "[Lines" not in content

    def test_read_file_defaults_to_limited_line_window(self, plugin, temp_dir, monkeypatch):
        """read_file should not return an entire large file by default."""
        monkeypatch.setattr(FileSystemPlugin, "_READ_FILE_DEFAULT_LINE_COUNT", 3)
        file_path = os.path.join(temp_dir, "large.txt")
        with open(file_path, "w") as f:
            f.write("line1\nline2\nline3\nline4\nline5\n")

        result = plugin.read_file(file_path)

        assert result.success is True
        content = output_text(result)
        assert content.startswith("[Lines 1-3 of 5]")
        assert "line3" in content
        assert "line4" not in content

    def test_read_file_char_seek_with_offset_limit(self, temp_dir):
        """char seek style should expose offset/limit as character range."""
        plugin = FileSystemPlugin(seek_style="char")
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("abc\ndef\n")

        result = plugin.read_file(file_path, offset=2, limit=4, show_line_numbers=False)
        assert result.success is True
        assert output_text(result) == "[Chars 2-6 of 8]\nc\nde"

    def test_read_file_defaults_to_limited_char_window(self, temp_dir, monkeypatch):
        """char seek style should not return an entire large file by default."""
        monkeypatch.setattr(FileSystemPlugin, "_READ_FILE_DEFAULT_CHAR_COUNT", 5)
        plugin = FileSystemPlugin(seek_style="char")
        file_path = os.path.join(temp_dir, "large.txt")
        with open(file_path, "w") as f:
            f.write("abcdefghi")

        result = plugin.read_file(file_path, show_line_numbers=False)

        assert result.success is True
        assert output_text(result) == "[Chars 0-5 of 9]\nabcde"

    def test_read_file_tool_schema_follows_seek_style(self):
        """Only one read_file tool should be exposed for the configured seek style."""
        line_plugin = FileSystemPlugin(seek_style="line")
        line_tools = [tool for tool in line_plugin.tools if tool.name == "read_file"]
        assert len(line_tools) == 1
        assert "start_line" in line_tools[0].parameters_schema["properties"]
        assert "line_count" in line_tools[0].parameters_schema["properties"]
        assert "offset" not in line_tools[0].parameters_schema["properties"]

        char_plugin = FileSystemPlugin(seek_style="char")
        char_tools = [tool for tool in char_plugin.tools if tool.name == "read_file"]
        assert len(char_tools) == 1
        assert "offset" in char_tools[0].parameters_schema["properties"]
        assert "limit" in char_tools[0].parameters_schema["properties"]
        assert "start_line" not in char_tools[0].parameters_schema["properties"]

    def test_filesystem_plugin_rejects_invalid_seek_style(self):
        with pytest.raises(ValueError, match="seek_style"):
            FileSystemPlugin(seek_style="byte")

    def test_read_file_without_line_numbers(self, plugin, temp_dir):
        """read_file should allow suppressing line numbers."""
        file_path = os.path.join(temp_dir, "plain.txt")
        with open(file_path, "w") as f:
            f.write("line1\nline2\n")

        result = plugin.read_file(file_path, show_line_numbers=False)
        assert result.success is True
        assert output_text(result) == "line1\nline2\n"
        assert "   1|" not in output_text(result)

    def test_read_file_cache_respects_line_number_setting(self, plugin, temp_dir):
        """Different formatting options should not return file_unchanged."""
        file_path = os.path.join(temp_dir, "cache.txt")
        with open(file_path, "w") as f:
            f.write("line1\n")

        first = plugin.read_file(file_path, show_line_numbers=False)
        second = plugin.read_file(file_path, show_line_numbers=True)

        assert output_text(first) == "line1\n"
        assert "   1|line1" in output_text(second)

    def test_read_file_reports_language(self, plugin, temp_dir):
        """read_file should include detected language metadata."""
        file_path = os.path.join(temp_dir, "script.py")
        with open(file_path, "w") as f:
            f.write("print('hi')\n")

        result = plugin.read_file(file_path)
        assert result.success is True
        assert "language: python" in output_text(result)

    def test_edit_file(self, plugin, temp_dir):
        """Test edit_file tool."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("foo bar baz")
        plugin.read_file(file_path)

        result = plugin.edit_file(file_path, old_string="bar", new_string="qux")
        assert result.success is True
        assert result.output == "success"

        # Clear cache to read actual file content
        plugin._read_state_cache.pop(os.path.abspath(file_path), None)
        read_result = plugin.read_file(file_path)
        assert "foo qux baz" in output_text(read_result)

    def test_edit_file_replace_all(self, plugin, temp_dir):
        """Test edit_file with replace_all."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("a a a")
        plugin.read_file(file_path)

        result = plugin.edit_file(file_path, old_string="a", new_string="b", replace_all=True)
        assert result.success is True
        assert result.output == "success"

        # Clear cache to read actual file content
        plugin._read_state_cache.pop(os.path.abspath(file_path), None)
        read_result = plugin.read_file(file_path)
        assert "b b b" in output_text(read_result)

    def test_edit_file_no_match(self, plugin, temp_dir):
        """Test edit_file with no match."""
        file_path = os.path.join(temp_dir, "test.txt")
        plugin.write_file(file_path, "hello world")
        plugin.read_file(file_path)

        result = plugin.edit_file(file_path, old_string="nonexistent", new_string="x")
        assert result.success is False
        assert "no match found" in result.error

    def test_glob(self, plugin, temp_dir):
        """Test glob tool."""
        open(os.path.join(temp_dir, "a.py"), "w").close()
        open(os.path.join(temp_dir, "b.txt"), "w").close()
        subdir = os.path.join(temp_dir, "sub")
        os.makedirs(subdir, exist_ok=True)
        open(os.path.join(subdir, "c.py"), "w").close()

        result = plugin.glob("*.py", directory=temp_dir)
        assert result.success is True
        matches = output_text(result).splitlines()
        assert any("a.py" in r for r in matches)
        assert not any("b.txt" in r for r in matches)

    def test_list_dir_non_recursive(self, plugin, temp_dir):
        """list_dir should return direct children as ls-style text."""
        open(os.path.join(temp_dir, "a.py"), "w").close()
        open(os.path.join(temp_dir, "b.txt"), "w").close()
        open(os.path.join(temp_dir, ".hidden"), "w").close()
        subdir = os.path.join(temp_dir, "sub")
        os.makedirs(subdir, exist_ok=True)
        open(os.path.join(subdir, "nested.py"), "w").close()

        result = plugin.list_dir(temp_dir)

        assert result.success is True
        content = output_text(result)
        assert "total " in content
        assert " sub" in content
        assert " a.py" in content
        assert " b.txt" in content
        assert ".hidden" not in content
        assert "nested.py" not in content

    def test_list_dir_recursive_respects_depth_and_hidden_flag(self, plugin, temp_dir):
        """list_dir recursion should obey max_depth and hidden filtering."""
        visible = os.path.join(temp_dir, "visible")
        nested = os.path.join(visible, "nested")
        hidden = os.path.join(temp_dir, ".hidden")
        os.makedirs(nested, exist_ok=True)
        os.makedirs(hidden, exist_ok=True)
        open(os.path.join(visible, "one.txt"), "w").close()
        open(os.path.join(nested, "two.txt"), "w").close()
        open(os.path.join(hidden, "secret.txt"), "w").close()

        result = plugin.list_dir(
            temp_dir,
            recursive=True,
            max_depth=2,
            include_hidden=False,
        )

        assert result.success is True
        content = output_text(result)
        assert "visible:" in content
        assert "nested" in content
        assert "one.txt" in content
        assert "two.txt" not in content
        assert ".hidden" not in content

        with_hidden = plugin.list_dir(temp_dir, include_hidden=True)
        assert ".hidden" in output_text(with_hidden)

    def test_list_dir_limit_truncates_long_directories(self, plugin, temp_dir):
        """list_dir should cap large directory outputs."""
        for i in range(5):
            open(os.path.join(temp_dir, f"{i}.txt"), "w").close()

        result = plugin.list_dir(temp_dir, limit=2)

        assert result.success is True
        content = output_text(result)
        assert content.count(".txt") == 2
        assert "truncated at 2 entries" in content

    def test_list_dir_missing_or_file_path(self, plugin, temp_dir):
        """list_dir should fail cleanly for missing paths and files."""
        missing = plugin.list_dir(os.path.join(temp_dir, "missing"))
        assert missing.success is False
        assert "Directory not found" in missing.error

        file_path = os.path.join(temp_dir, "file.txt")
        open(file_path, "w").close()
        file_result = plugin.list_dir(file_path)
        assert file_result.success is False
        assert "not a directory" in file_result.error

    def test_grep(self, plugin, temp_dir):
        """Test grep tool."""
        file_path = os.path.join(temp_dir, "test.py")
        plugin.write_file(file_path, "def hello():\n    pass\ndef world():\n    pass\n")

        result = plugin.grep("def .*\\(\\):", path=temp_dir)
        assert result.success is True
        content = output_text(result)
        assert "test.py" in content
        assert "hello" in content
        assert "world" in content
        assert "truncated" not in content

    def test_grep_with_file_glob(self, plugin, temp_dir):
        """Test grep with file_glob filter."""
        plugin.write_file(os.path.join(temp_dir, "a.py"), "def foo(): pass\n")
        plugin.write_file(os.path.join(temp_dir, "b.txt"), "def bar(): pass\n")

        result = plugin.grep("def .*\\(\\):", path=temp_dir, file_glob="*.py")
        assert result.success is True
        content = output_text(result)
        assert "a.py" in content
        assert "b.txt" not in content
        assert "foo" in content

    def test_grep_with_glob_alias(self, plugin, temp_dir):
        """grep should accept ClaudeCode-style glob parameter."""
        plugin.write_file(os.path.join(temp_dir, "a.py"), "def foo(): pass\n")
        plugin.write_file(os.path.join(temp_dir, "b.txt"), "def bar(): pass\n")

        result = plugin.grep("def .*\\(\\):", path=temp_dir, glob="*.py")
        assert result.success is True
        content = output_text(result)
        assert os.path.join(temp_dir, "a.py") in content
        assert os.path.join(temp_dir, "b.txt") not in content

    def test_grep_single_file_path(self, plugin, temp_dir):
        """grep should work when path points to a single file."""
        file_path = os.path.join(temp_dir, "single.py")
        plugin.write_file(file_path, "alpha\nbeta\ngamma\n")

        result = plugin.grep("beta", path=file_path)
        assert result.success is True
        assert output_text(result) == f"{file_path}:2: beta"

    def test_grep_default_match_limit_returns_total_count(self, plugin, temp_dir):
        """grep should return the first 1000 matches and expose total matches."""
        plugin._GREP_MAX_CONTENT_BYTES = 10_000_000
        file_path = os.path.join(temp_dir, "matches.txt")
        plugin.write_file(
            file_path,
            "\n".join(f"needle {index}" for index in range(1005)) + "\n",
        )

        result = plugin.grep("needle", path=temp_dir)

        assert result.success is True
        content = output_text(result)
        assert "needle 999" in content
        assert "needle 1000" not in content
        assert "returned 1000 of 1005 matches" in content
        assert "total matches: 1005" in content

    def test_grep_returns_up_to_500_matches_without_byte_truncation(
        self,
        plugin,
        temp_dir,
    ):
        """grep should fully return small result sets even past byte budget."""
        plugin._GREP_MAX_CONTENT_BYTES = 80
        file_path = os.path.join(temp_dir, "matches.txt")
        plugin.write_file(
            file_path,
            "\n".join(f"needle {index} {'x' * 20}" for index in range(500)) + "\n",
        )

        result = plugin.grep("needle", path=temp_dir)

        assert result.success is True
        content = output_text(result)
        assert "needle 499" in content
        assert "truncated" not in content

    def test_grep_content_truncation_ignores_filename_list_truncation(
        self,
        plugin,
        temp_dir,
    ):
        """A shortened filenames list should not mark complete content truncated."""
        plugin._GREP_MAX_FILENAMES = 1
        for index in range(2):
            plugin.write_file(
                os.path.join(temp_dir, f"{index}.txt"),
                f"needle {index}\n",
            )

        result = plugin.grep("needle", path=temp_dir)

        assert result.success is True
        content = output_text(result)
        assert "truncated:" not in content
        assert "needle 0" in content
        assert "needle 1" in content

    def test_grep_default_output_stays_under_tool_result_limit(self, plugin, temp_dir):
        """Default grep limits should avoid the executor's oversized fallback."""
        file_path = os.path.join(temp_dir, "many.txt")
        plugin.write_file(
            file_path,
            "\n".join(f"needle {index} {'x' * 80}" for index in range(2000)) + "\n",
        )

        result = plugin.grep("needle", path=temp_dir)

        payload = json.dumps(
            {
                "success": result.success,
                "output": result.output,
                "error": result.error,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        assert result.success is True
        assert "truncated:" in output_text(result)
        assert len(payload.encode("utf-8")) <= 50 * 1024

    def test_grep_truncates_large_results(self, plugin, temp_dir):
        """grep should cap returned content while preserving total counts."""
        plugin._GREP_MAX_RESULT_LINES = 3
        plugin._GREP_MAX_CONTENT_BYTES = 10_000
        for index in range(5):
            plugin.write_file(
                os.path.join(temp_dir, f"{index}.txt"),
                f"needle {index}\n",
            )

        result = plugin.grep("needle", path=temp_dir)

        assert result.success is True
        content = output_text(result)
        assert "returned 3 of 5 matches" in content
        assert "total matches: 5" in content

    def test_grep_truncates_by_byte_budget(self, plugin, temp_dir):
        """grep should apply the byte budget after the full-result window."""
        plugin._GREP_MAX_RESULT_LINES = 10
        plugin._GREP_MAX_CONTENT_BYTES = 80
        file_path = os.path.join(temp_dir, "large.txt")
        plugin.write_file(
            file_path,
            "needle " + ("x" * 500) + "\n"
            + "\n".join(f"needle {index}" for index in range(500))
            + "\n",
        )

        result = plugin.grep("needle", path=temp_dir)

        assert result.success is True
        assert "[line truncated]" in output_text(result)

    def test_grep_invalid_regex(self, plugin, temp_dir):
        """Invalid regex patterns should return a failure."""
        result = plugin.grep("(", path=temp_dir)
        assert result.success is False
        assert "invalid regex pattern" in result.error

    def test_grep_missing_path(self, plugin):
        """Missing grep paths should return a failure."""
        result = plugin.grep("hello", path="/definitely/not/real/path")
        assert result.success is False
        assert "does not exist" in result.error

    def test_read_file_not_found(self, plugin, temp_dir):
        """read_file returns ToolResult failure for missing file."""
        result = plugin.read_file(os.path.join(temp_dir, "nonexistent.txt"))
        assert result.success is False
        assert "File not found" in result.error

    def test_read_file_unchanged_dedup(self, plugin, temp_dir):
        """Second read without modification returns file_unchanged."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("hello")

        first = plugin.read_file(file_path)
        assert "hello" in output_text(first)

        second = plugin.read_file(file_path)
        assert output_text(second) == f"File unchanged: {os.path.abspath(file_path)}"

    def test_write_file_requires_read_first(self, plugin, temp_dir):
        """Writing to an existing file without reading first fails."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("existing")

        result = plugin.write_file(file_path, "new content")
        assert result.success is False
        assert "MUST read it first" in result.error

    def test_write_file_blocks_system_paths(self, plugin):
        """Writing into protected system directories should be rejected."""
        result = plugin.write_file("/System/test.txt", "blocked")
        assert result.success is False
        assert "Refusing to modify system path" in result.error

    def test_edit_file_requires_read_first(self, plugin, temp_dir):
        """Editing a file without reading first fails."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("existing")

        result = plugin.edit_file(file_path, old_string="existing", new_string="new")
        assert result.success is False
        assert "MUST read" in result.error

    def test_edit_file_blocks_system_paths(self, plugin):
        """Editing protected system directories should be rejected."""
        result = plugin.edit_file("/etc/hosts", old_string="a", new_string="b")
        assert result.success is False
        assert "Refusing to modify system path" in result.error

    def test_optimistic_concurrency_write_file(self, plugin, temp_dir):
        """External modification between read and write fails."""
        file_path = os.path.join(temp_dir, "test.txt")
        plugin.write_file(file_path, "original")
        plugin.read_file(file_path)

        # Simulate external modification
        import time
        time.sleep(0.01)
        with open(file_path, "w") as f:
            f.write("externally modified")

        result = plugin.write_file(file_path, "new content")
        assert result.success is False
        assert "modified externally" in result.error

    def test_optimistic_concurrency_edit_file(self, plugin, temp_dir):
        """External modification between read and edit fails."""
        file_path = os.path.join(temp_dir, "test.txt")
        plugin.write_file(file_path, "original")
        plugin.read_file(file_path)

        # Simulate external modification
        import time
        time.sleep(0.01)
        with open(file_path, "w") as f:
            f.write("externally modified")

        result = plugin.edit_file(file_path, old_string="original", new_string="new")
        assert result.success is False
        assert "modified externally" in result.error

    def test_write_file_recreates_deleted_file_after_read(self, plugin, temp_dir):
        """Deleting a file after read is treated as creating a fresh file."""
        file_path = os.path.join(temp_dir, "test.txt")
        plugin.write_file(file_path, "original")
        plugin.read_file(file_path)
        os.remove(file_path)

        result = plugin.write_file(file_path, "new content")
        assert result.success is True
        assert output_text(result).startswith(f"Created file: {os.path.abspath(file_path)}")
        with open(file_path, "r") as f:
            assert f.read() == "new content"

    def test_edit_file_fails_if_file_deleted_after_read(self, plugin, temp_dir):
        """Deleting a file after read should fail optimistic edit checks."""
        file_path = os.path.join(temp_dir, "test.txt")
        plugin.write_file(file_path, "original")
        plugin.read_file(file_path)
        os.remove(file_path)

        result = plugin.edit_file(file_path, old_string="original", new_string="new")
        assert result.success is False
        assert "deleted or is inaccessible" in result.error

    def test_edit_file_multiple_matches_no_replace_all(self, plugin, temp_dir):
        """Duplicate old_string without replace_all fails."""
        file_path = os.path.join(temp_dir, "test.txt")
        plugin.write_file(file_path, "a a a")
        plugin.read_file(file_path)

        result = plugin.edit_file(file_path, old_string="a", new_string="b")
        assert result.success is False
        assert "found 3 matches" in result.error

    def test_clone_clears_cache(self, plugin, temp_dir):
        """Cloned plugin has empty cache."""
        file_path = os.path.join(temp_dir, "test.txt")
        plugin.write_file(file_path, "hello")
        plugin.read_file(file_path)

        cloned = plugin.clone()
        result = cloned.read_file(file_path)
        # Should return full text, not file_unchanged, because cloned cache is empty
        assert "hello" in output_text(result)

    def test_read_file_binary(self, plugin, temp_dir):
        """read_file returns failure for binary files."""
        file_path = os.path.join(temp_dir, "binary.bin")
        with open(file_path, "wb") as f:
            f.write(b"\x00\x01\x02\x03\xff\xfe")

        result = plugin.read_file(file_path)
        assert result.success is False
        assert "binary" in result.error.lower()

    def test_read_file_empty(self, plugin, temp_dir):
        """read_file correctly handles empty files."""
        file_path = os.path.join(temp_dir, "empty.txt")
        open(file_path, "w").close()

        result = plugin.read_file(file_path)
        assert result.success is True
        assert output_text(result) == ""

    def test_write_file_update_existing(self, plugin, temp_dir):
        """write_file to an existing file returns a compact change summary."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("old content")
        plugin.read_file(file_path)

        result = plugin.write_file(file_path, "new content")
        assert result.success is True
        content = output_text(result)
        assert content.startswith(f"Updated file: {os.path.abspath(file_path)}")
        assert "Changed +11/-11 chars, +1/-1 lines." in content
        assert "new content" not in content
        assert "---" not in content
        assert "@@" not in content

    def test_write_file_creates_nested_directories(self, plugin, temp_dir):
        """write_file automatically creates parent directories."""
        file_path = os.path.join(temp_dir, "a", "b", "c", "deep.txt")
        result = plugin.write_file(file_path, "deep")
        assert result.success is True
        assert os.path.exists(file_path)
        with open(file_path, "r") as f:
            assert f.read() == "deep"

    def test_edit_file_returns_minimal_success(self, plugin, temp_dir):
        """edit_file should not return patch, diff, or original content."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("line1\nline2\nline3\n")
        plugin.read_file(file_path)

        result = plugin.edit_file(file_path, old_string="line2", new_string="LINE2")
        assert result.success is True
        assert result.output == "success"
        assert len(result.output) < 20

    def test_edit_file_updates_cache(self, plugin, temp_dir):
        """After edit_file, subsequent read_file should see unchanged new content."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("alpha")
        plugin.read_file(file_path)

        plugin.edit_file(file_path, old_string="alpha", new_string="beta")

        # Without clearing cache, read should return file_unchanged with the new mtime
        result = plugin.read_file(file_path)
        assert output_text(result) == f"File unchanged: {os.path.abspath(file_path)}"

        # Clear cache and re-read to verify actual disk content
        plugin._read_state_cache.pop(os.path.abspath(file_path), None)
        result = plugin.read_file(file_path)
        assert "beta" in output_text(result)

    def test_write_file_concurrent_ok(self, plugin, temp_dir):
        """write_file succeeds when file has not been modified since read."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("original")
        plugin.read_file(file_path)

        result = plugin.write_file(file_path, "new")
        assert result.success is True
        assert output_text(result).startswith(f"Updated file: {os.path.abspath(file_path)}")

    def test_edit_file_concurrent_ok(self, plugin, temp_dir):
        """edit_file succeeds when file has not been modified since read."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("original")
        plugin.read_file(file_path)

        result = plugin.edit_file(file_path, old_string="original", new_string="updated")
        assert result.success is True
        assert result.output == "success"

    def test_grep_skips_binary_files(self, plugin, temp_dir):
        """grep should skip files with binary extensions."""
        # Write a text file and a fake binary file containing matching text
        text_path = os.path.join(temp_dir, "a.txt")
        bin_path = os.path.join(temp_dir, "b.png")
        with open(text_path, "w") as f:
            f.write("secret: 123\n")
        with open(bin_path, "wb") as f:
            f.write(b"secret: 456\n")

        result = plugin.grep("secret", path=temp_dir)
        assert result.success is True
        content = output_text(result)
        assert "a.txt" in content
        assert "b.png" not in content

    def test_grep_skips_hidden_directories(self, plugin, temp_dir):
        """grep should ignore files under hidden directories."""
        hidden_dir = os.path.join(temp_dir, ".hidden")
        visible_dir = os.path.join(temp_dir, "visible")
        os.makedirs(hidden_dir, exist_ok=True)
        os.makedirs(visible_dir, exist_ok=True)

        with open(os.path.join(hidden_dir, "secret.txt"), "w") as f:
            f.write("needle\n")
        with open(os.path.join(visible_dir, "public.txt"), "w") as f:
            f.write("needle\n")

        result = plugin.grep("needle", path=temp_dir)
        assert result.success is True
        content = output_text(result)
        assert "public.txt" in content
        assert ".hidden" not in content

    def test_glob_absolute_pattern(self, plugin, temp_dir):
        """glob works with absolute path patterns."""
        open(os.path.join(temp_dir, "x.py"), "w").close()
        pattern = os.path.join(temp_dir, "*.py")

        result = plugin.glob(pattern)
        assert result.success is True
        matches = output_text(result).splitlines()
        assert any("x.py" in m for m in matches)

    def test_glob_truncates_large_results(self, plugin, temp_dir):
        """glob should cap returned matches and expose full counts."""
        plugin._GLOB_MAX_MATCHES = 2
        for index in range(4):
            open(os.path.join(temp_dir, f"{index}.txt"), "w").close()

        result = plugin.glob("*.txt", directory=temp_dir)

        assert result.success is True
        content = output_text(result)
        assert len([line for line in content.splitlines() if line.endswith(".txt")]) == 2
        assert "returned 2 of 4 matches" in content

    def test_read_file_structure_python(self, plugin, temp_dir):
        """read_file_structure extracts functions and classes from Python files."""
        file_path = os.path.join(temp_dir, "sample.py")
        content = "\n".join([
            "import os",
            "",
            "def hello(name):",
            "    print(f'Hello {name}')",
            "",
            "class Calculator:",
            "    def add(self, a, b):",
            "        return a + b",
            "",
            "async def fetch_data(url):",
            "    pass",
        ])
        with open(file_path, "w") as f:
            f.write(content)

        result = plugin.read_file(file_path, mode="structure")
        assert result.success is True
        content = output_text(result)
        assert "language: python" in content
        assert "symbols: 4" in content
        assert "function hello: lines 3-5" in content
        assert "class Calculator" in content
        assert "function add" in content
        assert "function fetch_data" in content

    def test_read_file_structure_file_not_found(self, plugin, temp_dir):
        """read_file_structure fails for missing files."""
        result = plugin.read_file(os.path.join(temp_dir, "missing.py"), mode="structure")
        assert result.success is False
        assert "File not found" in result.error

    def test_read_file_structure_binary(self, plugin, temp_dir):
        """read_file_structure fails for binary files."""
        file_path = os.path.join(temp_dir, "data.bin")
        with open(file_path, "wb") as f:
            f.write(b"\x00\x01\x02\xff")
        result = plugin.read_file(file_path, mode="structure")
        assert result.success is False
        assert "binary" in result.error.lower()

    def test_read_file_structure_markdown_sections(self, plugin, temp_dir):
        """read_file_structure should recognize Markdown headings."""
        file_path = os.path.join(temp_dir, "doc.md")
        content = "\n".join([
            "# Title",
            "Some text",
            "## Section 1",
            "Content here",
            "### Sub-section",
            "More content",
        ])
        with open(file_path, "w") as f:
            f.write(content)

        result = plugin.read_file(file_path, mode="structure")
        assert result.success is True
        content = output_text(result)
        assert "symbols: 3" in content
        assert "section Title" in content
        assert "section Section 1" in content

    def test_read_file_structure_empty_file(self, plugin, temp_dir):
        """read_file_structure returns empty symbols for empty file."""
        file_path = os.path.join(temp_dir, "empty.py")
        with open(file_path, "w") as f:
            f.write("")

        result = plugin.read_file(file_path, mode="structure")
        assert result.success is True
        assert "symbols: 0" in output_text(result)

    def test_read_file_language_header(self, plugin, temp_dir):
        """read_file includes language in header for recognized languages."""
        file_path = os.path.join(temp_dir, "script.py")
        with open(file_path, "w") as f:
            f.write("print('hello')\n")

        result = plugin.read_file(file_path)
        assert result.success is True
        content = output_text(result)
        # Header should contain language annotation
        assert "language: python" in content or "python" in content

    def test_read_file_no_line_numbers_fenced(self, plugin, temp_dir):
        """read_file with show_line_numbers=False wraps content in code fence."""
        file_path = os.path.join(temp_dir, "code.py")
        with open(file_path, "w") as f:
            f.write("def foo():\n    pass\n")

        result = plugin.read_file(file_path, show_line_numbers=False)
        assert result.success is True
        content = output_text(result)
        # Should be wrapped in python fenced code block after language header
        assert "```python" in content
