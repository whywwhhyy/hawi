import os
import tempfile
import shutil
import pytest
from hawi.builtin_plugins.filesystem_plugin.plugin import FileSystemPlugin


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
        assert result.output["type"] == "create"
        assert result.output["file_path"] == os.path.abspath(file_path)
        assert os.path.exists(file_path)

        # Clear cache to simulate fresh read (write_file populates cache)
        abs_path = os.path.abspath(file_path)
        plugin._read_state_cache.pop(abs_path, None)

        # Read file
        read_result = plugin.read_file(file_path)
        assert read_result.success is True
        assert read_result.output["type"] == "text"
        assert read_result.output["file"]["filePath"] == os.path.abspath(file_path)
        # read_file now returns line-numbered content
        assert "Hello, World!" in read_result.output["file"]["content"]
        assert read_result.output["file"]["numLines"] == 1
        assert read_result.output["file"]["totalLines"] == 1
        assert read_result.output["file"]["isTruncated"] is False

    def test_read_file_with_offset_limit(self, plugin, temp_dir):
        """Test read_file with offset and limit."""
        file_path = os.path.join(temp_dir, "test.txt")
        lines = ["line1\n", "line2\n", "line3\n", "line4\n", "line5\n"]
        with open(file_path, "w") as f:
            f.writelines(lines)

        result = plugin.read_file(file_path, offset=1, limit=2)
        assert result.success is True
        assert result.output["type"] == "text"
        content = result.output["file"]["content"]
        assert "   2|line2" in content
        assert "   3|line3" in content
        assert "line1" not in content.replace("[Lines", "")
        assert "line4" not in content
        assert result.output["file"]["numLines"] == 2
        assert result.output["file"]["isTruncated"] is True

    def test_read_file_clamps_offset_and_limit(self, plugin, temp_dir):
        """read_file should clamp negative offsets and over-large limits."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("line1\nline2\nline3\n")

        result = plugin.read_file(file_path, offset=-10, limit=50)
        assert result.success is True
        assert result.output["file"]["startLine"] == 0
        assert result.output["file"]["totalLines"] == 3
        assert result.output["file"]["isTruncated"] is False
        assert "   1|line1" in result.output["file"]["content"]
        assert "   3|line3" in result.output["file"]["content"]

    def test_read_file_without_line_numbers(self, plugin, temp_dir):
        """read_file should allow suppressing line numbers."""
        file_path = os.path.join(temp_dir, "plain.txt")
        with open(file_path, "w") as f:
            f.write("line1\nline2\n")

        result = plugin.read_file(file_path, show_line_numbers=False)
        assert result.success is True
        assert result.output["file"]["content"] == "line1\nline2\n"
        assert "   1|" not in result.output["file"]["content"]

    def test_read_file_cache_respects_line_number_setting(self, plugin, temp_dir):
        """Different formatting options should not return file_unchanged."""
        file_path = os.path.join(temp_dir, "cache.txt")
        with open(file_path, "w") as f:
            f.write("line1\n")

        first = plugin.read_file(file_path, show_line_numbers=False)
        second = plugin.read_file(file_path, show_line_numbers=True)

        assert first.output["type"] == "text"
        assert second.output["type"] == "text"
        assert "   1|line1" in second.output["file"]["content"]

    def test_read_file_reports_language(self, plugin, temp_dir):
        """read_file should include detected language metadata."""
        file_path = os.path.join(temp_dir, "script.py")
        with open(file_path, "w") as f:
            f.write("print('hi')\n")

        result = plugin.read_file(file_path)
        assert result.success is True
        assert result.output["file"]["language"] == "python"

    def test_edit_file(self, plugin, temp_dir):
        """Test edit_file tool."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("foo bar baz")
        plugin.read_file(file_path)

        result = plugin.edit_file(file_path, old_string="bar", new_string="qux")
        assert result.success is True
        assert result.output["type"] == "edit"
        assert result.output["replacements_made"] == 1
        assert "structured_patch" in result.output
        assert "git_diff" in result.output

        # Clear cache to read actual file content
        plugin._read_state_cache.pop(os.path.abspath(file_path), None)
        read_result = plugin.read_file(file_path)
        assert "foo qux baz" in read_result.output["file"]["content"]

    def test_edit_file_replace_all(self, plugin, temp_dir):
        """Test edit_file with replace_all."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("a a a")
        plugin.read_file(file_path)

        result = plugin.edit_file(file_path, old_string="a", new_string="b", replace_all=True)
        assert result.success is True
        assert result.output["replacements_made"] == 3

        # Clear cache to read actual file content
        plugin._read_state_cache.pop(os.path.abspath(file_path), None)
        read_result = plugin.read_file(file_path)
        assert "b b b" in read_result.output["file"]["content"]

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
        matches = result.output["matches"]
        assert any("a.py" in r for r in matches)
        assert not any("b.txt" in r for r in matches)

    def test_list_dir_non_recursive(self, plugin, temp_dir):
        """list_dir should return direct children with structured metadata."""
        open(os.path.join(temp_dir, "a.py"), "w").close()
        open(os.path.join(temp_dir, "b.txt"), "w").close()
        open(os.path.join(temp_dir, ".hidden"), "w").close()
        subdir = os.path.join(temp_dir, "sub")
        os.makedirs(subdir, exist_ok=True)
        open(os.path.join(subdir, "nested.py"), "w").close()

        result = plugin.list_dir(temp_dir)

        assert result.success is True
        assert result.output["type"] == "directory"
        assert result.output["path"] == os.path.abspath(temp_dir)
        assert result.output["recursive"] is False
        assert result.output["isTruncated"] is False

        entries = result.output["entries"]
        names = [entry["name"] for entry in entries]
        assert names == ["sub", "a.py", "b.txt"]
        assert ".hidden" not in names
        assert "nested.py" not in names

        sub_entry = entries[0]
        assert sub_entry["type"] == "directory"
        assert sub_entry["relativePath"] == "sub"
        assert sub_entry["depth"] == 1
        assert "size" in sub_entry
        assert "mtime" in sub_entry

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
        paths = [entry["relativePath"] for entry in result.output["entries"]]
        assert "visible" in paths
        assert os.path.join("visible", "nested") in paths
        assert os.path.join("visible", "one.txt") in paths
        assert os.path.join("visible", "nested", "two.txt") not in paths
        assert ".hidden" not in paths

        with_hidden = plugin.list_dir(temp_dir, include_hidden=True)
        hidden_paths = [entry["relativePath"] for entry in with_hidden.output["entries"]]
        assert ".hidden" in hidden_paths

    def test_list_dir_limit_truncates_long_directories(self, plugin, temp_dir):
        """list_dir should cap large directory outputs."""
        for i in range(5):
            open(os.path.join(temp_dir, f"{i}.txt"), "w").close()

        result = plugin.list_dir(temp_dir, limit=2)

        assert result.success is True
        assert result.output["numEntries"] == 2
        assert len(result.output["entries"]) == 2
        assert result.output["isTruncated"] is True

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
        assert result.output["mode"] == "content"
        assert result.output["numFiles"] == 1
        assert result.output["numLines"] == 2
        assert "test.py" in result.output["filenames"][0]
        assert "hello" in result.output["content"]
        assert "world" in result.output["content"]

    def test_grep_with_file_glob(self, plugin, temp_dir):
        """Test grep with file_glob filter."""
        plugin.write_file(os.path.join(temp_dir, "a.py"), "def foo(): pass\n")
        plugin.write_file(os.path.join(temp_dir, "b.txt"), "def bar(): pass\n")

        result = plugin.grep("def .*\\(\\):", path=temp_dir, file_glob="*.py")
        assert result.success is True
        assert result.output["numFiles"] == 1
        assert "foo" in result.output["content"]

    def test_grep_with_glob_alias(self, plugin, temp_dir):
        """grep should accept ClaudeCode-style glob parameter."""
        plugin.write_file(os.path.join(temp_dir, "a.py"), "def foo(): pass\n")
        plugin.write_file(os.path.join(temp_dir, "b.txt"), "def bar(): pass\n")

        result = plugin.grep("def .*\\(\\):", path=temp_dir, glob="*.py")
        assert result.success is True
        assert result.output["numFiles"] == 1
        assert result.output["filenames"] == [os.path.join(temp_dir, "a.py")]

    def test_grep_single_file_path(self, plugin, temp_dir):
        """grep should work when path points to a single file."""
        file_path = os.path.join(temp_dir, "single.py")
        plugin.write_file(file_path, "alpha\nbeta\ngamma\n")

        result = plugin.grep("beta", path=file_path)
        assert result.success is True
        assert result.output["filenames"] == [file_path]
        assert result.output["numFiles"] == 1
        assert result.output["numLines"] == 1
        assert result.output["content"] == f"{file_path}:2: beta"

    def test_grep_invalid_regex(self, plugin, temp_dir):
        """Invalid regex patterns should return a structured failure."""
        result = plugin.grep("(", path=temp_dir)
        assert result.success is False
        assert "invalid regex pattern" in result.error

    def test_grep_missing_path(self, plugin):
        """Missing grep paths should return a structured failure."""
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
        assert first.output["type"] == "text"

        second = plugin.read_file(file_path)
        assert second.output["type"] == "file_unchanged"
        assert second.output["file"]["filePath"] == os.path.abspath(file_path)

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
        assert result.output["type"] == "create"
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
        assert result.output["type"] == "text"
        assert "hello" in result.output["file"]["content"]

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
        assert result.output["type"] == "text"
        assert result.output["file"]["content"] == ""
        assert result.output["file"]["numLines"] == 0
        assert result.output["file"]["totalLines"] == 0
        assert result.output["file"]["isTruncated"] is False

    def test_write_file_update_existing(self, plugin, temp_dir):
        """write_file to an existing file returns type 'update' with original_content and patch."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("old content")
        plugin.read_file(file_path)

        result = plugin.write_file(file_path, "new content")
        assert result.success is True
        assert result.output["type"] == "update"
        assert result.output["original_content"] == "old content"
        assert len(result.output["structured_patch"]) > 0
        assert "new content" in result.output["git_diff"]

    def test_write_file_creates_nested_directories(self, plugin, temp_dir):
        """write_file automatically creates parent directories."""
        file_path = os.path.join(temp_dir, "a", "b", "c", "deep.txt")
        result = plugin.write_file(file_path, "deep")
        assert result.success is True
        assert os.path.exists(file_path)
        with open(file_path, "r") as f:
            assert f.read() == "deep"

    def test_edit_file_patch_content(self, plugin, temp_dir):
        """edit_file returns meaningful structured_patch and git_diff."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("line1\nline2\nline3\n")
        plugin.read_file(file_path)

        result = plugin.edit_file(file_path, old_string="line2", new_string="LINE2")
        assert result.success is True
        hunks = result.output["structured_patch"]
        assert len(hunks) > 0
        # At least one hunk line should show the removal and addition
        all_lines = "\n".join(line for hunk in hunks for line in hunk["lines"])
        assert "-line2" in all_lines
        assert "+LINE2" in all_lines
        assert "line2" in result.output["git_diff"]
        assert "LINE2" in result.output["git_diff"]

    def test_edit_file_updates_cache(self, plugin, temp_dir):
        """After edit_file, subsequent read_file should see unchanged new content."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("alpha")
        plugin.read_file(file_path)

        plugin.edit_file(file_path, old_string="alpha", new_string="beta")

        # Without clearing cache, read should return file_unchanged with the new mtime
        result = plugin.read_file(file_path)
        assert result.output["type"] == "file_unchanged"

        # Clear cache and re-read to verify actual disk content
        plugin._read_state_cache.pop(os.path.abspath(file_path), None)
        result = plugin.read_file(file_path)
        assert "beta" in result.output["file"]["content"]

    def test_write_file_concurrent_ok(self, plugin, temp_dir):
        """write_file succeeds when file has not been modified since read."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("original")
        plugin.read_file(file_path)

        result = plugin.write_file(file_path, "new")
        assert result.success is True
        assert result.output["type"] == "update"

    def test_edit_file_concurrent_ok(self, plugin, temp_dir):
        """edit_file succeeds when file has not been modified since read."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("original")
        plugin.read_file(file_path)

        result = plugin.edit_file(file_path, old_string="original", new_string="updated")
        assert result.success is True
        assert result.output["type"] == "edit"

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
        assert result.output["numLines"] == 1
        assert "a.txt" in result.output["content"]
        assert "b.png" not in result.output["content"]
        assert result.output["numFiles"] == 1

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
        assert result.output["numLines"] == 1
        assert "public.txt" in result.output["content"]
        assert ".hidden" not in result.output["content"]
        assert result.output["filenames"] == [os.path.join(visible_dir, "public.txt")]

    def test_glob_absolute_pattern(self, plugin, temp_dir):
        """glob works with absolute path patterns."""
        open(os.path.join(temp_dir, "x.py"), "w").close()
        pattern = os.path.join(temp_dir, "*.py")

        result = plugin.glob(pattern)
        assert result.success is True
        matches = result.output["matches"]
        assert any("x.py" in m for m in matches)

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
        assert result.output["type"] == "structure"
        assert result.output["language"] == "python"
        assert result.output["numSymbols"] == 4

        # Check symbol types and names
        symbols = result.output["symbols"]
        assert symbols[0]["type"] == "function"
        assert symbols[0]["name"] == "hello"
        assert symbols[0]["line"] == 3  # 1-based

        assert symbols[1]["type"] == "class"
        assert symbols[1]["name"] == "Calculator"

        assert symbols[2]["type"] == "function"
        assert symbols[2]["name"] == "add"  # Nested method within class

        assert symbols[3]["type"] == "function"
        assert symbols[3]["name"] == "fetch_data"

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
        assert result.output["numSymbols"] == 3

        symbols = result.output["symbols"]
        assert symbols[0]["type"] == "section"
        assert "Title" in symbols[0]["name"]
        assert symbols[1]["type"] == "section"
        assert "Section 1" in symbols[1]["name"]

    def test_read_file_structure_empty_file(self, plugin, temp_dir):
        """read_file_structure returns empty symbols for empty file."""
        file_path = os.path.join(temp_dir, "empty.py")
        with open(file_path, "w") as f:
            f.write("")

        result = plugin.read_file(file_path, mode="structure")
        assert result.success is True
        assert result.output["numSymbols"] == 0

    def test_read_file_language_header(self, plugin, temp_dir):
        """read_file includes language in header for recognized languages."""
        file_path = os.path.join(temp_dir, "script.py")
        with open(file_path, "w") as f:
            f.write("print('hello')\n")

        result = plugin.read_file(file_path)
        assert result.success is True
        content = result.output["file"]["content"]
        # Header should contain language annotation
        assert "language: python" in content or "python" in content

    def test_read_file_no_line_numbers_fenced(self, plugin, temp_dir):
        """read_file with show_line_numbers=False wraps content in code fence."""
        file_path = os.path.join(temp_dir, "code.py")
        with open(file_path, "w") as f:
            f.write("def foo():\n    pass\n")

        result = plugin.read_file(file_path, show_line_numbers=False)
        assert result.success is True
        content = result.output["file"]["content"]
        # Should be wrapped in python fenced code block after language header
        assert "```python" in content
