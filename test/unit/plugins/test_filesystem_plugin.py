import os
import tempfile
import shutil
import pytest
from hawi_plugins.filesystem_plugin.plugin import FileSystemPlugin


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
        # read_file now returns line-numbered content
        assert "Hello, World!" in read_result.output["content"]
        assert read_result.output["total_lines"] == 1

    def test_read_file_with_offset_limit(self, plugin, temp_dir):
        """Test read_file with offset and limit."""
        file_path = os.path.join(temp_dir, "test.txt")
        lines = ["line1\n", "line2\n", "line3\n", "line4\n", "line5\n"]
        with open(file_path, "w") as f:
            f.writelines(lines)

        result = plugin.read_file(file_path, offset=1, limit=2)
        assert result.success is True
        assert result.output["type"] == "text"
        content = result.output["content"]
        assert "   2|line2" in content
        assert "   3|line3" in content
        assert "line1" not in content.replace("[Lines", "")
        assert "line4" not in content
        assert result.output["is_partial_view"] is True

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
        assert "foo qux baz" in read_result.output["content"]

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
        assert "b b b" in read_result.output["content"]

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

    def test_grep(self, plugin, temp_dir):
        """Test grep tool."""
        file_path = os.path.join(temp_dir, "test.py")
        plugin.write_file(file_path, "def hello():\n    pass\ndef world():\n    pass\n")

        result = plugin.grep("def .*\\(\\):", path=temp_dir)
        assert result.success is True
        matches = result.output["matches"]
        assert len(matches) == 2
        assert any("hello" in r for r in matches)
        assert any("world" in r for r in matches)

    def test_grep_with_file_glob(self, plugin, temp_dir):
        """Test grep with file_glob filter."""
        plugin.write_file(os.path.join(temp_dir, "a.py"), "def foo(): pass\n")
        plugin.write_file(os.path.join(temp_dir, "b.txt"), "def bar(): pass\n")

        result = plugin.grep("def .*\\(\\):", path=temp_dir, file_glob="*.py")
        assert result.success is True
        matches = result.output["matches"]
        assert len(matches) == 1
        assert "foo" in matches[0]

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
        assert second.output["file_path"] == os.path.abspath(file_path)

    def test_write_file_requires_read_first(self, plugin, temp_dir):
        """Writing to an existing file without reading first fails."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("existing")

        result = plugin.write_file(file_path, "new content")
        assert result.success is False
        assert "MUST read it first" in result.error

    def test_edit_file_requires_read_first(self, plugin, temp_dir):
        """Editing a file without reading first fails."""
        file_path = os.path.join(temp_dir, "test.txt")
        with open(file_path, "w") as f:
            f.write("existing")

        result = plugin.edit_file(file_path, old_string="existing", new_string="new")
        assert result.success is False
        assert "MUST read" in result.error

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
        assert "hello" in result.output["content"]

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
        assert result.output["content"] == ""
        assert result.output["total_lines"] == 0
        assert result.output["is_partial_view"] is False

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
        assert "beta" in result.output["content"]

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
        matches = result.output["matches"]
        assert len(matches) == 1
        assert "a.txt" in matches[0]
        assert "b.png" not in matches[0]

    def test_glob_absolute_pattern(self, plugin, temp_dir):
        """glob works with absolute path patterns."""
        open(os.path.join(temp_dir, "x.py"), "w").close()
        pattern = os.path.join(temp_dir, "*.py")

        result = plugin.glob(pattern)
        assert result.success is True
        matches = result.output["matches"]
        assert any("x.py" in m for m in matches)
