import difflib
import fnmatch
import glob as glob_module
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

from hawi.plugin import HawiPlugin, tool
from hawi.tool import ToolResult


@dataclass
class FileReadState:
    content: str
    mtime: float
    timestamp: float
    offset: Optional[int]
    limit: Optional[int]
    show_line_numbers: bool


class FileSystemPlugin(HawiPlugin):
    """
    文件系统操作插件，提供文件读写、编辑、搜索能力。

    对应 Claude Code 的文件操作工具族：
    - read_file: 读取文件内容（支持分页、行号格式、缓存去重）
    - write_file: 覆盖写入文件（要求先读取，支持乐观并发控制）
    - edit_file: 精确字符串替换编辑（要求先读取，支持乐观并发控制）
    - glob: 基于 glob 模式查找文件
    - grep: 基于正则搜索文件内容
    """

    # 常见二进制文件扩展名，grep 时跳过
    _BINARY_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2",
        ".exe", ".dll", ".so", ".dylib", ".bin",
        ".mp3", ".mp4", ".avi", ".mkv", ".mov", ".wav",
        ".ttf", ".otf", ".woff", ".woff2",
        ".db", ".sqlite", ".sqlite3",
        ".pyc", ".pyo", ".class", ".o", ".a",
    }
    _SYSTEM_WRITE_PREFIXES = (
        "/System",
        "/Library",
        "/bin",
        "/sbin",
        "/etc",
        "/private/etc",
        "/usr",
    )

    def __init__(self):
        self._read_state_cache: dict[str, FileReadState] = {}

    def clone(self) -> "FileSystemPlugin":
        """Return a fresh FileSystemPlugin with an empty read state cache."""
        new_plugin = FileSystemPlugin()
        new_plugin._read_state_cache = {}
        return new_plugin

    def _resolve_path(self, file_path: str) -> str:
        return os.path.abspath(file_path)

    def _format_lines(self, lines: list[str], start_line: int) -> str:
        """Format lines with cat-n style line numbers (1-based)."""
        formatted = []
        for i, line in enumerate(lines, start=start_line + 1):
            formatted.append(f"{i:4d}|{line}")
        return "".join(formatted)

    def _detect_language(self, file_path: str) -> str:
        """Best-effort language detection from file name/extension."""
        basename = os.path.basename(file_path).lower()
        extension = os.path.splitext(basename)[1]

        special_names = {
            "dockerfile": "dockerfile",
            "makefile": "makefile",
        }
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascriptreact",
            ".ts": "typescript",
            ".tsx": "typescriptreact",
            ".json": "json",
            ".md": "markdown",
            ".markdown": "markdown",
            ".yml": "yaml",
            ".yaml": "yaml",
            ".toml": "toml",
            ".sh": "shell",
            ".bash": "shell",
            ".zsh": "shell",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".swift": "swift",
            ".kt": "kotlin",
            ".sql": "sql",
            ".xml": "xml",
            ".txt": "text",
        }

        return special_names.get(basename, extension_map.get(extension, "text"))

    def _is_system_write_path(self, file_path: str) -> bool:
        for prefix in self._SYSTEM_WRITE_PREFIXES:
            try:
                if os.path.commonpath([file_path, prefix]) == prefix:
                    return True
            except ValueError:
                continue
        return False

    def _validate_write_path(self, abs_path: str) -> Optional[ToolResult]:
        if self._is_system_write_path(abs_path):
            return ToolResult(
                success=False,
                error=(
                    f"Refusing to modify system path '{abs_path}'. "
                    "Please choose a path outside protected system directories."
                ),
            )
        return None

    def _check_concurrency(self, abs_path: str, state: FileReadState) -> Optional[ToolResult]:
        try:
            current_mtime = os.path.getmtime(abs_path)
        except OSError:
            return ToolResult(
                success=False,
                error=f"File '{abs_path}' was deleted or is inaccessible since it was last read. Please re-read the file before modifying.",
            )
        if current_mtime != state.mtime:
            return ToolResult(
                success=False,
                error=f"File '{abs_path}' has been modified externally since it was last read. Please re-read the file before modifying.",
            )
        return None

    def _atomic_write(self, file_path: str, content: str) -> None:
        dir_name = os.path.dirname(file_path) or "."
        os.makedirs(dir_name, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, file_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

    def _generate_structured_patch(
        self, old_content: str, new_content: str, file_path: str = "file"
    ) -> tuple[list[dict], str]:
        old_lines = old_content.splitlines(keepends=False)
        new_lines = new_content.splitlines(keepends=False)

        git_diff = "\n".join(
            difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=file_path,
                tofile=file_path,
            )
        )

        sm = difflib.SequenceMatcher(None, old_lines, new_lines)
        hunks: list[dict] = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            hunk_lines: list[str] = []
            context_start = max(i1 - 2, 0)
            for k in range(context_start, i1):
                hunk_lines.append(f" {old_lines[k]}")
            for k in range(i1, i2):
                hunk_lines.append(f"-{old_lines[k]}")
            for k in range(j1, j2):
                hunk_lines.append(f"+{new_lines[k]}")
            context_end = min(i2 + 2, len(old_lines))
            for k in range(i2, context_end):
                hunk_lines.append(f" {old_lines[k]}")

            hunks.append(
                {
                    "old_start": i1 + 1,
                    "old_lines": i2 - i1,
                    "new_start": j1 + 1,
                    "new_lines": j2 - j1,
                    "lines": hunk_lines,
                }
            )

        return hunks, git_diff

    @tool
    def read_file(
        self,
        file_path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        show_line_numbers: bool = True,
    ) -> ToolResult:
        """
        读取文件内容，支持分页读取和行号格式输出。

        Args:
            file_path: 文件路径（绝对路径或相对于当前工作目录）
            offset: 起始行号（0-based），用于大文件分块读取
            limit: 读取的最大行数
        """
        abs_path = self._resolve_path(file_path)

        if not os.path.exists(abs_path):
            return ToolResult(success=False, error=f"File not found: {abs_path}")

        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                error=f"File '{abs_path}' appears to be binary and cannot be read as text.",
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Error reading file: {e}")

        try:
            mtime = os.path.getmtime(abs_path)
        except Exception as e:
            return ToolResult(success=False, error=f"Error checking file: {e}")

        cached = self._read_state_cache.get(abs_path)
        if (
            cached is not None
            and cached.mtime == mtime
            and cached.offset == offset
            and cached.limit == limit
            and cached.show_line_numbers == show_line_numbers
        ):
            return ToolResult(
                success=True,
                output={
                    "type": "file_unchanged",
                    "file": {
                        "filePath": abs_path,
                    },
                },
            )

        lines = content.splitlines(keepends=True)
        total = len(lines)
        start = offset if offset is not None else 0
        end = (start + limit) if limit is not None else total

        if start < 0:
            start = 0
        if end > total:
            end = total

        selected = lines[start:end]
        formatted = (
            self._format_lines(selected, start)
            if show_line_numbers
            else "".join(selected)
        )
        is_partial = (start > 0) or (end < total)
        returned_line_count = len(selected)
        detected_language = self._detect_language(abs_path)

        header = ""
        if is_partial:
            header = f"[Lines {start}-{end} of {total}]\n"

        self._read_state_cache[abs_path] = FileReadState(
            content=content,
            mtime=mtime,
            timestamp=time.time(),
            offset=offset,
            limit=limit,
            show_line_numbers=show_line_numbers,
        )

        return ToolResult(
            success=True,
            output={
                "type": "text",
                "file": {
                    "filePath": abs_path,
                    "content": header + formatted,
                    "numLines": returned_line_count,
                    "startLine": start,
                    "totalLines": total,
                    "language": detected_language,
                    "isTruncated": is_partial,
                },
            },
        )

    @tool
    def write_file(self, file_path: str, content: str) -> ToolResult:
        """
        覆盖写入文件内容。如果文件已存在，必须先使用 read_file 读取。

        Args:
            file_path: 要写入的文件路径
            content: 文件内容
        """
        abs_path = self._resolve_path(file_path)
        path_error = self._validate_write_path(abs_path)
        if path_error is not None:
            return path_error
        file_exists = os.path.exists(abs_path)

        if file_exists and abs_path not in self._read_state_cache:
            return ToolResult(
                success=False,
                error=(
                    f"File '{abs_path}' already exists. "
                    f"You MUST read it first using read_file before writing."
                ),
            )

        original_content: Optional[str] = None
        if file_exists:
            state = self._read_state_cache[abs_path]
            concurrency_error = self._check_concurrency(abs_path, state)
            if concurrency_error is not None:
                return concurrency_error
            original_content = state.content

        try:
            self._atomic_write(abs_path, content)
        except Exception as e:
            return ToolResult(success=False, error=f"Error writing file: {e}")

        try:
            new_mtime = os.path.getmtime(abs_path)
        except Exception as e:
            return ToolResult(success=False, error=f"Error verifying written file: {e}")

        self._read_state_cache[abs_path] = FileReadState(
            content=content,
            mtime=new_mtime,
            timestamp=time.time(),
            offset=None,
            limit=None,
            show_line_numbers=True,
        )

        hunks, git_diff = self._generate_structured_patch(
            original_content or "", content, file_path=abs_path
        )

        return ToolResult(
            success=True,
            output={
                "type": "update" if file_exists else "create",
                "file_path": abs_path,
                "structured_patch": hunks,
                "original_content": original_content,
                "git_diff": git_diff,
            },
        )

    @tool
    def edit_file(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> ToolResult:
        """
        精确编辑文件内容（字符串替换）。
        必须先使用 read_file 读取文件。如果 replace_all 为 false，old_string 必须唯一匹配。

        Args:
            file_path: 要编辑的文件路径
            old_string: 要替换的精确文本
            new_string: 用于替换的新文本
            replace_all: 如果为 true，替换所有匹配项；否则要求 old_string 必须唯一匹配
        """
        abs_path = self._resolve_path(file_path)
        path_error = self._validate_write_path(abs_path)
        if path_error is not None:
            return path_error

        if abs_path not in self._read_state_cache:
            return ToolResult(
                success=False,
                error=(
                    f"You MUST read '{abs_path}' using read_file before editing."
                ),
            )

        state = self._read_state_cache[abs_path]
        concurrency_error = self._check_concurrency(abs_path, state)
        if concurrency_error is not None:
            return concurrency_error

        original_content = state.content
        matches = original_content.count(old_string)

        if matches == 0:
            return ToolResult(
                success=False,
                error=f'Error: no match found for "{old_string}"',
            )
        if not replace_all and matches > 1:
            return ToolResult(
                success=False,
                error=(
                    f"Error: found {matches} matches, but replace_all is false. "
                    f"Please use a more specific old_string or set replace_all to true."
                ),
            )

        new_content = (
            original_content.replace(old_string, new_string)
            if replace_all
            else original_content.replace(old_string, new_string, 1)
        )

        try:
            self._atomic_write(abs_path, new_content)
        except Exception as e:
            return ToolResult(success=False, error=f"Error writing file after editing: {e}")

        try:
            new_mtime = os.path.getmtime(abs_path)
        except Exception as e:
            return ToolResult(
                success=False, error=f"Error verifying edited file: {e}"
            )

        self._read_state_cache[abs_path] = FileReadState(
            content=new_content,
            mtime=new_mtime,
            timestamp=time.time(),
            offset=None,
            limit=None,
            show_line_numbers=True,
        )

        hunks, git_diff = self._generate_structured_patch(
            original_content, new_content, file_path=abs_path
        )

        return ToolResult(
            success=True,
            output={
                "type": "edit",
                "file_path": abs_path,
                "replacements_made": matches if replace_all else 1,
                "structured_patch": hunks,
                "original_content": original_content,
                "git_diff": git_diff,
            },
        )

    @tool
    def glob(self, pattern: str, directory: Optional[str] = None) -> ToolResult:
        """
        基于 glob 模式查找文件。

        Args:
            pattern: glob 模式，如 '*.py' 或 'src/**/*.ts'
            directory: 搜索的根目录，默认为当前工作目录
        """
        root = directory or os.getcwd()
        if os.path.isabs(pattern):
            search_pattern = pattern
        else:
            search_pattern = os.path.join(root, pattern)
        try:
            results = glob_module.glob(search_pattern, recursive=True)
            return ToolResult(
                success=True,
                output={"matches": sorted(results)},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Error: {e}")

    @tool
    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
        file_glob: Optional[str] = None,
    ) -> ToolResult:
        """
        基于正则搜索文件内容。

        Args:
            pattern: 正则表达式模式
            path: 搜索的文件或目录路径，默认为当前工作目录
            file_glob: 用于过滤文件名的 glob 模式，如 '*.py'
        """
        target = path or os.getcwd()
        active_glob = glob or file_glob

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResult(
                success=False,
                error=f"Error: invalid regex pattern: {e}",
            )

        files_to_search = []
        if os.path.isfile(target):
            files_to_search.append(target)
        elif os.path.isdir(target):
            for root, _, files in os.walk(target):
                if any(part.startswith(".") for part in root.split(os.sep) if part):
                    continue
                for filename in files:
                    filepath = os.path.join(root, filename)
                    if any(filename.lower().endswith(ext) for ext in self._BINARY_EXTENSIONS):
                        continue
                    if active_glob and not fnmatch.fnmatch(filename, active_glob):
                        continue
                    files_to_search.append(filepath)
        else:
            return ToolResult(
                success=False,
                error=f"Error: path '{target}' does not exist",
            )

        results = []
        for filepath in files_to_search:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, start=1):
                        if regex.search(line):
                            results.append(f"{filepath}:{line_no}: {line.rstrip()}")
            except Exception:
                continue

        filenames = sorted({match.split(":", 1)[0] for match in results})
        content = "\n".join(results)

        return ToolResult(
            success=True,
            output={
                "mode": "content",
                "numFiles": len(filenames),
                "filenames": filenames,
                "content": content,
                "numLines": len(results),
                "numMatches": len(results),
            },
        )
