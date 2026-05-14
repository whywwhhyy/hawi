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
    - read_file: 读取文件内容（支持分页、行号格式、缓存去重；支持 mode="structure" 查看文件符号结构）
    - write_file: 覆盖写入文件（要求先读取，支持乐观并发控制）
    - edit_file: 精确字符串替换编辑（要求先读取，支持乐观并发控制）
    - list_dir: 列出目录内容（支持递归、深度、隐藏文件和数量限制）
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

    @classmethod
    def gui_config_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    @classmethod
    def gui_default_config(cls) -> dict:
        return {}

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
        """Best-effort language detection from file name/extension.

        Supports common programming languages, markup formats, config files,
        and domain-specific formats. Returns ``"text"`` when no match is found.
        """
        basename = os.path.basename(file_path).lower()
        extension = os.path.splitext(basename)[1]

        special_names: dict[str, str] = {
            # Build & CI
            "dockerfile": "dockerfile",
            "makefile": "makefile",
            "cmakelists.txt": "cmake",
            "gemfile": "ruby",
            "rakefile": "ruby",
            "podfile": "ruby",
            "cargo.toml": "toml",
            "cargo.lock": "toml",
            "justfile": "makefile",
            # Config (dotfiles)
            ".gitignore": "gitignore",
            ".dockerignore": "gitignore",
            ".env": "env",
            ".env.example": "env",
            ".editorconfig": "ini",
            ".gitattributes": "gitattributes",
            ".gitmodules": "ini",
            ".prettierrc": "json",
            ".eslintrc": "json",
            ".babelrc": "json",
            # Lisp-family
            "deps.edn": "clojure",
            "project.clj": "clojure",
        }
        # Detect Dockerfile variants and similar prefix-less names
        if basename.startswith("dockerfile"):
            special_names[basename] = "dockerfile"

        extension_map: dict[str, str] = {
            # Python & data science
            ".py": "python",
            ".pyi": "python",
            ".pyw": "python",
            ".pyx": "cython",
            ".ipynb": "json",
            ".r": "r",
            ".rdata": "r",
            ".jl": "julia",
            # JavaScript / TypeScript ecosystem
            ".js": "javascript",
            ".jsx": "javascriptreact",
            ".mjs": "javascript",
            ".cjs": "javascript",
            ".ts": "typescript",
            ".tsx": "typescriptreact",
            ".vue": "vue",
            ".svelte": "svelte",
            ".astro": "astro",
            ".d.ts": "typescript",
            # Web template / markup
            ".html": "html",
            ".htm": "html",
            ".css": "css",
            ".scss": "scss",
            ".sass": "sass",
            ".less": "less",
            ".styl": "stylus",
            ".ejs": "ejs",
            ".hbs": "handlebars",
            ".mustache": "mustache",
            ".pug": "pug",
            # JSON & data interchange
            ".json": "json",
            ".jsonc": "json",
            ".json5": "json",
            ".jsonl": "jsonl",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".xml": "xml",
            ".csv": "csv",
            ".tsv": "tsv",
            ".plist": "xml",
            ".proto": "protobuf",
            ".graphql": "graphql",
            ".gql": "graphql",
            # Documentation
            ".md": "markdown",
            ".markdown": "markdown",
            ".mdx": "mdx",
            ".rst": "rst",
            ".adoc": "asciidoc",
            ".asciidoc": "asciidoc",
            ".tex": "latex",
            ".bib": "bibtex",
            ".org": "org",
            # Shell & scripting
            ".sh": "shell",
            ".bash": "shell",
            ".zsh": "shell",
            ".fish": "fish",
            ".ps1": "powershell",
            ".psm1": "powershell",
            ".bat": "batch",
            ".cmd": "batch",
            ".lua": "lua",
            ".pl": "perl",
            ".pm": "perl",
            ".t": "perl",
            # C family
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".hpp": "cpp",
            ".hxx": "cpp",
            ".hh": "cpp",
            ".cppm": "cpp",
            ".ixx": "cpp",
            ".tpp": "cpp",
            ".cbl": "cobol",
            ".cob": "cobol",
            # Objective-C / Swift
            ".m": "objectivec",
            ".mm": "objectivecpp",
            ".swift": "swift",
            # Java & JVM
            ".java": "java",
            ".class": "java",
            ".jar": "java",
            ".kt": "kotlin",
            ".kts": "kotlin",
            ".scala": "scala",
            ".sc": "scala",
            ".clj": "clojure",
            ".cljs": "clojurescript",
            ".cljc": "clojure",
            ".edn": "clojure",
            ".groovy": "groovy",
            ".gvy": "groovy",
            ".gy": "groovy",
            ".gsh": "groovy",
            # .NET
            ".cs": "csharp",
            ".csx": "csharp",
            ".fs": "fsharp",
            ".fsx": "fsharp",
            ".vb": "vbnet",
            # Go
            ".go": "go",
            ".mod": "go",
            ".sum": "go",
            # Rust
            ".rs": "rust",
            ".rlib": "rust",
            # Ruby
            ".rb": "ruby",
            ".erb": "erb",
            ".rake": "ruby",
            ".gemspec": "ruby",
            # PHP
            ".php": "php",
            ".phtml": "php",
            ".ctp": "php",
            # Haskell
            ".hs": "haskell",
            ".lhs": "literatehaskell",
            # Erlang / Elixir
            ".erl": "erlang",
            ".hrl": "erlang",
            ".ex": "elixir",
            ".exs": "elixir",
            ".heex": "heex",
            ".leex": "elixir",
            # Mobile / embedded
            ".dart": "dart",
            ".kt": "kotlin",
            ".swift": "swift",
            # Data / config
            ".sql": "sql",
            ".sqlite": "sql",
            ".prisma": "prisma",
            ".tf": "terraform",
            ".tfvars": "terraform",
            ".hcl": "hcl",
            ".ini": "ini",
            ".cfg": "ini",
            ".conf": "ini",
            ".env": "env",
            ".env.example": "env",
            # Build systems
            ".cmake": "cmake",
            ".bzl": "starlark",
            ".star": "starlark",
            # Zig / Nim / Odin
            ".zig": "zig",
            ".nim": "nim",
            ".nims": "nim",
            # Assembly
            ".asm": "assembly",
            ".s": "assembly",
            ".S": "assembly",
            # Plain text
            ".txt": "text",
            ".text": "text",
            ".log": "text",
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

    def _parse_structure(self, content: str, language: str) -> list[dict]:
        """Parse file content to extract structural elements (functions, classes, sections).

        Returns a list of ``{type, name, start_line, end_line}`` dicts sorted
        by position. Uses lightweight regex heuristics; not a full parser.
        """
        lines = content.splitlines(keepends=False)
        symbols: list[dict] = []

        # Language-agnostic symbol patterns: function/class definitions
        # Ordered roughly by likelihood so earlier matches dominate.
        patterns = [
            # Python
            (r'^\s*(async\s+)?def\s+(\w+)\s*\(', 'function'),
            (r'^\s*class\s+(\w+)\s*[\(:]', 'class'),
            # JavaScript / TypeScript
            (r'^\s*(export\s+)?(async\s+)?function\s+\*?\s*(\w+)\s*\(', 'function'),
            (r'^\s*(export\s+)?class\s+(\w+)', 'class'),
            (r'^\s*(export\s+)?(default\s+)?(async\s+)?\(?\s*(\w+)\s*=\s*(\(|async\s*\()', 'function'),
            # Go
            (r'^\s*func\s+(\w+)\s*\(', 'function'),
            (r'^\s*type\s+(\w+)\s+struct\s*\{', 'struct'),
            (r'^\s*type\s+(\w+)\s+interface\s*\{', 'interface'),
            (r'^\s*func\s+\([^)]+\)\s+(\w+)\s*\(', 'method'),
            # Rust
            (r'^\s*fn\s+(\w+)\s*\(', 'function'),
            (r'^\s*struct\s+(\w+)\s*[<{]', 'struct'),
            (r'^\s*enum\s+(\w+)\s*[<{]', 'enum'),
            (r'^\s*trait\s+(\w+)\s*[<{]', 'trait'),
            (r'^\s*impl\s+(\w+)', 'impl'),
            # C / C++ / Java / C#
            (r'^\s*(public|private|protected|static|virtual|override|abstract)?\s*(public|private|protected|static|virtual|override|abstract)?\s*(class|struct|interface)\s+(\w+)', 'class'),
            (r'^\s*(public|private|protected|static|virtual|override|abstract|inline|const)?\s*(public|private|protected|static|virtual|override|abstract|inline|const)?\s*[\w:*<>]+\s+(\w+)\s*\(', 'function'),
            # Ruby
            (r'^\s*(def|def self\.)\s+(\w+)', 'function'),
            (r'^\s*class\s+(\w+)', 'class'),
            (r'^\s*module\s+(\w+)', 'module'),
            # PHP
            (r'^\s*function\s+(\w+)\s*\(', 'function'),
            (r'^\s*(abstract\s+)?class\s+(\w+)', 'class'),
            # Swift
            (r'^\s*(public|private|internal|open)?\s*(func|class|struct|enum|protocol)\s+(\w+)', 'declaration'),
            # Kotlin
            (r'^\s*(fun|class|data class|object|interface|enum class)\s+(\w+)', 'declaration'),
            # Haskell
            (r'^\s*(\w+)\s*::', 'type_signature'),
            # Elixir
            (r'^\s*def\s+(\w+)', 'function'),
            (r'^\s*defmodule\s+(\w+)', 'module'),
            # Shell
            (r'^\s*(function\s+)?(\w+)\s*\(\)\s*\{', 'function'),
            # Lua
            (r'^\s*function\s+(\w+[\.:]\w+|\w+)', 'function'),
            # Markdown heading as section
            (r'^#{1,6}\s+(.+)', 'section'),
        ]

        for i, line in enumerate(lines):
            line_stripped = line.rstrip()
            for pat, sym_type in patterns:
                m = re.search(pat, line_stripped)
                if m:
                    # Extract the name (last non-None group is the name)
                    groups = m.groups()
                    name = next((g for g in reversed(groups) if g is not None), f"<{sym_type}>")
                    symbols.append({
                        "type": sym_type,
                        "name": name,
                        "start_line": i,  # 0-based
                        "line": i + 1,    # 1-based for display
                    })
                    break

        # Estimate end_line by looking at next symbol or end of file
        for idx, sym in enumerate(symbols):
            if idx + 1 < len(symbols):
                sym["end_line"] = symbols[idx + 1]["start_line"] - 1
            else:
                sym["end_line"] = len(lines) - 1
            sym["line_count"] = sym["end_line"] - sym["start_line"] + 1

        return symbols



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

    def _directory_entry_type(self, entry: os.DirEntry) -> str:
        if entry.is_symlink():
            return "symlink"
        if entry.is_dir(follow_symlinks=False):
            return "directory"
        if entry.is_file(follow_symlinks=False):
            return "file"
        return "other"

    def _directory_entry_info(
        self,
        entry: os.DirEntry,
        *,
        root_path: str,
        depth: int,
    ) -> dict:
        entry_type = self._directory_entry_type(entry)
        try:
            stat_result = entry.stat(follow_symlinks=False)
            size = stat_result.st_size
            mtime = stat_result.st_mtime
        except OSError:
            size = None
            mtime = None

        abs_path = os.path.abspath(entry.path)
        return {
            "name": entry.name,
            "path": abs_path,
            "relativePath": os.path.relpath(abs_path, root_path),
            "type": entry_type,
            "size": size,
            "mtime": mtime,
            "depth": depth,
        }

    @tool(name="read_file")
    def read_file(
        self,
        file_path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        show_line_numbers: bool = True,
        mode: str = "content",
    ) -> ToolResult:
        """
        读取文件内容，支持分页读取和行号格式输出。

        支持两种 mode：
        - ``mode="content"``（默认）：返回文件具体内容，带行号、语言标注、缓存去重
        - ``mode="structure"``：返回文件的符号结构表，包含函数、类、章节等定义的行号

        mode="content" 规则：
        - 优先使用 read_file 读取文件后再编辑或写入，触发乐观并发控制
        - 同一文件、相同参数读取两次，第二次返回 ``file_unchanged``（去重）
        - 大文件建议先 mode="structure" 看清结构，再按 offset/limit 读具体区间

        mode="structure" 规则：
        - 使用正则启发式解析符号（非编译器），覆盖 Python、JS/TS、Go、Rust 等 15+ 语言
        - Markdown 文件的章节标题也会被识别
        - 返回每个符号的类型、名称、起止行号

        Args:
            file_path: 文件路径（绝对路径或相对于当前工作目录）
            offset: 起始行号（0-based），用于大文件分块读取（仅 mode="content"）
            limit: 读取的最大行数（仅 mode="content"）
            show_line_numbers: 是否在每行前显示行号（默认 true，仅 mode="content"）
            mode: 读取模式 - "content" 或 "structure"

        示例:
            read_file("src/main.py")                                   # 读取整个文件
            read_file("src/main.py", mode="structure")                 # 查看文件结构
            read_file("src/main.py", offset=0, limit=50)               # 读取前 50 行

        典型流程:
            read_file("large.py", mode="structure")                    # 1. 看结构
            read_file("large.py", offset=10, limit=30)                 # 2. 读具体函数
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

        # mode="structure" → return symbol table
        if mode == "structure":
            language = self._detect_language(abs_path)
            symbols = self._parse_structure(content, language)
            return ToolResult(
                success=True,
                output={
                    "type": "structure",
                    "file": abs_path,
                    "language": language,
                    "symbols": symbols,
                    "numSymbols": len(symbols),
                    "totalLines": len(content.splitlines()),
                },
            )

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

        # Build header with language annotation and line range
        header_parts: list[str] = []
        if is_partial:
            header_parts.append(f"Lines {start}-{end} of {total}")
        if detected_language != "text":
            header_parts.append(f"language: {detected_language}")
        header = f"[{' | '.join(header_parts)}]\n" if header_parts else ""

        # When line numbers are off and language is known, wrap in fenced code block
        # for syntax highlighting in model-facing text
        if not show_line_numbers and detected_language != "text":
            formatted = f"```{detected_language}\n{formatted}```\n"

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

    @tool(name="write_file")
    def write_file(self, file_path: str, content: str) -> ToolResult:
        """
        覆盖写入文件内容。如果文件已存在，必须先使用 read_file 读取。

        写入后自动更新内部缓存，并生成结构化 patch 和 git diff 用于审计。

        规则：
        - 已存在文件必须先 read_file 再 write_file（乐观并发控制）
        - 写入前文件如果被外部修改，write_file 会拒绝并提示重新读取
        - 新文件路径会自动创建中间目录（类似 ``mkdir -p``）
        - 使用原子写入（先写入临时文件再 rename），避免写入中断导致文件损坏
        - 系统保护路径（/System、/bin、/etc 等）拒绝写入

        注意：
        - 如果要修改文件的部分内容，优先使用 edit_file（更精确安全）
        - write_file 会覆盖整个文件，适合创建新文件或完全重写
        - 需要压缩的工具结果或二进制内容不适合用此工具

        Args:
            file_path: 要写入的文件路径
            content: 完整的文件内容

        示例:
            write_file("new_file.py", "print('hello')\\n")           # 创建新文件
            write_file("output.yaml", yaml.dumps(data))             # 写入序列化数据
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

    @tool(name="edit_file")
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

        编辑后自动更新内部缓存，并生成结构化 patch 和 git diff 用于审计。

        规则：
        - edit_file 前必须先用 read_file 读取文件（写保护）
        - 读取后文件如果被外部修改，edit_file 会拒绝并提示重新读取
        - 默认要求 old_string 唯一匹配，避免误替换；确认语义唯一后再设置 replace_all=True
        - old_string 必须精确匹配，包含缩进和空格

        什么时候用 edit_file vs write_file：
        - 修改文件的部分内容 → edit_file（更安全，只替换精确匹配的片段）
        - 创建新文件或完全重写 → write_file

        Args:
            file_path: 要编辑的文件路径
            old_string: 要替换的精确文本
            new_string: 用于替换的新文本
            replace_all: 如果为 true，替换所有匹配项；否则要求 old_string 必须唯一匹配

        示例:
            edit_file("main.py", old_string="foo", new_string="bar")              # 单处替换
            edit_file("main.py", "foo\\n", "bar\\n", replace_all=True)            # 全部替换
            edit_file("config.py", "DEBUG = True", "DEBUG = False")               # 修改配置项
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

    @tool(name="list_dir")
    def list_dir(
        self,
        path: Optional[str] = None,
        recursive: bool = False,
        max_depth: int = 1,
        include_hidden: bool = False,
        limit: int = 200,
    ) -> ToolResult:
        """
        列出目录内容，返回文件和子目录的结构化元数据。

        Args:
            path: 要列出的目录路径，默认为当前工作目录
            recursive: 是否递归列出子目录
            max_depth: 递归最大深度；1 表示只列直接子项
            include_hidden: 是否包含以 "." 开头的隐藏文件和目录
            limit: 最多返回的条目数量，避免目录结果过大
        """
        root_path = self._resolve_path(path or os.getcwd())
        if not os.path.exists(root_path):
            return ToolResult(success=False, error=f"Directory not found: {root_path}")
        if not os.path.isdir(root_path):
            return ToolResult(success=False, error=f"Path is not a directory: {root_path}")

        effective_depth = max(1, max_depth)
        effective_limit = max(0, limit)
        entries: list[dict] = []
        truncated = False

        def should_include(name: str) -> bool:
            return include_hidden or not name.startswith(".")

        def sorted_children(directory: str) -> list[os.DirEntry]:
            with os.scandir(directory) as iterator:
                children = [
                    entry
                    for entry in iterator
                    if should_include(entry.name)
                ]
            return sorted(
                children,
                key=lambda entry: (
                    0 if entry.is_dir(follow_symlinks=False) else 1,
                    entry.name.lower(),
                    entry.name,
                ),
            )

        def visit(directory: str, depth: int) -> None:
            nonlocal truncated
            if truncated:
                return
            try:
                children = sorted_children(directory)
            except OSError as e:
                entries.append({
                    "name": os.path.basename(directory),
                    "path": directory,
                    "relativePath": os.path.relpath(directory, root_path),
                    "type": "error",
                    "size": None,
                    "mtime": None,
                    "depth": depth,
                    "error": str(e),
                })
                return

            for entry in children:
                if len(entries) >= effective_limit:
                    truncated = True
                    return

                info = self._directory_entry_info(
                    entry,
                    root_path=root_path,
                    depth=depth,
                )
                entries.append(info)

                if (
                    recursive
                    and info["type"] == "directory"
                    and depth < effective_depth
                ):
                    visit(entry.path, depth + 1)

        visit(root_path, 1)

        return ToolResult(
            success=True,
            output={
                "type": "directory",
                "path": root_path,
                "entries": entries,
                "numEntries": len(entries),
                "isTruncated": truncated,
                "recursive": recursive,
                "maxDepth": effective_depth,
                "includeHidden": include_hidden,
                "limit": effective_limit,
            },
        )

    @tool(name="glob")
    def glob(self, pattern: str, directory: Optional[str] = None) -> ToolResult:
        """
        基于 glob 模式查找文件。

        支持递归搜索（使用 ``**``）、单层匹配（``*``）、字符集（``[abc]``）等标准 glob 语法。
        返回匹配文件的排序后绝对路径列表。

        规则：
        - 默认在当前工作目录搜索
        - ``**`` 模式需要 Python >= 3.5，递归匹配所有子目录
        - 结果不包含隐藏文件（可以用 ``.*`` 显式匹配）
        - 返回绝对路径列表

        Args:
            pattern: glob 模式，如 '*.py' 或 'src/**/*.ts'
            directory: 搜索的根目录，默认为当前工作目录

        示例:
            glob("*.py")                          # 当前目录下所有 .py 文件
            glob("**/*.md")                       # 递归匹配所有 .md 文件
            glob("src/**/test_*.py")              # src 下所有 test_ 开头的 .py
            glob("data/*.csv", "project/")        # project/data/ 下所有 .csv

        提示:
            glob 适合按文件名/路径模式搜索。如果需要在文件内容中搜索文本，请使用 grep。
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

    @tool(name="grep")
    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
        file_glob: Optional[str] = None,
    ) -> ToolResult:
        """
        基于正则搜索文件内容。

        返回 ``filepath:lineno: matched_line`` 格式的搜索结果列表。

        规则：
        - 默认递归搜索当前工作目录（自动跳过隐藏目录如 .git、__pycache__）
        - 自动跳过常见二进制文件（图片、压缩包、编译产物等）
        - 支持 ``file_glob``（或别名 ``glob``）过滤文件名，如 ``file_glob="*.py"``
        - 可指定单个文件路径直接搜索
        - 正则无效时返回明确的失败信息

        Args:
            pattern: 正则表达式模式（Python re 语法）
            path: 搜索的文件或目录路径，默认为当前工作目录
            file_glob: 用于过滤文件名的 glob 模式，如 '*.py'
            glob: file_glob 的别名（Claude Code 兼容）

        示例:
            grep("def .*\\\\(\\\\):", "src/")                       # 搜索函数定义
            grep("TODO|FIXME|HACK", file_glob="*.py")               # 搜索 TODO 注释
            grep("raise.*Error", path="src/main.py")                # 搜索单个文件
            grep("from hawi import", "hawi/", glob="*.py")          # 限定 *.py

        提示:
            grep 适合在文件内容中搜索文本。如果只需要按文件名查找，请使用 glob。
            正则中的特殊字符（``.``, ``(``, ``)``, ``+``, ``*`` 等）需要转义。
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
