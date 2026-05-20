"""Workspace path helpers."""

from __future__ import annotations

from pathlib import Path

GIT_DIR_NAME = ".git"


def find_git_root(start: str | Path | None = None) -> Path:
    """Return the nearest ancestor containing ``.git``, or ``start``.

    ``.git`` may be either a directory or a file, which covers regular
    repositories, submodules, and worktrees.
    """

    if start is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start).expanduser()

    start_path = start_path.resolve()
    if start_path.is_file():
        start_path = start_path.parent

    for directory in (start_path, *start_path.parents):
        if (directory / GIT_DIR_NAME).exists():
            return directory
    return start_path
