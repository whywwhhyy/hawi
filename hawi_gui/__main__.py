"""Launcher for the Node/Electron Hawi GUI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hawi_gui",
        description="Hawi GUI (Electron, powered by hawi-core)",
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        default="",
        help="Model factory name (optional; shows selection dialog if omitted)",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Do not run npm install when node_modules is missing",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use the Node GUI dev script",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    node_gui_dir = repo_root / "hawi_node_gui"
    if not node_gui_dir.exists():
        print(f"Node GUI project not found: {node_gui_dir}", file=sys.stderr)
        sys.exit(1)

    npm = shutil.which("npm")
    if npm is None:
        print("npm is required to launch the Hawi Electron GUI.", file=sys.stderr)
        sys.exit(1)

    node_modules = node_gui_dir / "node_modules"
    if not node_modules.exists() and not args.no_install:
        install = subprocess.run([npm, "install"], cwd=node_gui_dir)
        if install.returncode != 0:
            sys.exit(install.returncode)
    elif not node_modules.exists():
        print(
            "node_modules is missing. Run `npm --prefix hawi_node_gui install` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    script = "dev" if args.dev else "start"
    cmd = [npm, "run", script, "--"]
    if args.model_name:
        cmd.extend(["--model", args.model_name])
    result = subprocess.run(cmd, cwd=node_gui_dir)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
