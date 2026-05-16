#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to package Hawi GUI." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to package Hawi GUI." >&2
  exit 1
fi

echo "Syncing Hawi Python dependencies..."
(
  cd "$repo_root"
  uv sync
)

cd "$script_dir"

if [[ ! -d node_modules ]]; then
  echo "Installing Hawi GUI dependencies..."
  npm install
fi

echo "Packaging Hawi GUI..."
if [[ $# -gt 0 ]]; then
  exec npm run dist -- "$@"
else
  exec npm run dist
fi
