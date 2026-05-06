#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to launch Hawi GUI." >&2
  exit 1
fi

if [[ ! -d node_modules ]]; then
  echo "Installing Hawi GUI dependencies..."
  npm install
fi

args=("$@")
if [[ ${#args[@]} -eq 1 && "${args[0]}" != --* ]]; then
  args=(--model "${args[0]}")
fi

if [[ ${#args[@]} -gt 0 ]]; then
  exec npm run start -- "${args[@]}"
else
  exec npm run start
fi
