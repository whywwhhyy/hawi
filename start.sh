#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
gui_dir="$script_dir/hawi_gui"
launch_cwd="$PWD"
cd "$gui_dir"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to launch Hawi GUI." >&2
  exit 1
fi

if [[ ! -d node_modules ]]; then
  echo "Installing Hawi GUI dependencies..."
  npm install
fi

if [[ ! -f dist/index.html || ! -f dist-electron/main/main.js ]]; then
  echo "Hawi GUI build output is missing. Run '$script_dir/release.sh' once, or run 'npm run build' in $gui_dir." >&2
  exit 1
fi

args=("$@")
export HAWI_GUI_CWD="${HAWI_GUI_CWD:-$launch_cwd}"
if [[ ${#args[@]} -eq 1 && "${args[0]}" != --* ]]; then
  args=(--model "${args[0]}")
fi

if [[ ${#args[@]} -gt 0 ]]; then
  exec npm run start -- "${args[@]}"
else
  exec npm run start
fi
