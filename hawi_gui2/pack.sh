#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
shell="${HAWI_GUI_SHELL:-tauri}"
forward_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shell|--runtime|--gui)
      if [[ $# -lt 2 ]]; then
        echo "$1 requires a value" >&2
        exit 1
      fi
      shell="$2"
      shift 2
      ;;
    --shell=*|--runtime=*|--gui=*)
      shell="${1#*=}"
      shift
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

shell="$(printf '%s' "$shell" | tr '[:upper:]' '[:lower:]')"
if [[ "$shell" != "tauri" && "$shell" != "electron" ]]; then
  echo "--shell must be one of: tauri, electron" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to package Hawi GUI." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to package Hawi GUI." >&2
  exit 1
fi

if [[ "$shell" == "tauri" ]] && ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is required to package Hawi GUI with Tauri." >&2
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

echo "Packaging Hawi GUI with $shell..."
if [[ "$shell" == "tauri" ]]; then
  exec npm run tauri:build -- "${forward_args[@]}"
else
  exec npm run dist:electron -- "${forward_args[@]}"
fi
