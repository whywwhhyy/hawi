#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
gui_dir_name="hawi_gui"
gui_dir="$script_dir/$gui_dir_name"
launch_cwd="$PWD"

if [[ ! -f "$gui_dir/package.json" ]]; then
  echo "Current Hawi GUI directory is missing: $gui_dir" >&2
  exit 1
fi

cd "$gui_dir"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to launch Hawi GUI." >&2
  exit 1
fi

if [[ ! -d node_modules ]]; then
  echo "Installing Hawi GUI dependencies..."
  npm install
fi

args=("$@")
desktop_shell="${HAWI_GUI_SHELL:-tauri}"
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[$i]}" in
    --shell|--runtime|--gui)
      if (( i + 1 >= ${#args[@]} )); then
        echo "${args[$i]} requires a value." >&2
        exit 1
      fi
      desktop_shell="${args[$((i + 1))]}"
      i=$((i + 1))
      ;;
    --shell=*)
      desktop_shell="${args[$i]#--shell=}"
      ;;
    --runtime=*)
      desktop_shell="${args[$i]#--runtime=}"
      ;;
    --gui=*)
      desktop_shell="${args[$i]#--gui=}"
      ;;
  esac
done
desktop_shell="$(printf '%s' "$desktop_shell" | tr '[:upper:]' '[:lower:]')"

if [[ "$desktop_shell" != "tauri" && "$desktop_shell" != "electron" ]]; then
  echo "--shell must be one of: tauri, electron" >&2
  exit 1
fi

if [[ "$desktop_shell" == "tauri" ]]; then
  if ! command -v cargo >/dev/null 2>&1; then
    echo "cargo is required to launch the Tauri Hawi GUI." >&2
    exit 1
  fi
  if [[ ! -f src-tauri/tauri.conf.json ]]; then
    echo "Tauri project files are missing under $gui_dir/src-tauri." >&2
    exit 1
  fi
elif [[ ! -f dist/index.html || ! -f dist-electron/main/main.js ]]; then
  echo "Electron GUI build output is missing. Run '$script_dir/install.sh --shell electron' once, or run 'npm run build' in $gui_dir." >&2
  exit 1
fi

export HAWI_GUI_CWD="${HAWI_GUI_CWD:-$launch_cwd}"
if [[ ${#args[@]} -eq 1 && "${args[0]}" != --* ]]; then
  args=(--model "${args[0]}")
fi

unset ELECTRON_RUN_AS_NODE

if [[ ${#args[@]} -gt 0 ]]; then
  exec npm run start -- "${args[@]}"
else
  exec npm run start
fi
