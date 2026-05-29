# Hawi GUI

Desktop GUI for Hawi core-cli. Tauri is the default desktop shell; Electron is
still available for compatibility builds.

## Prerequisites

- Node.js 22+
- npm

## Getting Started

```bash
npm install
npm run build
npm start
```

`npm start` and `npm run dev` launch Tauri by default. Use
`npm start -- --shell electron` or `npm run dev -- --shell electron` to launch
the Electron shell instead.

## Scripts

| Script                                         | Description                                                              |
| ---------------------------------------------- | ------------------------------------------------------------------------ |
| `npm run build`                                | Type check + Vite build                                                  |
| `npm run build:core`                           | Build the bundled `hawi-engine` executable for the current platform      |
| `npm run package`                              | Build the current platform desktop app with Tauri                        |
| `npm run package:electron`                     | Build an unpacked Electron desktop app                                   |
| `npm run dist` / `npm run dist:tauri`          | Build the current platform Tauri installer/distributable                 |
| `npm run dist:electron`                        | Build the current platform Electron installer/distributable              |
| `npm run dist:mac:electron` / `dist:win:electron` / `dist:linux:electron` | Build platform-specific Electron distributables |
| `./pack.sh` / `./pack.ps1`                     | Sync dependencies and build the current platform distributable           |
| `npm start` / `npm run dev`                    | Launch Tauri by default; pass `--shell electron` for Electron            |
| `npm run start:electron` / `npm run dev:electron` | Launch the Electron shell directly                                    |
| `npm test`                                     | Run unit tests                                                           |
| `npm run test:coverage`                        | Run tests with coverage report                                           |
| `npm run lint`                                 | ESLint check                                                             |
| `npm run lint:fix`                             | ESLint auto-fix                                                          |
| `npm run format`                               | Format code with Prettier                                                |
| `npm run format:check`                         | Check formatting without writing                                         |

## Log Management

The backend writes logs to `.hawi/hawi-engine.log`. These logs can grow large over time.

To truncate the log manually:

```bash
truncate -s 0 .hawi/hawi-engine.log
```

Or add a cron job (macOS/Linux) to rotate logs weekly:

```bash
0 0 * * 0 truncate -s 0 /path/to/hawi_gui/.hawi/hawi-engine.log
```

The log file is excluded from git via `.gitignore`.

## Packaging

The packaged app includes a PyInstaller-built `hawi-engine` binary, so users can
double-click the GUI without installing Python, `uv`, or checking out this repo.

```bash
npm install
uv sync
npm run dist
```

Electron packaging remains available:

```bash
npm run dist:electron
```

Or use the wrapper scripts from any working directory:

```bash
./hawi_gui/pack.sh
./hawi_gui/pack.sh --shell electron
```

```powershell
.\hawi_gui\pack.ps1
.\hawi_gui\pack.ps1 --shell electron
```

Tauri build output is written under `hawi_gui/src-tauri/target/release/bundle/`.
Electron build output is written to `hawi_gui/release/`. Both create native
targets for the machine running the command, such as `.dmg` on macOS, `.exe` on
Windows, and `.AppImage` on Linux.

`install.sh` / `npm run release:local` also install `Hawi.app` into
`/Applications` on macOS and point the `hawi` launcher at that installed app.
Use `--no-app-install` to keep the app only under the local release directory,
or `--app-dir <dir>` to choose a different macOS install directory.

On macOS, Dock and Finder launches activate an already-running app instance.
Use `hawi --new` from a terminal to launch an additional Hawi app instance for
the current working directory.

When a packaged GUI is launched from a terminal, the engine workspace is the
terminal's current directory, so project-local data is written to `.hawi/` under
that directory. Double-click launches usually start from a system or install
directory; in those cases the app falls back to the user's home directory so
persistent Hawi data lands under `~/.hawi/` instead of platform-specific
AppData/Application Support folders. An explicit `--cwd`, `HAWI_GUI_CWD`, or
`INIT_CWD` value still takes precedence.

Current packaging is intentionally unsigned for pre-release builds. macOS
code signing and notarization are disabled, and Windows signing is skipped.
