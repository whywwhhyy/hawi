# Hawi GUI

Electron GUI for Hawi core-cli.

## Prerequisites

- Node.js 22+
- npm

## Getting Started

```bash
npm install
npm run build
npm start
```

## Scripts

| Script                                         | Description                                                              |
| ---------------------------------------------- | ------------------------------------------------------------------------ |
| `npm run build`                                | Type check + Vite build                                                  |
| `npm run build:core`                           | Build the bundled `hawi-engine` executable for the current platform      |
| `npm run package`                              | Build an unpacked, double-clickable desktop app for the current platform |
| `npm run dist`                                 | Build the current platform installer/distributable                       |
| `npm run dist:mac` / `dist:win` / `dist:linux` | Build platform-specific Electron distributables                          |
| `./pack.sh` / `./pack.ps1`                     | Sync dependencies and build the current platform distributable           |
| `npm start` / `npm run dev`                    | Build and launch Electron                                                |
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

Or use the wrapper scripts from any working directory:

```bash
./hawi_gui/pack.sh
```

```powershell
.\hawi_gui\pack.ps1
```

Build output is written to `hawi_gui/release/`. Electron Builder creates the
native target for the machine running the command, such as `.dmg` on macOS,
`.exe` on Windows, and `.AppImage` on Linux.

When a packaged GUI is launched from a terminal, the engine workspace is the
terminal's current directory, so project-local data is written to `.hawi/` under
that directory. Double-click launches usually start from a system or install
directory; in those cases the app falls back to the user's home directory so
persistent Hawi data lands under `~/.hawi/` instead of platform-specific
AppData/Application Support folders. An explicit `--cwd`, `HAWI_GUI_CWD`, or
`INIT_CWD` value still takes precedence.

Current packaging is intentionally unsigned for pre-release builds. macOS
code signing and notarization are disabled, and Windows signing is skipped.
