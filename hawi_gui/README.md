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

| Script | Description |
|--------|-------------|
| `npm run build` | Type check + Vite build |
| `npm start` / `npm run dev` | Build and launch Electron |
| `npm test` | Run unit tests |
| `npm run test:coverage` | Run tests with coverage report |
| `npm run lint` | ESLint check |
| `npm run lint:fix` | ESLint auto-fix |
| `npm run format` | Format code with Prettier |
| `npm run format:check` | Check formatting without writing |

## Log Management

The backend writes logs to `.hawi/hawi-core.log`. These logs can grow large over time.

To truncate the log manually:

```bash
truncate -s 0 .hawi/hawi-core.log
```

Or add a cron job (macOS/Linux) to rotate logs weekly:

```bash
0 0 * * 0 truncate -s 0 /path/to/hawi_gui/.hawi/hawi-core.log
```

The log file is excluded from git via `.gitignore`.
