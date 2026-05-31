#!/usr/bin/env node
import { spawnSync } from "node:child_process";

const defaultShell = "tauri";
const validShells = new Set(["tauri", "electron"]);

const options = parseArgs(process.argv.slice(2));
if (options.help) {
  printHelp();
  process.exit(0);
}

const lifecycle = process.env.npm_lifecycle_event ?? "start";
const targetScript = selectTargetScript(options.shell, lifecycle);
const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";
const args = ["run", targetScript];
if (options.forwardArgs.length > 0) {
  args.push("--", ...targetRuntimeArgs(options.shell, options.forwardArgs));
}

console.log(`[start-desktop] ${npmCommand} ${args.join(" ")}`);
const result = spawnSync(npmCommand, args, {
  stdio: "inherit",
  shell: process.platform === "win32",
});
if (result.error) {
  throw result.error;
}
process.exit(result.status ?? 1);

function parseArgs(args) {
  const parsed = {
    shell: process.env.HAWI_GUI_SHELL ?? defaultShell,
    help: false,
    forwardArgs: [],
  };

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--help" || arg === "-h") {
      parsed.help = true;
    } else if (arg === "--shell" || arg === "--runtime" || arg === "--gui") {
      parsed.shell = requireValue(args, ++index, arg);
    } else if (arg.startsWith("--shell=")) {
      parsed.shell = arg.slice("--shell=".length);
    } else if (arg.startsWith("--runtime=")) {
      parsed.shell = arg.slice("--runtime=".length);
    } else if (arg.startsWith("--gui=")) {
      parsed.shell = arg.slice("--gui=".length);
    } else {
      parsed.forwardArgs.push(arg);
    }
  }

  parsed.shell = String(parsed.shell || defaultShell).trim().toLowerCase();
  if (!validShells.has(parsed.shell)) {
    throw new Error(`--shell must be one of: ${[...validShells].join(", ")}`);
  }
  return parsed;
}

function requireValue(args, index, name) {
  const value = args[index];
  if (!value || value.startsWith("--")) {
    throw new Error(`${name} requires a value`);
  }
  return value;
}

function selectTargetScript(shell, lifecycle) {
  if (shell === "tauri") {
    return lifecycle === "dev" ? "dev:tauri" : "start:tauri";
  }
  return lifecycle === "dev" ? "dev:electron" : "start:electron";
}

function targetRuntimeArgs(shell, args) {
  if (shell === "tauri") {
    return ["--", "--", ...args];
  }
  return args;
}

function printHelp() {
  console.log(`Usage: npm start -- [options] [runtime args]

Options:
  --shell NAME       Desktop shell to launch: tauri or electron. Default: tauri
  -h, --help         Show this help.

Examples:
  npm start
  npm start -- --shell electron
  npm run dev -- --shell electron
`);
}
