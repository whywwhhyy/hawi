#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const guiRoot = path.resolve(scriptDir, "..");
const tauriScript = path.join(guiRoot, "node_modules", "@tauri-apps", "cli", "tauri.js");
const args = process.argv.slice(2);

if (args.length === 0 || args.includes("--help-wrapper")) {
  console.log(`Usage: node scripts/run-tauri.mjs <dev|build|...> [args]

Runs the Tauri CLI. Windows prefers the Rust cargo-tauri CLI because npm
optional native bindings are fragile on fresh Windows installs. Other
platforms prefer the local npm CLI to keep the existing macOS flow intact.

Set HAWI_TAURI_CLI=node or HAWI_TAURI_CLI=cargo to force a backend.`);
  process.exit(args.length === 0 ? 1 : 0);
}

const preference = String(process.env.HAWI_TAURI_CLI || "").trim().toLowerCase();
const nodeCandidate = {
  name: "node @tauri-apps/cli",
  command: process.execPath,
  args: [tauriScript, ...args],
  available: () => existsSync(tauriScript)
};
const cargoCandidate = {
  name: "cargo tauri",
  command: "cargo",
  args: ["tauri", ...args],
  available: () => true
};

const candidates = orderedCandidates(preference);
let sawCandidate = false;

for (const candidate of candidates) {
  if (!candidate.available()) {
    continue;
  }
  sawCandidate = true;
  console.log(`[run-tauri] ${candidate.name} ${args.join(" ")}`);
  const result = spawnSync(candidate.command, candidate.args, {
    cwd: guiRoot,
    env: process.env,
    stdio: "inherit"
  });

  if (result.error) {
    if (result.error.code === "ENOENT") {
      console.warn(`[run-tauri] ${candidate.name} was not found; trying the next Tauri CLI.`);
      continue;
    }
    throw result.error;
  }
  if (result.status === 0) {
    process.exit(0);
  }
  if (process.platform === "win32" && candidate === cargoCandidate) {
    process.exit(result.status ?? 1);
  }
  if (candidate === candidates[candidates.length - 1]) {
    process.exit(result.status ?? 1);
  }
  console.warn(`[run-tauri] ${candidate.name} exited with ${result.status}; trying the next Tauri CLI.`);
}

if (!sawCandidate) {
  console.error("[run-tauri] No Tauri CLI was available. Run `npm install` or `cargo install tauri-cli --version 2.11.2 --locked`.");
} else {
  console.error("[run-tauri] All Tauri CLI candidates failed.");
}
process.exit(1);

function orderedCandidates(preferred) {
  if (preferred === "node") {
    return [nodeCandidate, cargoCandidate];
  }
  if (preferred === "cargo") {
    return [cargoCandidate, nodeCandidate];
  }
  if (process.platform === "win32") {
    return [cargoCandidate, nodeCandidate];
  }
  return [nodeCandidate, cargoCandidate];
}
