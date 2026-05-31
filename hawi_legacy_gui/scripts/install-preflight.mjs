#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const guiRoot = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(guiRoot, "..");
const srcTauriRoot = path.join(guiRoot, "src-tauri");
const args = process.argv.slice(2);
const helpRequested = args.some((arg) => arg === "-h" || arg === "--help");
const skipBuild = args.some((arg) => arg === "--skip-build");
const noGlobalInstall = process.env.HAWI_INSTALL_NO_GLOBAL === "1";
const skipDevWarmup = process.env.HAWI_INSTALL_SKIP_DEV_WARMUP === "1";
const expectedTauriCliVersion = resolveTauriCliVersion();

if (helpRequested) {
  console.log("[install-preflight] Help requested; skipping checks.");
  process.exit(0);
}

console.log(`[install-preflight] ${os.platform()} ${os.arch()} checks`);

if (skipBuild) {
  console.log("[install-preflight] Build skipped; checking launcher prerequisites only.");
}

if (process.platform === "win32") {
  ensureWindowsCargo();
  ensureCargoTauri();
  checkNodeTauriBinding();
  if (!skipBuild && !skipDevWarmup) {
    warmWindowsDebugBuild();
  }
} else if (!skipBuild) {
  checkNodeTauriBinding({ fatal: false });
}

console.log("[install-preflight] ok");

function resolveTauriCliVersion() {
  const packageJson = JSON.parse(readFileSync(path.join(guiRoot, "package.json"), "utf-8"));
  const declared = packageJson.devDependencies?.["@tauri-apps/cli"] ?? "2";
  return String(declared).replace(/^[^\d]*/, "");
}

function ensureWindowsCargo() {
  if (commandOk("cargo", ["--version"])) {
    return;
  }

  const cargoBin = path.join(os.homedir(), ".cargo", "bin");
  prependPath(cargoBin);
  if (commandOk("cargo", ["--version"])) {
    return;
  }

  if (noGlobalInstall) {
    fail("cargo was not found. Install Rust with rustup, or unset HAWI_INSTALL_NO_GLOBAL to let install.ps1 try winget.");
  }
  if (!commandOk("winget", ["--version"])) {
    fail("cargo was not found, and winget is unavailable. Install Rust from https://rustup.rs/ and run install again.");
  }

  console.log("[install-preflight] cargo not found; installing Rustup with winget...");
  run("winget", [
    "install",
    "--id",
    "Rustlang.Rustup",
    "-e",
    "--source",
    "winget",
    "--accept-package-agreements",
    "--accept-source-agreements"
  ], { cwd: repoRoot });

  prependPath(cargoBin);
  if (!commandOk("cargo", ["--version"])) {
    fail(`Rustup installed, but cargo is still not on PATH. Add ${cargoBin} to PATH and run install again.`);
  }
}

function ensureCargoTauri() {
  const current = capture("cargo", ["tauri", "--version"], { cwd: guiRoot });
  if (current.ok && versionMatches(current.stdout, expectedTauriCliVersion)) {
    console.log(`[install-preflight] cargo-tauri ${expectedTauriCliVersion} available`);
    return;
  }

  if (noGlobalInstall) {
    fail(`cargo-tauri ${expectedTauriCliVersion} is required. Run: cargo install tauri-cli --version ${expectedTauriCliVersion} --locked`);
  }

  if (current.ok) {
    console.log(`[install-preflight] cargo-tauri version differs (${current.stdout.trim()}); installing ${expectedTauriCliVersion}...`);
  } else {
    console.log(`[install-preflight] cargo-tauri missing; installing ${expectedTauriCliVersion}...`);
  }
  run("cargo", ["install", "tauri-cli", "--version", expectedTauriCliVersion, "--locked"], { cwd: guiRoot });

  const installed = capture("cargo", ["tauri", "--version"], { cwd: guiRoot });
  if (!installed.ok || !versionMatches(installed.stdout, expectedTauriCliVersion)) {
    fail(`cargo-tauri ${expectedTauriCliVersion} did not install cleanly.`);
  }
}

function checkNodeTauriBinding(options = {}) {
  const fatal = options.fatal ?? false;
  const tauriScript = path.join(guiRoot, "node_modules", "@tauri-apps", "cli", "tauri.js");
  if (!existsSync(tauriScript)) {
    if (fatal) {
      fail("@tauri-apps/cli is missing. Run npm install and try again.");
    }
    console.log("[install-preflight] npm Tauri CLI is not installed yet; skipping native binding probe.");
    return;
  }

  const result = capture(process.execPath, [tauriScript, "--version"], { cwd: guiRoot });
  if (result.ok) {
    console.log(`[install-preflight] npm Tauri CLI ok (${result.stdout.trim()})`);
    return;
  }

  const message = result.stderr || result.stdout || "unknown error";
  if (process.platform === "win32") {
    console.log("[install-preflight] npm Tauri native binding is not usable on this Windows install; cargo-tauri will be used.");
    if (process.env.HAWI_TAURI_CLI === "node") {
      fail("HAWI_TAURI_CLI=node is set, but the npm Tauri native binding is broken. Unset it or set HAWI_TAURI_CLI=cargo.");
    }
    return;
  }
  if (fatal) {
    fail(`npm Tauri CLI failed:\n${message}`);
  }
  console.warn(`[install-preflight] npm Tauri CLI failed; run-tauri can fall back to cargo if installed.\n${message}`);
}

function warmWindowsDebugBuild() {
  console.log("[install-preflight] warming Windows Tauri debug build...");
  const first = spawnSync("cargo", ["build"], {
    cwd: srcTauriRoot,
    env: process.env,
    stdio: "inherit"
  });
  if (first.status === 0) {
    return;
  }
  if (first.error) {
    throw first.error;
  }

  console.warn("[install-preflight] debug warm-up failed; retrying once with a single cargo job.");
  const retryEnv = { ...process.env, CARGO_BUILD_JOBS: "1" };
  const retry = spawnSync("cargo", ["build", "-j", "1"], {
    cwd: srcTauriRoot,
    env: retryEnv,
    stdio: "inherit"
  });
  if (retry.error) {
    throw retry.error;
  }
  if (retry.status !== 0) {
    process.exit(retry.status ?? 1);
  }
}

function versionMatches(output, expected) {
  return new RegExp(`\\b${escapeRegExp(expected)}\\b`).test(output);
}

function commandOk(command, args) {
  return capture(command, args).ok;
}

function capture(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd ?? repoRoot,
    env: process.env,
    encoding: "utf-8",
    maxBuffer: 16 * 1024 * 1024
  });
  return {
    ok: !result.error && result.status === 0,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
    status: result.status,
    error: result.error
  };
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd ?? repoRoot,
    env: process.env,
    stdio: "inherit"
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

function prependPath(directory) {
  if (!existsSync(directory)) {
    return;
  }
  const delimiter = path.delimiter;
  const parts = String(process.env.PATH || "").split(delimiter).filter(Boolean);
  if (!parts.some((part) => path.resolve(part).toLowerCase() === path.resolve(directory).toLowerCase())) {
    process.env.PATH = `${directory}${delimiter}${process.env.PATH || ""}`;
  }
}

function fail(message) {
  console.error(`[install-preflight] ${message}`);
  process.exit(1);
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
