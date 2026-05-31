#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const guiRoot = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(guiRoot, "..");
const outputDir = path.join(guiRoot, "build", "bin");
const workDir = path.join(guiRoot, "build", "pyinstaller", "work");
const specDir = path.join(guiRoot, "build", "pyinstaller", "spec");
const executableName = process.platform === "win32" ? "hawi-engine.exe" : "hawi-engine";
const executablePath = path.join(outputDir, "hawi-engine", executableName);
const dataSeparator = process.platform === "win32" ? ";" : ":";
const uvCommand = process.env.HAWI_GUI_UV_COMMAND || "uv";

rmSync(outputDir, { recursive: true, force: true });
mkdirSync(outputDir, { recursive: true });
mkdirSync(workDir, { recursive: true });
mkdirSync(specDir, { recursive: true });

const pyinstallerArgs = [
  "run",
  "--project",
  repoRoot,
  "pyinstaller",
  "--clean",
  "--noconfirm",
  "--onedir",
  "--name",
  "hawi-engine",
  "--distpath",
  outputDir,
  "--workpath",
  workDir,
  "--specpath",
  specDir,
  "--paths",
  repoRoot,
  "--add-data",
  `${path.join(repoRoot, "hawi", "engine", "templates")}${dataSeparator}${path.join("hawi", "engine", "templates")}`,
  "--collect-submodules",
  "hawi",
  "--collect-submodules",
  "hawi.engine",
  "--collect-submodules",
  "hawi.builtin_plugins",
  path.join(repoRoot, "packaging", "pyinstaller", "hawi_engine_entry.py")
];

const env = {
  ...process.env,
  PYTHONPATH: process.env.PYTHONPATH
    ? `${repoRoot}${path.delimiter}${process.env.PYTHONPATH}`
    : repoRoot
};

console.log(`[build-core] Building ${os.platform()} ${os.arch()} hawi-engine directory...`);
const result = spawnSync(uvCommand, pyinstallerArgs, {
  cwd: repoRoot,
  env,
  stdio: "inherit"
});

if (result.error) {
  throw result.error;
}
if (result.status !== 0) {
  process.exit(result.status ?? 1);
}
if (!existsSync(executablePath)) {
  throw new Error(`PyInstaller completed but did not create ${executablePath}`);
}
if (process.platform !== "win32") {
  chmodSync(executablePath, 0o755);
}

console.log(`[build-core] Created ${path.relative(guiRoot, path.dirname(executablePath))}`);
