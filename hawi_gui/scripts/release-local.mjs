#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { chmodSync, cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const guiRoot = path.resolve(scriptDir, "..");
const packageJson = JSON.parse(readFileSync(path.join(guiRoot, "package.json"), "utf-8"));
const productName = packageJson.build?.productName ?? "Hawi";
const packageName = packageJson.name ?? "hawi-gui";

const options = parseArgs(process.argv.slice(2));
if (options.help) {
  printHelp();
  process.exit(0);
}

const prefix = path.resolve(expandHome(options.prefix ?? process.env.HAWI_RELEASE_PREFIX ?? "~/.local"));
const releaseRoot = path.resolve(expandHome(options.releaseRoot ?? process.env.HAWI_RELEASE_ROOT ?? path.join(prefix, "share", "hawi", "release")));
const binDir = path.resolve(expandHome(options.binDir ?? process.env.HAWI_RELEASE_BIN_DIR ?? path.join(prefix, "bin")));
const currentDir = path.join(releaseRoot, "current");

if (!options.skipBuild) {
  run("npm", ["run", "build:all"], guiRoot);
  run("npx", ["electron-builder", "--dir"], guiRoot);
}

const packaged = findPackagedApp(path.join(guiRoot, "release"));
if (!packaged) {
  throw new Error("Could not find an unpacked Electron build. Run without --skip-build or check hawi_gui/release.");
}
verifyBundledEngine(packaged);
preparePackagedApp(packaged);

clearDirectory(currentDir);

const installedApp = installPackagedApp(packaged, currentDir);
verifyBundledEngine(installedApp);
verifyInstalledApp(installedApp);
if (!options.noShim) {
  installShim(installedApp, binDir);
}

console.log(`[release-local] Installed ${productName} to ${currentDir}`);
if (!options.noShim) {
  console.log(`[release-local] Installed hawi launcher to ${path.join(binDir, shimName())}`);
  console.log(`[release-local] Make sure ${binDir} is on PATH.`);
}

function parseArgs(args) {
  const parsed = {
    prefix: null,
    releaseRoot: null,
    binDir: null,
    skipBuild: false,
    noShim: false,
    help: false
  };

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--help" || arg === "-h") {
      parsed.help = true;
    } else if (arg === "--skip-build") {
      parsed.skipBuild = true;
    } else if (arg === "--no-shim") {
      parsed.noShim = true;
    } else if (arg === "--prefix") {
      parsed.prefix = requireValue(args, ++index, arg);
    } else if (arg.startsWith("--prefix=")) {
      parsed.prefix = arg.slice("--prefix=".length);
    } else if (arg === "--release-root") {
      parsed.releaseRoot = requireValue(args, ++index, arg);
    } else if (arg.startsWith("--release-root=")) {
      parsed.releaseRoot = arg.slice("--release-root=".length);
    } else if (arg === "--bin-dir") {
      parsed.binDir = requireValue(args, ++index, arg);
    } else if (arg.startsWith("--bin-dir=")) {
      parsed.binDir = arg.slice("--bin-dir=".length);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
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

function printHelp() {
  const command = process.env.HAWI_RELEASE_COMMAND ?? "npm run release:local --";
  console.log(`Usage: ${command} [options]

Builds an unpacked Hawi GUI release, installs it to a local release directory,
and writes a "hawi" launcher into a bin directory.

Options:
  --prefix DIR        Base install prefix. Default: ~/.local
  --release-root DIR  Release root. Default: <prefix>/share/hawi/release
  --bin-dir DIR       Launcher directory. Default: <prefix>/bin
  --skip-build        Reuse hawi_gui/release instead of rebuilding.
  --no-shim           Do not install the hawi launcher.
  -h, --help          Show this help.
`);
}

function run(command, args, cwd) {
  console.log(`[release-local] ${command} ${args.join(" ")}`);
  const env = { ...process.env };
  delete env.ELECTRON_RUN_AS_NODE;
  const result = spawnSync(command, args, {
    cwd,
    env,
    stdio: "inherit",
    shell: process.platform === "win32"
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

function expandHome(value) {
  if (value === "~") return os.homedir();
  if (value.startsWith(`~${path.sep}`) || value.startsWith("~/")) {
    return path.join(os.homedir(), value.slice(2));
  }
  return value;
}

function findPackagedApp(releaseDir) {
  if (!existsSync(releaseDir)) return null;
  if (process.platform === "darwin") {
    return findFirst(releaseDir, (candidate) => candidate.endsWith(`${productName}.app`));
  }
  if (process.platform === "win32") {
    const unpacked = findFirst(releaseDir, (candidate) => path.basename(candidate) === "win-unpacked");
    if (!unpacked) return null;
    const executable = path.join(unpacked, `${productName}.exe`);
    return existsSync(executable) ? { kind: "directory", path: unpacked, executable } : null;
  }

  const unpacked = findFirst(releaseDir, (candidate) => path.basename(candidate) === "linux-unpacked");
  if (!unpacked) return null;
  const executable = linuxExecutableCandidates(unpacked).find((candidate) => existsSync(candidate));
  return executable ? { kind: "directory", path: unpacked, executable } : null;
}

function findFirst(root, predicate) {
  const stack = [root];
  while (stack.length > 0) {
    const current = stack.shift();
    if (!current) continue;
    if (predicate(current)) {
      if (process.platform === "darwin" && current.endsWith(".app")) {
        return { kind: "app", path: current, executable: path.join(current, "Contents", "MacOS", productName) };
      }
      return current;
    }
    if (!existsSync(current) || !statSync(current).isDirectory()) continue;
    for (const entry of readdirSync(current)) {
      stack.push(path.join(current, entry));
    }
  }
  return null;
}

function linuxExecutableCandidates(unpacked) {
  return [
    path.join(unpacked, productName),
    path.join(unpacked, packageName),
    path.join(unpacked, packageName.replace(/-/g, "")),
    path.join(unpacked, packageName.replace(/-/g, "_")),
    path.join(unpacked, packageName.toLowerCase()),
    path.join(unpacked, productName.toLowerCase())
  ];
}

function installPackagedApp(packaged, destinationRoot) {
  if (packaged.kind === "app") {
    const destination = path.join(destinationRoot, path.basename(packaged.path));
    ditto(packaged.path, destination);
    return {
      kind: "app",
      root: destination,
      executable: path.join(destination, "Contents", "MacOS", productName)
    };
  }

  cpSync(packaged.path, destinationRoot, { recursive: true });
  return {
    kind: "directory",
    root: destinationRoot,
    executable: path.join(destinationRoot, path.relative(packaged.path, packaged.executable))
  };
}

function clearDirectory(directory) {
  mkdirSync(directory, { recursive: true });
  for (const entry of readdirSync(directory)) {
    rmSync(path.join(directory, entry), {
      recursive: true,
      force: true,
      maxRetries: 5,
      retryDelay: 100,
    });
  }
}

function preparePackagedApp(packaged) {
  if (process.platform !== "darwin" || packaged.kind !== "app") {
    return;
  }
  run("codesign", ["--force", "--deep", "--sign", "-", packaged.path], guiRoot);
  run("codesign", ["--verify", "--deep", "--strict", "--verbose=2", packaged.path], guiRoot);
}

function verifyInstalledApp(installedApp) {
  if (process.platform !== "darwin" || installedApp.kind !== "app") {
    return;
  }
  run("codesign", ["--verify", "--deep", "--strict", "--verbose=2", installedApp.root], guiRoot);
}

function verifyBundledEngine(app) {
  const command = resolveBundledEngineCommand(resourcesRootFor(app));
  if (!command) {
    throw new Error(`Bundled hawi-engine executable was not found in ${resourcesRootFor(app)}.`);
  }
}

function resourcesRootFor(app) {
  if (app.kind === "app") {
    return path.join(app.root ?? app.path, "Contents", "Resources");
  }
  return path.join(app.root ?? app.path, "resources");
}

function resolveBundledEngineCommand(resourcesPath) {
  const executable = process.platform === "win32" ? "hawi-engine.exe" : "hawi-engine";
  const candidates = [
    path.join(resourcesPath, "bin", executable),
    path.join(resourcesPath, "bin", "hawi-engine", executable),
    path.join(resourcesPath, "app.asar.unpacked", "build", "bin", executable),
    path.join(resourcesPath, "app.asar.unpacked", "build", "bin", "hawi-engine", executable),
  ];
  return candidates.find((candidate) => {
    try {
      return statSync(candidate).isFile();
    } catch {
      return false;
    }
  }) ?? null;
}

function ditto(source, destination) {
  if (process.platform !== "darwin") {
    cpSync(source, destination, { recursive: true });
    return;
  }
  run("/usr/bin/ditto", [source, destination], guiRoot);
}

function installShim(installedApp, targetBinDir) {
  mkdirSync(targetBinDir, { recursive: true });
  const target = path.join(targetBinDir, shimName());
  if (process.platform === "win32") {
    writeFileSync(target, windowsShim(installedApp.executable), "utf-8");
  } else {
    writeFileSync(target, posixShim(installedApp.executable), "utf-8");
    chmodSync(target, 0o755);
  }
}

function shimName() {
  return process.platform === "win32" ? "hawi.cmd" : "hawi";
}

function posixShim(executable) {
  return `#!/usr/bin/env bash
set -euo pipefail
export HAWI_GUI_CWD="\${HAWI_GUI_CWD:-$PWD}"
unset ELECTRON_RUN_AS_NODE
exec ${shellQuote(executable)} "$@"
`;
}

function windowsShim(executable) {
  return `@echo off\r
set HAWI_GUI_CWD=%CD%\r
set ELECTRON_RUN_AS_NODE=\r
"${executable}" %*\r
`;
}

function shellQuote(value) {
  return `'${value.replace(/'/g, "'\\''")}'`;
}
