#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { chmodSync, cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const guiRoot = path.resolve(scriptDir, "..");
const packageJson = JSON.parse(readFileSync(path.join(guiRoot, "package.json"), "utf-8"));
const productName = packageJson.build?.productName ?? "Hawi";
const packageName = packageJson.name ?? "hawi-gui";
const tauriBinaryName = "hawi-gui-tauri";
const defaultShell = "tauri";
const validShells = new Set(["tauri", "electron"]);

if (isMainModule()) {
  main();
}

function main(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);
  if (options.help) {
    printHelp();
    return;
  }

  const prefix = path.resolve(expandHome(options.prefix ?? process.env.HAWI_RELEASE_PREFIX ?? "~/.local"));
  const releaseRoot = path.resolve(expandHome(options.releaseRoot ?? process.env.HAWI_RELEASE_ROOT ?? path.join(prefix, "share", "hawi", "release")));
  const binDir = path.resolve(expandHome(options.binDir ?? process.env.HAWI_RELEASE_BIN_DIR ?? path.join(prefix, "bin")));
  const macAppDir = resolveMacAppDir(options);
  const currentDir = path.join(releaseRoot, "current");

  if (!options.skipBuild) {
    if (options.shell === "tauri") {
      run("npm", ["run", "tauri:build"], guiRoot);
    } else {
      run("npm", ["run", "build:all"], guiRoot);
      run("npx", ["electron-builder", "--dir"], guiRoot);
    }
  }

  const packaged = findPackagedApp(packagedRootForShell(options.shell), options.shell);
  if (!packaged) {
    throw new Error(`Could not find a ${options.shell} build. Run without --skip-build or check ${packagedRootForShell(options.shell)}.`);
  }
  verifyBundledEngine(packaged);
  preparePackagedApp(packaged);

  clearDirectory(currentDir);

  const installedApp = installPackagedApp(packaged, currentDir);
  const launchApp = installMacLaunchApp(installedApp, macAppDir);
  const externalEngineCommand = installExternalEngine(packaged, currentDir);
  const launchEngineCommand = resolvePackagedEngineCommand(launchApp, { allowTauriBuildFallback: false }) ?? externalEngineCommand;
  launchApp.engineCommand = launchEngineCommand;
  verifyEngineCommand(launchEngineCommand);
  verifyInstalledApp(launchApp);
  prewarmEngine(launchEngineCommand, currentDir);
  if (!options.noShim) {
    installShim(launchApp, binDir);
  }

  console.log(`[release-local] Installed ${productName} (${options.shell}) to ${currentDir}`);
  if (launchApp.root !== installedApp.root) {
    console.log(`[release-local] Installed macOS app to ${launchApp.root}`);
  }
  if (!options.noShim) {
    console.log(`[release-local] Installed hawi launcher to ${path.join(binDir, shimName())}`);
    console.log(`[release-local] Make sure ${binDir} is on PATH.`);
  }
}

function isMainModule() {
  return process.argv[1] ? import.meta.url === pathToFileURL(process.argv[1]).href : false;
}

function parseArgs(args) {
  const parsed = {
    prefix: null,
    releaseRoot: null,
    binDir: null,
    shell: process.env.HAWI_RELEASE_SHELL ?? defaultShell,
    appDir: null,
    skipBuild: false,
    noAppInstall: false,
    noShim: false,
    help: false
  };

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--help" || arg === "-h") {
      parsed.help = true;
    } else if (arg === "--skip-build") {
      parsed.skipBuild = true;
    } else if (arg === "--no-app-install") {
      parsed.noAppInstall = true;
    } else if (arg === "--no-shim") {
      parsed.noShim = true;
    } else if (arg === "--shell" || arg === "--runtime" || arg === "--gui") {
      parsed.shell = requireValue(args, ++index, arg);
    } else if (arg.startsWith("--shell=")) {
      parsed.shell = arg.slice("--shell=".length);
    } else if (arg.startsWith("--runtime=")) {
      parsed.shell = arg.slice("--runtime=".length);
    } else if (arg.startsWith("--gui=")) {
      parsed.shell = arg.slice("--gui=".length);
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
    } else if (arg === "--app-dir") {
      parsed.appDir = requireValue(args, ++index, arg);
    } else if (arg.startsWith("--app-dir=")) {
      parsed.appDir = arg.slice("--app-dir=".length);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
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

function printHelp() {
  const command = process.env.HAWI_RELEASE_COMMAND ?? "npm run release:local --";
  console.log(`Usage: ${command} [options]

Builds an unpacked Hawi GUI release, installs it to a local release directory,
and writes a "hawi" launcher into a bin directory.

Options:
  --shell NAME       Desktop shell to build/install: tauri or electron. Default: tauri
  --prefix DIR        Base install prefix. Default: ~/.local
  --release-root DIR  Release root. Default: <prefix>/share/hawi/release
  --bin-dir DIR       Launcher directory. Default: <prefix>/bin
  --app-dir DIR       macOS app install directory. Default: /Applications
  --skip-build        Reuse the existing build output for the selected shell.
  --no-app-install    Do not install Hawi.app into /Applications on macOS.
  --no-shim           Do not install the hawi launcher.
  -h, --help          Show this help.
`);
}

function resolveMacAppDir(parsedOptions) {
  if (process.platform !== "darwin") {
    return null;
  }
  if (parsedOptions.noAppInstall || process.env.HAWI_RELEASE_NO_APP_INSTALL === "1") {
    return null;
  }
  const configured = parsedOptions.appDir ?? process.env.HAWI_RELEASE_APP_DIR ?? "/Applications";
  return path.resolve(expandHome(configured));
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

function packagedRootForShell(shell) {
  if (shell === "tauri") {
    return path.join(guiRoot, "src-tauri", "target", "release", "bundle");
  }
  return path.join(guiRoot, "release");
}

function findPackagedApp(releaseDir, shell) {
  if (!existsSync(releaseDir)) return null;
  if (process.platform === "darwin") {
    return findFirst(releaseDir, (candidate) => candidate.endsWith(`${productName}.app`), shell);
  }
  if (shell === "tauri") {
    return findTauriExecutable();
  }
  if (process.platform === "win32") {
    const unpacked = findFirst(releaseDir, (candidate) => path.basename(candidate) === "win-unpacked");
    if (!unpacked) return null;
    const executable = path.join(unpacked, `${productName}.exe`);
    return existsSync(executable) ? { kind: "directory", shell, path: unpacked, executable } : null;
  }

  const unpacked = findFirst(releaseDir, (candidate) => path.basename(candidate) === "linux-unpacked");
  if (!unpacked) return null;
  const executable = linuxExecutableCandidates(unpacked).find((candidate) => existsSync(candidate));
  return executable ? { kind: "directory", shell, path: unpacked, executable } : null;
}

function findFirst(root, predicate, shell = "electron") {
  const stack = [root];
  while (stack.length > 0) {
    const current = stack.shift();
    if (!current) continue;
    if (predicate(current)) {
      if (process.platform === "darwin" && current.endsWith(".app")) {
        return { kind: "app", shell, path: current, executable: executableForAppBundle(current, shell) };
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

function findTauriExecutable() {
  const executable = process.platform === "win32" ? `${tauriBinaryName}.exe` : tauriBinaryName;
  const candidate = path.join(guiRoot, "src-tauri", "target", "release", executable);
  if (existsSync(candidate)) {
    return { kind: "file", shell: "tauri", path: candidate, executable: candidate };
  }
  if (process.platform === "linux") {
    const appImage = findFirst(path.join(guiRoot, "src-tauri", "target", "release", "bundle"), (candidate) => candidate.endsWith(".AppImage"), "tauri");
    if (typeof appImage === "string") {
      return { kind: "file", shell: "tauri", path: appImage, executable: appImage };
    }
  }
  return null;
}

function executableForAppBundle(appPath, shell) {
  const candidates = [
    path.join(appPath, "Contents", "MacOS", shell === "tauri" ? tauriBinaryName : productName),
    path.join(appPath, "Contents", "MacOS", productName),
    path.join(appPath, "Contents", "MacOS", tauriBinaryName),
  ];
  return candidates.find((candidate) => existsSync(candidate)) ?? candidates[0];
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
      shell: packaged.shell,
      root: destination,
      executable: executableForAppBundle(destination, packaged.shell ?? "electron")
    };
  }
  if (packaged.kind === "file") {
    const destination = path.join(destinationRoot, path.basename(packaged.executable));
    cpSync(packaged.executable, destination);
    if (process.platform !== "win32") {
      chmodSync(destination, 0o755);
    }
    return {
      kind: "file",
      shell: packaged.shell,
      root: destination,
      executable: destination
    };
  }

  cpSync(packaged.path, destinationRoot, { recursive: true });
  return {
    kind: "directory",
    shell: packaged.shell,
    root: destinationRoot,
    executable: path.join(destinationRoot, path.relative(packaged.path, packaged.executable))
  };
}

function installMacLaunchApp(installedApp, targetDir) {
  if (process.platform !== "darwin" || !targetDir || installedApp.kind !== "app") {
    return installedApp;
  }
  const destination = path.join(targetDir, path.basename(installedApp.root));
  if (path.resolve(destination) === path.resolve(installedApp.root)) {
    return installedApp;
  }
  mkdirSync(targetDir, { recursive: true });
  rmSync(destination, {
    recursive: true,
    force: true,
    maxRetries: 5,
    retryDelay: 100,
  });
  ditto(installedApp.root, destination);
  return {
    kind: "app",
    shell: installedApp.shell,
    root: destination,
    executable: executableForAppBundle(destination, installedApp.shell ?? "electron"),
  };
}

function installExternalEngine(packaged, destinationRoot) {
  const sourceCommand = resolvePackagedEngineCommand(packaged, { allowTauriBuildFallback: true });
  if (!sourceCommand) {
    throw new Error(`Bundled hawi-engine executable was not found for ${packaged.path}.`);
  }
  const sourceRoot = path.dirname(sourceCommand);
  const targetRoot = path.join(destinationRoot, "bin", "hawi-engine");
  ditto(sourceRoot, targetRoot);
  return path.join(targetRoot, path.basename(sourceCommand));
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
  const command = resolvePackagedEngineCommand(app, { allowTauriBuildFallback: true });
  if (!command) {
    throw new Error(`Bundled hawi-engine executable was not found for ${app.path}.`);
  }
}

function resolvePackagedEngineCommand(app, options = {}) {
  const resourcesRoot = resourcesRootFor(app);
  const bundled = resourcesRoot ? resolveBundledEngineCommand(resourcesRoot) : null;
  if (bundled) {
    return bundled;
  }
  if (options.allowTauriBuildFallback && app.shell === "tauri") {
    return resolveBundledEngineCommand(path.join(guiRoot, "build"));
  }
  return null;
}

function verifyEngineCommand(command) {
  try {
    if (statSync(command).isFile()) {
      return;
    }
  } catch {
    // Report a clearer error below.
  }
  throw new Error(`Installed hawi-engine executable was not found at ${command}.`);
}

function prewarmEngine(command, cwd) {
  console.log(`[release-local] prewarm ${command} --inspect`);
  const env = { ...process.env };
  delete env.ELECTRON_RUN_AS_NODE;
  const result = spawnSync(command, ["--inspect"], {
    cwd,
    env,
    encoding: "utf-8",
    stdio: ["ignore", "ignore", "inherit"]
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`Installed hawi-engine prewarm failed with exit ${result.status ?? "unknown"}.`);
  }
}

function resourcesRootFor(app) {
  if (app.kind === "app") {
    return path.join(app.root ?? app.path, "Contents", "Resources");
  }
  if (app.kind === "file") {
    return null;
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
    writeFileSync(target, windowsShim(installedApp.executable, installedApp.engineCommand), "utf-8");
  } else {
    writeFileSync(target, posixShim(installedApp.executable, installedApp.engineCommand), "utf-8");
    chmodSync(target, 0o755);
  }
}

function shimName() {
  return process.platform === "win32" ? "hawi.cmd" : "hawi";
}

function posixShim(executable, engineCommand) {
  return `#!/usr/bin/env bash
set -euo pipefail
launcher_executable=${shellQuote(executable)}
launcher_engine=${shellQuote(engineCommand)}
launch_cwd="\${HAWI_GUI_CWD:-$PWD}"
new_instance=0
args=()
for arg in "$@"; do
  case "$arg" in
    --new|--new-instance)
      new_instance=1
      ;;
    *)
      args+=("$arg")
      ;;
  esac
done
export HAWI_GUI_CWD="$launch_cwd"
unset ELECTRON_RUN_AS_NODE
if [[ "$new_instance" == "1" ]]; then
  app_bundle=""
  if [[ "$launcher_executable" == *.app/Contents/MacOS/* ]]; then
    app_bundle="\${launcher_executable%%.app/Contents/MacOS/*}.app"
  fi
  if [[ -n "$app_bundle" && -d "$app_bundle" && -x /usr/bin/open ]]; then
    if (( \${#args[@]} > 0 )); then
      exec /usr/bin/open -n "$app_bundle" --args --cwd "$launch_cwd" "\${args[@]}"
    fi
    exec /usr/bin/open -n "$app_bundle" --args --cwd "$launch_cwd"
  fi
fi
export HAWI_GUI_ENGINE_COMMAND="$launcher_engine"
if (( \${#args[@]} > 0 )); then
  exec "$launcher_executable" "\${args[@]}"
fi
exec "$launcher_executable"
`;
}

function windowsShim(executable, engineCommand) {
  return `@echo off\r
set "HAWI_GUI_CWD=%CD%"\r
set "HAWI_GUI_ENGINE_COMMAND=${engineCommand}"\r
set ELECTRON_RUN_AS_NODE=\r
"${executable}" %*\r
`;
}

function shellQuote(value) {
  return `'${value.replace(/'/g, "'\\''")}'`;
}

export {
  main,
  parseArgs,
  posixShim,
  shellQuote,
  windowsShim,
};
