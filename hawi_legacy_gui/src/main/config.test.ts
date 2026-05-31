import fs from "node:fs";
import os, { homedir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildEngineEnv,
  buildEngineRunArgs,
  isUsablePackagedWorkspaceCwd,
  loadInspectPayloadAsync,
  preserveProviderOrder,
  resolveBundledEngineCommand,
  resolveEngineLauncher,
  resolveWorkspaceRoot,
  type EngineLauncher,
} from "./config";

describe("engine launch helpers", () => {
  it("runs the engine module through uv so dependencies can sync", () => {
    expect(buildEngineRunArgs("C:\\repo\\hawi", ["--inspect"])).toEqual([
      "run",
      "--project",
      "C:\\repo\\hawi",
      "python",
      "-m",
      "hawi.engine",
      "--inspect",
    ]);
  });

  it("passes model refresh args through the engine launcher", () => {
    expect(buildEngineRunArgs("/repo/hawi", ["--inspect", "--refresh-provider", "openai"])).toContain("--refresh-provider");
  });

  it("loads inspect metadata asynchronously", async () => {
    const workspace = fs.mkdtempSync(path.join(os.tmpdir(), "hawi-inspect-async-"));
    const launcher: EngineLauncher = {
      command: process.execPath,
      argsPrefix: [
        "-e",
        "process.stdout.write(JSON.stringify({version:'hawi.core.v1',models:['openai/gpt'],plugin_catalog:[],default_system_prompt:'system'}))",
        "--",
      ],
      source: "bundled",
    };

    try {
      await expect(loadInspectPayloadAsync("/repo/hawi", workspace, launcher, ["--refresh-provider", "openai"])).resolves.toMatchObject({
        models: ["openai/gpt"],
        default_system_prompt: "system",
      });
    } finally {
      fs.rmSync(workspace, { recursive: true, force: true });
    }
  });

  it("surfaces async inspect command failures with stderr", async () => {
    const workspace = fs.mkdtempSync(path.join(os.tmpdir(), "hawi-inspect-fail-"));
    const launcher: EngineLauncher = {
      command: process.execPath,
      argsPrefix: ["-e", "console.error('provider failed'); process.exit(3)", "--"],
      source: "bundled",
    };

    try {
      await expect(loadInspectPayloadAsync("/repo/hawi", workspace, launcher, [])).rejects.toThrow(/stderr: provider failed/);
    } finally {
      fs.rmSync(workspace, { recursive: true, force: true });
    }
  });

  it("runs a bundled engine executable directly", () => {
    const launcher: EngineLauncher = { command: "/app/resources/bin/hawi-engine", argsPrefix: [], source: "bundled" };

    expect(buildEngineRunArgs("/repo/hawi", ["--inspect"], launcher)).toEqual(["--inspect"]);
  });

  it("resolves a bundled one-dir engine executable", () => {
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), "hawi-engine-dir-"));
    try {
      const executable = process.platform === "win32" ? "hawi-engine.exe" : "hawi-engine";
      const enginePath = path.join(temp, "bin", "hawi-engine", executable);
      fs.mkdirSync(path.dirname(enginePath), { recursive: true });
      fs.writeFileSync(enginePath, "");

      expect(resolveBundledEngineCommand(temp)).toBe(enginePath);
    } finally {
      fs.rmSync(temp, { recursive: true, force: true });
    }
  });

  it("uses the source engine in dev even when stale bundled output exists", () => {
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), "hawi-engine-dir-"));
    try {
      const executable = process.platform === "win32" ? "hawi-engine.exe" : "hawi-engine";
      const enginePath = path.join(temp, "bin", "hawi-engine", executable);
      fs.mkdirSync(path.dirname(enginePath), { recursive: true });
      fs.writeFileSync(enginePath, "");

      expect(resolveEngineLauncher("/repo/hawi", { isPackaged: false, resourcesPath: temp }, "uv")).toEqual({
        command: "uv",
        argsPrefix: ["run", "--project", "/repo/hawi", "python", "-m", "hawi.engine"],
        source: "uv",
      });
      expect(resolveEngineLauncher("/repo/hawi", { isPackaged: true, resourcesPath: temp }, "uv")).toEqual({
        command: enginePath,
        argsPrefix: [],
        source: "bundled",
      });
    } finally {
      fs.rmSync(temp, { recursive: true, force: true });
    }
  });

  it("adds the repo root to PYTHONPATH", () => {
    const env = buildEngineEnv("C:\\repo\\hawi", {
      PATH: "C:\\tools",
      PYTHONPATH: "C:\\existing",
    });

    expect(env.PATH).toBe("C:\\tools");
    expect(env.PYTHONPATH).toBe(`C:\\repo\\hawi${path.delimiter}C:\\existing`);
  });

  it("uses the current directory for packaged command-line launches", () => {
    const originalArgv = process.argv;
    const originalCwd = process.env.HAWI_GUI_CWD;
    const originalInitCwd = process.env.INIT_CWD;
    const workspace = path.join(homedir(), "hawi-cli-workspace");
    process.argv = ["electron", "."];
    delete process.env.HAWI_GUI_CWD;
    delete process.env.INIT_CWD;
    try {
      expect(resolveWorkspaceRoot({ isPackaged: true, cwd: workspace })).toBe(path.resolve(workspace));
    } finally {
      process.argv = originalArgv;
      if (originalCwd === undefined) {
        delete process.env.HAWI_GUI_CWD;
      } else {
        process.env.HAWI_GUI_CWD = originalCwd;
      }
      if (originalInitCwd === undefined) {
        delete process.env.INIT_CWD;
      } else {
        process.env.INIT_CWD = originalInitCwd;
      }
    }
  });

  it("falls back to the home directory for packaged launches from unusable system locations", () => {
    const originalArgv = process.argv;
    const originalCwd = process.env.HAWI_GUI_CWD;
    const originalInitCwd = process.env.INIT_CWD;
    process.argv = ["electron", "."];
    delete process.env.HAWI_GUI_CWD;
    delete process.env.INIT_CWD;
    try {
      expect(resolveWorkspaceRoot({ isPackaged: true, cwd: path.parse(homedir()).root })).toBe(path.resolve(homedir()));
    } finally {
      process.argv = originalArgv;
      if (originalCwd === undefined) {
        delete process.env.HAWI_GUI_CWD;
      } else {
        process.env.HAWI_GUI_CWD = originalCwd;
      }
      if (originalInitCwd === undefined) {
        delete process.env.INIT_CWD;
      } else {
        process.env.INIT_CWD = originalInitCwd;
      }
    }
  });

  it("does not use the packaged app install directory as a workspace", () => {
    const resourcesPath = path.join(path.sep, "Applications", "Hawi.app", "Contents", "Resources");

    expect(isUsablePackagedWorkspaceCwd(path.join(path.sep, "Applications", "Hawi.app", "Contents", "MacOS"), resourcesPath)).toBe(false);
    expect(isUsablePackagedWorkspaceCwd(path.join(homedir(), "project"), resourcesPath)).toBe(true);
  });
});

describe("provider order", () => {
  it("keeps an existing provider in its original position after refresh", () => {
    expect(preserveProviderOrder(["alpha/a1", "target/t1", "omega/o1"], ["target/t1", "target/t2", "alpha/a1", "omega/o1"])).toEqual([
      "alpha/a1",
      "target/t1",
      "target/t2",
      "omega/o1",
    ]);
  });

  it("appends newly discovered providers after known providers", () => {
    expect(preserveProviderOrder(["alpha/a1", "omega/o1"], ["new/n1", "omega/o1", "alpha/a1"])).toEqual(["alpha/a1", "omega/o1", "new/n1"]);
  });
});
