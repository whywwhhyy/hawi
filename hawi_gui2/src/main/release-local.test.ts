import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";

type ReleaseLocalModule = {
  posixShim(executable: string, engineCommand: string): string;
};

const bashIt = hasBash() ? it : it.skip;

describe("release-local launcher shim", () => {
  bashIt("runs with no forwarded args under nounset", async () => {
    const result = await runGeneratedShim([]);

    expect(result.status).toBe(0);
    expect(result.stderr).toBe("");
    expect(result.stdout).toContain("argc=0\n");
    expect(result.stdout).toContain("cwd=/workspace with space\n");
    expect(result.stdout).toContain("engine=/engine with space/hawi-engine\n");
  });

  bashIt("strips the multi-instance flag while preserving forwarded args", async () => {
    const result = await runGeneratedShim(["--new", "--model", "model with space"]);

    expect(result.status).toBe(0);
    expect(result.stderr).toBe("");
    expect(result.stdout).toContain("argc=2\n");
    expect(result.stdout).toContain("arg1=--model\n");
    expect(result.stdout).toContain("arg2=model with space\n");
    expect(result.stdout).not.toContain("arg1=--new\n");
  });

  it("uses macOS open -n for app bundle new-instance launches", async () => {
    const { posixShim } = await loadReleaseLocal();
    const shim = posixShim(
      "/Applications/Hawi Shadcn.app/Contents/MacOS/hawi-gui2-tauri",
      "/Applications/Hawi Shadcn.app/Contents/Resources/bin/hawi-engine/hawi-engine",
    );

    expect(shim).toContain("exec /usr/bin/open -n");
    expect(shim).toContain("if (( ${#args[@]} > 0 ))");
  });
});

async function runGeneratedShim(args: string[]) {
  const { posixShim } = await loadReleaseLocal();
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "hawi-shim-test-"));
  try {
    const launcher = path.join(temp, "fake launcher");
    fs.writeFileSync(
      launcher,
      `#!/usr/bin/env bash
set -euo pipefail
printf 'cwd=%s\\n' "\${HAWI_GUI_CWD:-}"
printf 'engine=%s\\n' "\${HAWI_GUI_ENGINE_COMMAND:-}"
printf 'argc=%s\\n' "$#"
index=0
for arg in "$@"; do
  index=$((index + 1))
  printf 'arg%s=%s\\n' "$index" "$arg"
done
`,
      "utf-8",
    );
    fs.chmodSync(launcher, 0o755);

    const shim = path.join(temp, "hawi");
    fs.writeFileSync(shim, posixShim(launcher, "/engine with space/hawi-engine"), "utf-8");
    fs.chmodSync(shim, 0o755);

    return spawnSync("bash", [shim, ...args], {
      cwd: temp,
      encoding: "utf-8",
      env: {
        PATH: process.env.PATH ?? "",
        HAWI_GUI_CWD: "/workspace with space",
      },
    });
  } finally {
    fs.rmSync(temp, { recursive: true, force: true });
  }
}

async function loadReleaseLocal(): Promise<ReleaseLocalModule> {
  const guiRootScript = path.resolve(process.cwd(), "scripts/release-local.mjs");
  const repoRootScript = path.resolve(process.cwd(), "hawi_gui/scripts/release-local.mjs");
  const scriptPath = fs.existsSync(guiRootScript) ? guiRootScript : repoRootScript;
  return await import(pathToFileURL(scriptPath).href) as ReleaseLocalModule;
}

function hasBash(): boolean {
  return spawnSync("bash", ["-c", "true"]).status === 0;
}
