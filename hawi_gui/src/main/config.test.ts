import path from "node:path";
import { describe, expect, it } from "vitest";
import { buildEngineEnv, buildEngineRunArgs } from "./config";

describe("engine launch helpers", () => {
  it("runs the engine module through uv so dependencies can sync", () => {
    expect(buildEngineRunArgs("C:\\repo\\hawi", ["--inspect"])).toEqual([
      "run",
      "--project",
      "C:\\repo\\hawi",
      "python",
      "-m",
      "hawi_engine",
      "--inspect"
    ]);
  });

  it("passes model refresh args through the engine launcher", () => {
    expect(buildEngineRunArgs("/repo/hawi", ["--inspect", "--refresh-provider", "openai"])).toContain("--refresh-provider");
  });

  it("adds the repo root to PYTHONPATH", () => {
    const env = buildEngineEnv("C:\\repo\\hawi", {
      PATH: "C:\\tools",
      PYTHONPATH: "C:\\existing"
    });

    expect(env.PATH).toBe("C:\\tools");
    expect(env.PYTHONPATH).toBe(`C:\\repo\\hawi${path.delimiter}C:\\existing`);
  });
});
