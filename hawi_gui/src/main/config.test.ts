import path from "node:path";
import { describe, expect, it } from "vitest";
import { buildEngineEnv, buildEngineRunArgs, preserveProviderOrder } from "./config";

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

describe("provider order", () => {
  it("keeps an existing provider in its original position after refresh", () => {
    expect(preserveProviderOrder(
      ["alpha/a1", "target/t1", "omega/o1"],
      ["target/t1", "target/t2", "alpha/a1", "omega/o1"]
    )).toEqual(["alpha/a1", "target/t1", "target/t2", "omega/o1"]);
  });

  it("appends newly discovered providers after known providers", () => {
    expect(preserveProviderOrder(
      ["alpha/a1", "omega/o1"],
      ["new/n1", "omega/o1", "alpha/a1"]
    )).toEqual(["alpha/a1", "omega/o1", "new/n1"]);
  });
});
