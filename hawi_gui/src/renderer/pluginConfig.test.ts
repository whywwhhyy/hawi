import { describe, expect, it } from "vitest";
import { coerceSchemaValue, invertPluginSelection, mergePluginDefaults, selectAllPluginKeys, validatePluginConfig } from "./pluginConfig";
import type { PluginCatalogItem } from "../shared/protocol";

const item: PluginCatalogItem = {
  key: "mcp",
  label: "MCPPlugin",
  defaults: { config_path: ".hawi/mcp/config.json" },
  schema: {
    type: "object",
    required: ["config_path"],
    properties: {
      config_path: { type: "string", title: "Path" },
      enabled: { type: "boolean" },
      count: { type: "integer" },
      ratio: { type: "number" }
    }
  }
};

const catalog: PluginCatalogItem[] = [
  item,
  {
    key: "plan",
    label: "PlanPlugin",
    defaults: {},
    schema: {
      type: "object",
      properties: {}
    }
  },
  {
    key: "workflow",
    label: "WorkflowPlugin",
    defaults: {},
    schema: {
      type: "object",
      properties: {}
    }
  }
];

describe("plugin config helpers", () => {
  it("merges defaults for selected plugins", () => {
    const result = mergePluginDefaults([item], ["mcp"], { mcp: { config_path: "custom.json" } });
    expect(result.pluginConfigs.mcp.config_path).toBe("custom.json");
  });

  it("returns catalog keys for select all in display order", () => {
    expect(selectAllPluginKeys(catalog)).toEqual(["mcp", "plan", "workflow"]);
  });

  it("inverts selection while ignoring unknown selected keys", () => {
    expect(invertPluginSelection(catalog, new Set(["mcp", "missing"]))).toEqual(["plan", "workflow"]);
  });

  it("validates required fields", () => {
    expect(validatePluginConfig(item, { config_path: "" })).toEqual(["MCPPlugin: config_path is required"]);
    expect(validatePluginConfig(item, { config_path: "x.json" })).toEqual([]);
  });

  it("coerces primitive JSON schema values", () => {
    expect(coerceSchemaValue({ type: "boolean" }, 1)).toBe(true);
    expect(coerceSchemaValue({ type: "integer" }, "12")).toBe(12);
    expect(coerceSchemaValue({ type: "number" }, "1.25")).toBe(1.25);
    expect(coerceSchemaValue({ type: "string" }, 3)).toBe("3");
  });
});
