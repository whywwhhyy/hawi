import { describe, expect, it } from "vitest";
import { coerceSchemaValue, expandPluginSelection, mergePluginDefaults, resolvePluginSelectionChange, selectAllPluginKeys, validatePluginConfig } from "./pluginConfig";
import type { PluginCatalogItem } from "../shared/protocol";

const item: PluginCatalogItem = {
  key: "hawi/mcp",
  name: "hawi/mcp",
  display_name: "MCP",
  dependencies: [],
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
    key: "hawi/plan",
    name: "hawi/plan",
    display_name: "Plan",
    dependencies: [],
    defaults: {},
    schema: {
      type: "object",
      properties: {}
    }
  },
  {
    key: "hawi/workflow",
    name: "hawi/workflow",
    display_name: "Workflow",
    dependencies: ["hawi/plan"],
    defaults: {},
    schema: {
      type: "object",
      properties: {}
    }
  }
];

describe("plugin config helpers", () => {
  it("merges defaults for selected plugins", () => {
    const result = mergePluginDefaults([item], ["hawi/mcp"], { "hawi/mcp": { config_path: "custom.json" } });
    expect(result.pluginConfigs["hawi/mcp"].config_path).toBe("custom.json");
  });

  it("returns catalog keys for select all in display order", () => {
    expect(selectAllPluginKeys(catalog)).toEqual(["hawi/mcp", "hawi/plan", "hawi/workflow"]);
  });

  it("expands dependencies when selecting a plugin", () => {
    expect(resolvePluginSelectionChange(catalog, ["hawi/mcp"], ["hawi/mcp", "hawi/workflow"])).toEqual([
      "hawi/mcp",
      "hawi/plan",
      "hawi/workflow"
    ]);
  });

  it("removes unsupported plugins when a dependency is deselected", () => {
    expect(resolvePluginSelectionChange(
      catalog,
      ["hawi/mcp", "hawi/plan", "hawi/workflow"],
      ["hawi/mcp", "hawi/workflow"]
    )).toEqual(["hawi/mcp"]);
  });

  it("expands plugin dependencies", () => {
    expect(expandPluginSelection(catalog, ["hawi/workflow"])).toEqual(["hawi/plan", "hawi/workflow"]);
  });

  it("validates required fields", () => {
    expect(validatePluginConfig(item, { config_path: "" })).toEqual(["MCP: config_path is required"]);
    expect(validatePluginConfig(item, { config_path: "x.json" })).toEqual([]);
  });

  it("coerces primitive JSON schema values", () => {
    expect(coerceSchemaValue({ type: "boolean" }, 1)).toBe(true);
    expect(coerceSchemaValue({ type: "integer" }, "12")).toBe(12);
    expect(coerceSchemaValue({ type: "number" }, "1.25")).toBe(1.25);
    expect(coerceSchemaValue({ type: "string" }, 3)).toBe("3");
  });
});
