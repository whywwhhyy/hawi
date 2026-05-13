import type { JsonSchemaObject, PluginCatalogItem } from "../shared/protocol";

export interface PluginFormState {
  selectedPlugins: string[];
  pluginConfigs: Record<string, Record<string, unknown>>;
}

export function mergePluginDefaults(
  catalog: PluginCatalogItem[],
  selectedPlugins: string[],
  pluginConfigs: Record<string, Record<string, unknown>>
): PluginFormState {
  const selected = new Set(selectedPlugins);
  const configs: Record<string, Record<string, unknown>> = {};
  for (const item of catalog) {
    if (!selected.has(item.key)) continue;
    configs[item.key] = {
      ...item.defaults,
      ...(pluginConfigs[item.key] ?? {})
    };
  }
  return {
    selectedPlugins: catalog.filter((item) => selected.has(item.key)).map((item) => item.key),
    pluginConfigs: configs
  };
}

export function selectAllPluginKeys(catalog: PluginCatalogItem[]): string[] {
  return catalog.map((item) => item.key);
}

export function invertPluginSelection(catalog: PluginCatalogItem[], selectedPlugins: Iterable<string>): string[] {
  const selected = new Set(selectedPlugins);
  return catalog.filter((item) => !selected.has(item.key)).map((item) => item.key);
}

export function validatePluginConfig(item: PluginCatalogItem, config: Record<string, unknown>): string[] {
  const errors: string[] = [];
  const required = item.schema.required ?? [];
  for (const field of required) {
    const value = config[field];
    if (value == null || (typeof value === "string" && !value.trim())) {
      errors.push(`${item.label}: ${field} is required`);
    }
  }
  return errors;
}

export function coerceSchemaValue(schema: JsonSchemaObject, raw: unknown): unknown {
  if (schema.type === "boolean") return Boolean(raw);
  if (schema.type === "integer") return Number.parseInt(String(raw || "0"), 10);
  if (schema.type === "number") return Number.parseFloat(String(raw || "0"));
  return raw == null ? "" : String(raw);
}
