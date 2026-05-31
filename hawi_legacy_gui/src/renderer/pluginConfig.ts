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
  const selected = new Set(expandPluginSelection(catalog, selectedPlugins));
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

export function resolvePluginSelectionChange(
  catalog: PluginCatalogItem[],
  currentPlugins: Iterable<string>,
  nextPlugins: Iterable<string>
): string[] {
  const current = new Set(currentPlugins);
  const next = new Set(nextPlugins);
  const removed = [...current].some((key) => !next.has(key));
  return removed ? pruneUnsupportedPluginSelection(catalog, next) : expandPluginSelection(catalog, next);
}

export function expandPluginSelection(catalog: PluginCatalogItem[], selectedPlugins: Iterable<string>): string[] {
  const byKey = new Map(catalog.map((item) => [item.key, item]));
  const result = new Set<string>();
  const visiting = new Set<string>();

  function visit(key: string): void {
    const item = byKey.get(key);
    if (!item || result.has(key)) return;
    if (visiting.has(key)) return;
    visiting.add(key);
    for (const dependency of item.dependencies ?? []) {
      visit(dependency);
    }
    visiting.delete(key);
    result.add(key);
  }

  for (const key of selectedPlugins) {
    visit(key);
  }
  return catalog.filter((item) => result.has(item.key)).map((item) => item.key);
}

function pruneUnsupportedPluginSelection(catalog: PluginCatalogItem[], selectedPlugins: Iterable<string>): string[] {
  const known = new Set(catalog.map((item) => item.key));
  const selected = new Set([...selectedPlugins].filter((key) => known.has(key)));
  let changed = true;

  while (changed) {
    changed = false;
    for (const item of catalog) {
      if (!selected.has(item.key)) continue;
      if ((item.dependencies ?? []).some((dependency) => !selected.has(dependency))) {
        selected.delete(item.key);
        changed = true;
      }
    }
  }

  return catalog.filter((item) => selected.has(item.key)).map((item) => item.key);
}

export function validatePluginConfig(item: PluginCatalogItem, config: Record<string, unknown>): string[] {
  const errors: string[] = [];
  const required = item.schema.required ?? [];
  for (const field of required) {
    const value = config[field];
    if (value == null || (typeof value === "string" && !value.trim())) {
      errors.push(`${item.display_name}: ${field} is required`);
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
