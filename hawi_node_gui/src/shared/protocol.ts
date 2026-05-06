export const VERSION = "hawi.core.v1";

export type QueueKind = "normal" | "high_prio" | "urgent";

export type CoreCommandType =
  | "hello"
  | "enqueue"
  | "interrupt"
  | "clear_context"
  | "clear_queue"
  | "set_system_prompt"
  | "switch_model"
  | "apply_plugins"
  | "get_status"
  | "shutdown"
  | "ping";

export interface CoreFrame<TPayload = Record<string, unknown>> {
  version: typeof VERSION;
  type: string;
  id?: string | null;
  ts?: number;
  payload: TPayload;
}

export interface CoreCommand<TPayload = Record<string, unknown>> {
  version: typeof VERSION;
  type: CoreCommandType;
  id: string;
  payload: TPayload;
}

export interface PluginCatalogItem {
  key: string;
  label: string;
  schema: JsonSchemaObject;
  defaults: Record<string, unknown>;
}

export interface InspectPayload {
  version: typeof VERSION;
  models: string[];
  plugin_catalog: PluginCatalogItem[];
  default_system_prompt: string;
}

export interface JsonSchemaObject {
  type?: string;
  title?: string;
  description?: string;
  default?: unknown;
  properties?: Record<string, JsonSchemaObject>;
  required?: string[];
  additionalProperties?: boolean;
}

export interface PersistedConfig {
  version: 1;
  modelName: string;
  systemPrompt: string;
  selectedPlugins: string[];
  pluginConfigs: Record<string, Record<string, unknown>>;
  showDebug: boolean;
}

export interface GuiMetadata {
  inspect: InspectPayload;
  config: PersistedConfig;
  coreRunning: boolean;
}

export function makeCommand<TPayload extends Record<string, unknown>>(
  type: CoreCommandType,
  payload: TPayload,
  id = cryptoId()
): CoreCommand<TPayload> {
  return {
    version: VERSION,
    type,
    id,
    payload
  };
}

export function parseNdjsonChunk(
  buffer: string,
  chunk: string
): { frames: CoreFrame[]; buffer: string; errors: string[] } {
  const lines = (buffer + chunk).split(/\r?\n/);
  const nextBuffer = lines.pop() ?? "";
  const frames: CoreFrame[] = [];
  const errors: string[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    try {
      const parsed = JSON.parse(trimmed) as CoreFrame;
      if (parsed.version !== VERSION || typeof parsed.type !== "string") {
        errors.push(`Invalid core frame: ${trimmed}`);
        continue;
      }
      frames.push(parsed);
    } catch (error) {
      errors.push(error instanceof Error ? error.message : String(error));
    }
  }
  return { frames, buffer: nextBuffer, errors };
}

export function cryptoId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
