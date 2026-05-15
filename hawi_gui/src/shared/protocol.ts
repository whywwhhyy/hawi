export const VERSION = "hawi.core.v1";

export type QueueKind = "normal" | "high_prio" | "urgent";

export interface QueueMessageSnapshot {
  id: string;
  queue: QueueKind;
  content_preview: string;
  content?: string;
  created_at?: number;
  metadata?: Record<string, unknown>;
}

export type QueueMessagesSnapshot = Record<QueueKind, QueueMessageSnapshot[]>;

export interface RuntimeControlState {
  paused: boolean;
  pause_reason?: string | null;
  resumable: boolean;
  paused_at?: number | null;
  last_error_message?: string | null;
}

export type DisplayMessageType = "normal" | "steer" | "urgent" | "resume";

export type SessionLoadState = "unloaded" | "loaded" | "running";

export type CoreCommandType =
  | "hello"
  | "enqueue"
  | "interrupt"
  | "stop"
  | "resume"
  | "clear_context"
  | "clear_queue"
  | "queue_task_add"
  | "queue_task_update"
  | "queue_task_remove"
  | "queue_task_reorder"
  | "set_system_prompt"
  | "switch_model"
  | "refresh_models"
  | "apply_plugins"
  | "get_status"
  | "shutdown"
  | "ping"
  | "session_list"
  | "session_new"
  | "session_fork"
  | "session_rewind"
  | "session_load"
  | "session_switch"
  | "session_delete"
  | "session_save_now"
  | "session_history"
  | "session_export_markdown";

export interface SessionMetaPayload {
  session_id: string;
  name: string;
  created_at: string;
  updated_at: string;
  last_checkpoint_event: string | null;
  components_present: string[];
  locked?: boolean;
  lock_owner?: Record<string, unknown> | null;
  load_state?: SessionLoadState;
  loaded_at?: number;
  last_finished_at?: number;
  gui_launch_profile?: SessionLaunchProfile | null;
}

export interface CoreFrame<TPayload = Record<string, unknown>> {
  version: typeof VERSION;
  type: string;
  id?: string | null;
  ts?: number;
  payload: TPayload;
}

export type PluginEventType =
  | "plugin.event"
  | "plugin.message"
  | "plugin.status"
  | "plugin.tool_progress"
  | "plugin.artifact.upsert"
  | "plugin.artifact.delta"
  | "plugin.artifact.remove"
  | "plugin.artifact.clear";

export interface PluginArtifactPayload {
  id?: string;
  artifact_id?: string;
  type?: string;
  artifact_type?: string;
  title?: string;
  content?: string;
  data?: unknown;
  mime_type?: string;
  mimeType?: string;
  language?: string;
  uri?: string;
  path?: string;
  description?: string;
  status?: string;
  metadata?: Record<string, unknown>;
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
  enum?: unknown[];
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

export interface SessionLaunchProfile {
  version: 1;
  modelName: string;
  systemPrompt: string;
  selectedPlugins: string[];
  pluginConfigs: Record<string, Record<string, unknown>>;
  engineArgs?: string[];
}

export interface GuiMetadata {
  inspect: InspectPayload;
  config: PersistedConfig;
  coreRunning: boolean;
  currentSessionId?: string | null;
  runningSessionCount?: number;
  loadedSessionCount?: number;
  maxLoadedSessions?: number;
}

export interface MarkdownExportReference {
  filename: string;
  content: string;
  mime_type?: string;
}

export interface MarkdownExportPayload {
  suggested_filename: string;
  reference_dir_name?: string;
  markdown: string;
  references?: MarkdownExportReference[];
}

export interface SaveMarkdownExportResult {
  canceled: boolean;
  markdownPath?: string;
  referenceDir?: string;
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

export function cryptoId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
