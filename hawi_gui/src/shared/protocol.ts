export const VERSION = "hawi.core.v1";

export type QueueKind = "normal" | "high_prio" | "urgent";

export type ContentPartType =
  | "text"
  | "image"
  | "document"
  | "audio"
  | "video"
  | "file"
  | "steer"
  | "tool_call"
  | "tool_result"
  | "reasoning"
  | "cache_point"
  | "cache_control"
  | "refusal"
  | "guard_content"
  | "citation";

export type BlobDirection = "inbound" | "outbound";
export type MediaSourceKind = "url" | "data_uri" | "blob" | "file_id" | "path";

export interface MediaSource {
  kind?: MediaSourceKind;
  uri?: string;
  url?: string;
  data_uri?: string;
  data?: string;
  blob_id?: string;
  file_id?: string;
  path?: string;
  mime?: string;
  mime_type?: string;
  mimeType?: string;
  filename?: string;
  size?: number;
  sha256?: string;
  direction?: BlobDirection;
  format?: string;
  metadata?: Record<string, unknown>;
}

export interface BlobSource extends MediaSource {
  kind: "blob";
  blob_id: string;
  uri: string;
}

export interface ContentPart {
  type: ContentPartType | string;
  text?: string;
  reasoning?: string | null;
  source?: MediaSource;
  title?: string | null;
  context?: string | null;
  id?: string;
  name?: string;
  arguments?: unknown;
  tool_call_id?: string;
  content?: string | ContentPart[];
  is_error?: boolean | null;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface QueueMessageSnapshot {
  id: string;
  queue: QueueKind;
  content_preview: string;
  content?: string | ContentPart[];
  content_parts?: ContentPart[];
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
  | "compact_context"
  | "set_auto_compact"
  | "clear_queue"
  | "queue_task_add"
  | "queue_task_update"
  | "queue_task_remove"
  | "queue_task_reorder"
  | "queue_message_remove"
  | "queue_message_promote"
  | "blob.upload_init"
  | "blob.upload_chunk"
  | "blob.upload_finalize"
  | "blob.info"
  | "blob.has"
  | "blob.fetch"
  | "blob.release"
  | "blob.request_retransmit"
  | "set_system_prompt"
  | "set_profiling"
  | "switch_model"
  | "refresh_models"
  | "save_model_provider_config"
  | "change_cwd"
  | "apply_plugins"
  | "plugin_action"
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
  | "session_rename"
  | "session_save_now"
  | "session_message_edit"
  | "session_history"
  | "session_search"
  | "session_export_markdown"
  | "side_thread_list"
  | "side_thread_start"
  | "side_thread_message"
  | "side_thread_delete";

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
  last_cwd?: string | null;
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

export type SubAgentEventType =
  | "subagent.created"
  | "subagent.event"
  | "subagent.closed";

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
  name: string;
  display_name: string;
  description: string;
  dependencies: string[];
  schema: JsonSchemaObject;
  defaults: Record<string, unknown>;
  permissions?: PluginPermissionDeclared[];
}

export interface PluginPermissionDeclared {
  id: string;
  description: string;
  risk_level: string;
  default_policy: string;
  tool_names: string[];
}

export interface PluginToolPreviewItem {
  name: string;
  short_name: string;
  description: string;
  schema: JsonSchemaObject;
  audit?: boolean;
  context?: string | null;
  timeout?: number | null;
  tags?: string[];
  supports_sync?: boolean;
  supports_async?: boolean;
}

export interface PluginToolPreviewPayload {
  version: typeof VERSION;
  plugin_key: string;
  plugin_name: string;
  display_name: string;
  description: string;
  tools: PluginToolPreviewItem[];
}

export interface InspectPayload {
  version: typeof VERSION;
  models: string[];
  model_provider_configs?: Record<string, ModelProviderConfigPreview>;
  plugin_catalog: PluginCatalogItem[];
  default_system_prompt: string;
}

export interface ModelProviderConfigPreview {
  adapter: string;
  model_count: number;
  properties: Record<string, unknown>;
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
  modelProviderConfigs: Record<string, Record<string, unknown>>;
  systemPrompt: string;
  selectedPlugins: string[];
  pluginConfigs: Record<string, Record<string, unknown>>;
  toolCallPurposeEnabled: boolean;
  profilingEnabled: boolean;
  showDebug: boolean;
  focusModeEnabled: boolean;
}

export interface SessionLaunchProfile {
  version: 1;
  modelName: string;
  modelProviderConfigs: Record<string, Record<string, unknown>>;
  systemPrompt: string;
  selectedPlugins: string[];
  pluginConfigs: Record<string, Record<string, unknown>>;
  toolCallPurposeEnabled: boolean;
  profilingEnabled: boolean;
  engineArgs?: string[];
}

export interface GuiMetadata {
  inspect: InspectPayload;
  config: PersistedConfig;
  coreRunning: boolean;
  currentSessionId?: string | null;
  currentWorkspaceRoot?: string | null;
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

export interface JsonlExportPayload {
  suggested_filename: string;
  records: unknown[];
}

export interface SaveJsonlExportResult {
  canceled: boolean;
  jsonlPath?: string;
}

export interface SelectWorkingDirectoryResult {
  canceled: boolean;
  path?: string;
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
