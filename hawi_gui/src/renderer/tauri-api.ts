import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type {
  CoreCommandType,
  CoreFrame,
  GuiMetadata,
  JsonlExportPayload,
  MarkdownExportPayload,
  PersistedConfig,
  PluginToolPreviewPayload,
  SaveJsonlExportResult,
  SaveMarkdownExportResult,
  SelectWorkingDirectoryResult,
} from "../shared/protocol";
import { VERSION } from "../shared/protocol";
import type { LayoutSize } from "../shared/layout";

declare global {
  interface Window {
    __TAURI_INTERNALS__?: unknown;
  }
}

type ViteImportMeta = ImportMeta & {
  env?: {
    DEV?: boolean;
  };
};

function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && typeof window.__TAURI_INTERNALS__ !== "undefined";
}

function isDevRuntime(): boolean {
  return (import.meta as ViteImportMeta).env?.DEV === true;
}

if (isTauriRuntime() && !window.hawi) {
  window.hawi = {
    getMetadata(): Promise<GuiMetadata> {
      return invoke("get_metadata");
    },
    saveConfig(config: PersistedConfig): Promise<PersistedConfig> {
      return invoke("save_config", { config });
    },
    restartCore(config: PersistedConfig): Promise<{ ok: boolean }> {
      return invoke("restart_core", { config });
    },
    refreshProviderModels(provider: string): Promise<GuiMetadata> {
      return invoke("refresh_provider_models", { provider });
    },
    previewPluginTools(pluginKey: string, pluginConfig: Record<string, unknown>): Promise<PluginToolPreviewPayload> {
      return invoke("preview_plugin_tools", { pluginKey, pluginConfig });
    },
    sendCommand(type: CoreCommandType, payload: Record<string, unknown>, sessionId?: string | null): Promise<CoreFrame> {
      return invoke("send_command", { type, payload, sessionId: sessionId ?? null });
    },
    saveMarkdownExport(payload: MarkdownExportPayload): Promise<SaveMarkdownExportResult> {
      return invoke("save_markdown_export", { payload });
    },
    saveJsonlExport(payload: JsonlExportPayload): Promise<SaveJsonlExportResult> {
      return invoke("save_jsonl_export", { payload });
    },
    selectWorkingDirectory(): Promise<SelectWorkingDirectoryResult> {
      return invoke("select_working_directory");
    },
    setMinimumContentSize(size: Partial<LayoutSize>): Promise<{ ok: boolean }> {
      return invoke("set_minimum_content_size", { size });
    },
    onCoreEvent(callback: (frame: CoreFrame) => void): () => void {
      let active = true;
      const unlisten = listen<CoreFrame>("core:event", (event) => {
        if (active) {
          callback(event.payload);
        }
      });
      return () => {
        active = false;
        void unlisten.then((off) => off());
      };
    },
    onCoreLog(callback: (message: string) => void): () => void {
      let active = true;
      const listeners = [
        listen<string>("core:stderr", (event) => {
          if (active) {
            callback(event.payload);
          }
        }),
        listen<unknown>("core:exit", (event) => {
          if (active) {
            callback(`core exited ${JSON.stringify(event.payload)}`);
          }
        }),
        listen<unknown>("core:spawn", (event) => {
          if (active) {
            callback(`core spawned ${JSON.stringify(event.payload)}`);
          }
        }),
      ];
      return () => {
        active = false;
        for (const listener of listeners) {
          void listener.then((off) => off());
        }
      };
    },
  };
}

if (!isTauriRuntime() && isDevRuntime() && !window.hawi) {
  const metadata: GuiMetadata = {
    inspect: {
      version: VERSION,
      models: ["preview/model"],
      model_provider_configs: {},
      plugin_catalog: [],
      default_system_prompt: "You are Hawi.",
    },
    config: {
      version: 1,
      modelName: "preview/model",
      modelProviderConfigs: {},
      systemPrompt: "You are Hawi.",
      selectedPlugins: [],
      pluginConfigs: {},
      toolCallPurposeEnabled: true,
      profilingEnabled: true,
      showDebug: true,
      focusModeEnabled: true,
    },
    coreRunning: false,
    currentSessionId: null,
    currentWorkspaceRoot: "Browser preview",
    runningSessionCount: 0,
    loadedSessionCount: 0,
    maxLoadedSessions: 5,
  };

  window.hawi = {
    async getMetadata(): Promise<GuiMetadata> {
      return metadata;
    },
    async saveConfig(config: PersistedConfig): Promise<PersistedConfig> {
      metadata.config = config;
      return config;
    },
    async restartCore(config: PersistedConfig): Promise<{ ok: boolean }> {
      metadata.config = config;
      return { ok: false };
    },
    async refreshProviderModels(): Promise<GuiMetadata> {
      return metadata;
    },
    async previewPluginTools(pluginKey: string): Promise<PluginToolPreviewPayload> {
      return {
        version: VERSION,
        plugin_key: pluginKey,
        plugin_name: pluginKey,
        display_name: pluginKey,
        description: "Preview mode plugin placeholder.",
        tools: [],
      };
    },
    async sendCommand(type: CoreCommandType): Promise<CoreFrame> {
      return {
        version: VERSION,
        type: "error",
        payload: {
          command: type,
          message: "Browser preview mode is not connected to hawi-engine. Use npm run dev for the Tauri shell.",
        },
      };
    },
    async saveMarkdownExport(): Promise<SaveMarkdownExportResult> {
      return { canceled: true };
    },
    async saveJsonlExport(): Promise<SaveJsonlExportResult> {
      return { canceled: true };
    },
    async selectWorkingDirectory(): Promise<SelectWorkingDirectoryResult> {
      return { canceled: true };
    },
    async setMinimumContentSize(): Promise<{ ok: boolean }> {
      return { ok: true };
    },
    onCoreEvent(): () => void {
      return () => undefined;
    },
    onCoreLog(callback: (message: string) => void): () => void {
      callback("Browser preview mode: hawi-engine IPC is unavailable.");
      return () => undefined;
    },
  };
}

export {};
