import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type {
  CoreCommandType,
  CoreFrame,
  GuiMetadata,
  MarkdownExportPayload,
  PersistedConfig,
  SaveMarkdownExportResult,
} from "../shared/protocol";

declare global {
  interface Window {
    __TAURI_INTERNALS__?: unknown;
  }
}

function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && typeof window.__TAURI_INTERNALS__ !== "undefined";
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
    sendCommand(type: CoreCommandType, payload: Record<string, unknown>, sessionId?: string | null): Promise<CoreFrame> {
      return invoke("send_command", { type, payload, sessionId: sessionId ?? null });
    },
    saveMarkdownExport(payload: MarkdownExportPayload): Promise<SaveMarkdownExportResult> {
      return invoke("save_markdown_export", { payload });
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

export {};
