import { contextBridge, ipcRenderer } from "electron";
import type { CoreCommandType, CoreFrame, GuiMetadata, JsonlExportPayload, MarkdownExportPayload, PersistedConfig, SaveJsonlExportResult, SaveMarkdownExportResult, SelectWorkingDirectoryResult } from "../shared/protocol";
import type { LayoutSize } from "../shared/layout";

const api = {
  getMetadata(): Promise<GuiMetadata> {
    return ipcRenderer.invoke("gui:get-metadata");
  },
  saveConfig(config: PersistedConfig): Promise<PersistedConfig> {
    return ipcRenderer.invoke("gui:save-config", config);
  },
  restartCore(config: PersistedConfig): Promise<{ ok: boolean }> {
    return ipcRenderer.invoke("core:restart", config);
  },
  refreshProviderModels(provider: string): Promise<GuiMetadata> {
    return ipcRenderer.invoke("gui:refresh-provider-models", provider);
  },
  sendCommand(type: CoreCommandType, payload: Record<string, unknown>, sessionId?: string | null): Promise<CoreFrame> {
    return ipcRenderer.invoke("core:command", type, payload, sessionId);
  },
  saveMarkdownExport(payload: MarkdownExportPayload): Promise<SaveMarkdownExportResult> {
    return ipcRenderer.invoke("gui:save-markdown-export", payload);
  },
  saveJsonlExport(payload: JsonlExportPayload): Promise<SaveJsonlExportResult> {
    return ipcRenderer.invoke("gui:save-jsonl-export", payload);
  },
  selectWorkingDirectory(): Promise<SelectWorkingDirectoryResult> {
    return ipcRenderer.invoke("gui:select-working-directory");
  },
  setMinimumContentSize(size: Partial<LayoutSize>): Promise<{ ok: boolean }> {
    return ipcRenderer.invoke("gui:set-minimum-content-size", size);
  },
  onCoreEvent(callback: (frame: CoreFrame) => void): () => void {
    const listener = (_event: Electron.IpcRendererEvent, frame: CoreFrame) => callback(frame);
    ipcRenderer.on("core:event", listener);
    return () => ipcRenderer.off("core:event", listener);
  },
  onCoreLog(callback: (message: string) => void): () => void {
    const stderr = (_event: Electron.IpcRendererEvent, chunk: string) => callback(chunk);
    const exit = (_event: Electron.IpcRendererEvent, payload: unknown) => callback(`core exited ${JSON.stringify(payload)}`);
    const spawn = (_event: Electron.IpcRendererEvent, payload: unknown) => callback(`core spawned ${JSON.stringify(payload)}`);
    ipcRenderer.on("core:stderr", stderr);
    ipcRenderer.on("core:exit", exit);
    ipcRenderer.on("core:spawn", spawn);
    return () => {
      ipcRenderer.off("core:stderr", stderr);
      ipcRenderer.off("core:exit", exit);
      ipcRenderer.off("core:spawn", spawn);
    };
  }
};

contextBridge.exposeInMainWorld("hawi", api);

declare global {
  interface Window {
    hawi: typeof api;
  }
}
