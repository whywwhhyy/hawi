import type { CoreCommandType, CoreFrame, GuiMetadata, JsonlExportPayload, MarkdownExportPayload, PersistedConfig, PluginToolPreviewPayload, SaveJsonlExportResult, SaveMarkdownExportResult, SelectWorkingDirectoryResult } from "../shared/protocol";
import type { LayoutSize } from "../shared/layout";

declare global {
  interface Window {
    hawi: {
      getMetadata(): Promise<GuiMetadata>;
      saveConfig(config: PersistedConfig): Promise<PersistedConfig>;
      restartCore(config: PersistedConfig): Promise<{ ok: boolean }>;
      refreshProviderModels(provider: string): Promise<GuiMetadata>;
      previewPluginTools(pluginKey: string, pluginConfig: Record<string, unknown>): Promise<PluginToolPreviewPayload>;
      sendCommand(type: CoreCommandType, payload: Record<string, unknown>, sessionId?: string | null): Promise<CoreFrame>;
      saveMarkdownExport(payload: MarkdownExportPayload): Promise<SaveMarkdownExportResult>;
      saveJsonlExport(payload: JsonlExportPayload): Promise<SaveJsonlExportResult>;
      selectWorkingDirectory(): Promise<SelectWorkingDirectoryResult>;
      setMinimumContentSize(size: Partial<LayoutSize>): Promise<{ ok: boolean }>;
      onCoreEvent(callback: (frame: CoreFrame) => void): () => void;
      onCoreLog(callback: (message: string) => void): () => void;
    };
  }
}

export {};
