import type { CoreCommandType, CoreFrame, GuiMetadata, MarkdownExportPayload, PersistedConfig, SaveMarkdownExportResult, SelectWorkingDirectoryResult } from "../shared/protocol";
import type { LayoutSize } from "../shared/layout";

declare global {
  interface Window {
    hawi: {
      getMetadata(): Promise<GuiMetadata>;
      saveConfig(config: PersistedConfig): Promise<PersistedConfig>;
      restartCore(config: PersistedConfig): Promise<{ ok: boolean }>;
      refreshProviderModels(provider: string): Promise<GuiMetadata>;
      sendCommand(type: CoreCommandType, payload: Record<string, unknown>, sessionId?: string | null): Promise<CoreFrame>;
      saveMarkdownExport(payload: MarkdownExportPayload): Promise<SaveMarkdownExportResult>;
      selectWorkingDirectory(): Promise<SelectWorkingDirectoryResult>;
      setMinimumContentSize(size: Partial<LayoutSize>): Promise<{ ok: boolean }>;
      onCoreEvent(callback: (frame: CoreFrame) => void): () => void;
      onCoreLog(callback: (message: string) => void): () => void;
    };
  }
}

export {};
