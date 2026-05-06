import type { CoreCommandType, CoreFrame, GuiMetadata, PersistedConfig } from "../shared/protocol";

declare global {
  interface Window {
    hawi: {
      getMetadata(): Promise<GuiMetadata>;
      saveConfig(config: PersistedConfig): Promise<PersistedConfig>;
      restartCore(config: PersistedConfig): Promise<{ ok: boolean }>;
      sendCommand(type: CoreCommandType, payload: Record<string, unknown>): Promise<CoreFrame>;
      onCoreEvent(callback: (frame: CoreFrame) => void): () => void;
      onCoreLog(callback: (message: string) => void): () => void;
    };
  }
}

export {};
