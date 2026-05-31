import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    watch: {
      ignored: [
        "**/build/**",
        "**/coverage/**",
        "**/dist/**",
        "**/dist-electron/**",
        "**/src-tauri/target/**"
      ]
    }
  },
  build: {
    outDir: "dist",
    emptyOutDir: true
  }
});
