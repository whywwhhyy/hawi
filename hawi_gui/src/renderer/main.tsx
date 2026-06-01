import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./tauri-api";
import "./styles.css";
import "./shadcn-theme.css";

installButtonMouseFocusGuard();

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

function installButtonMouseFocusGuard(): void {
  const root = document.documentElement;
  root.dataset.hawiInputModality = "keyboard";

  document.addEventListener("keydown", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    root.dataset.hawiInputModality = "keyboard";
  }, true);

  document.addEventListener("mousedown", (event) => {
    root.dataset.hawiInputModality = "pointer";
    if (event.button !== 0 || !(event.target instanceof Element)) return;

    const button = event.target.closest("button");
    if (!(button instanceof HTMLButtonElement) || button.disabled) return;

    event.preventDefault();
    if (document.activeElement === button) {
      button.blur();
    }
  }, true);
}
