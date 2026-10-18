import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { AccountGate } from "./AccountGate";
import { App } from "./App";
import { registerServiceWorker } from "./registerServiceWorker";

const rootElement = document.getElementById("root");

if (rootElement === null) {
  throw new Error("Root element was not found.");
}

createRoot(rootElement).render(
  <StrictMode>
    <AccountGate><App /></AccountGate>
  </StrictMode>
);

registerServiceWorker();
