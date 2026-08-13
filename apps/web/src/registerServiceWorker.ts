import { resolveAppPath } from "./appPath";

export function registerServiceWorker(): void {
  if (!("serviceWorker" in navigator)) {
    return;
  }

  const serviceWorkerUrl = resolveAppPath(
    import.meta.env.DEV ? "sw.js?dev-sw" : "sw.js"
  );
  const serviceWorkerScope = resolveAppPath("");

  window.addEventListener("load", () => {
    void navigator.serviceWorker
      .register(serviceWorkerUrl, {
        scope: serviceWorkerScope,
        updateViaCache: "none"
      })
      .then((registration) => {
        if (registration.waiting !== null) {
          registration.waiting.postMessage({ type: "SKIP_WAITING" });
        }

        registration.addEventListener("updatefound", () => {
          const installingWorker = registration.installing;

          if (installingWorker === null) {
            return;
          }

          installingWorker.addEventListener("statechange", () => {
            if (
              installingWorker.state === "installed" &&
              navigator.serviceWorker.controller !== null
            ) {
              installingWorker.postMessage({ type: "SKIP_WAITING" });
            }
          });
        });
      });
  });
}
