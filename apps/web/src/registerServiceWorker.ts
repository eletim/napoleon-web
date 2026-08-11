export function registerServiceWorker(): void {
  if (!("serviceWorker" in navigator) || !import.meta.env.PROD) {
    return;
  }

  window.addEventListener("load", () => {
    const hadController = navigator.serviceWorker.controller !== null;

    void navigator.serviceWorker.register("/sw.js").then((registration) => {
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

    let hasReloadedForUpdate = false;

    navigator.serviceWorker.addEventListener("controllerchange", () => {
      if (!hadController || hasReloadedForUpdate) {
        return;
      }

      hasReloadedForUpdate = true;
      window.location.reload();
    });
  });
}
