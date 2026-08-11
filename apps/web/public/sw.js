const CACHE_NAME = "napoleon-web-v1";
const APP_SHELL_URLS = [
  "/",
  "/index.html",
  "/manifest.webmanifest",
  "/icons/napoleon-192.png",
  "/icons/napoleon-512.png",
  "/icons/napoleon-maskable-512.png"
];
const MAX_RUNTIME_CACHE_ENTRIES = 64;

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(APP_SHELL_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((cacheNames) =>
        Promise.all(
          cacheNames
            .filter((cacheName) => cacheName !== CACHE_NAME)
            .map((cacheName) => caches.delete(cacheName))
        )
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener("message", (event) => {
  if (event.data?.type === "SKIP_WAITING") {
    event.waitUntil(self.skipWaiting());
  }
});

self.addEventListener("fetch", (event) => {
  const request = event.request;

  if (request.method !== "GET") {
    return;
  }

  const url = new URL(request.url);

  if (
    url.origin !== self.location.origin ||
    url.pathname === "/sw.js" ||
    url.pathname.startsWith("/api/")
  ) {
    return;
  }

  if (request.mode === "navigate") {
    event.respondWith(networkFirst(request));
    return;
  }

  event.respondWith(cacheFirst(request));
});

async function networkFirst(request) {
  const cache = await caches.open(CACHE_NAME);

  try {
    const response = await fetch(request);

    if (response.ok) {
      await cache.put(request, response.clone());
    }

    return response;
  } catch {
    return (await cache.match(request)) ?? cache.match("/index.html");
  }
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE_NAME);
  const cachedResponse = await cache.match(request);

  if (cachedResponse !== undefined) {
    return cachedResponse;
  }

  const response = await fetch(request);

  if (response.ok) {
    await cache.put(request, response.clone());
    await trimRuntimeCache(cache);
  }

  return response;
}

async function trimRuntimeCache(cache) {
  const requests = await cache.keys();

  if (requests.length <= MAX_RUNTIME_CACHE_ENTRIES) {
    return;
  }

  await cache.delete(requests[0]);
}
