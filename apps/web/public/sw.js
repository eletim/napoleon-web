const CACHE_NAME = "napoleon-web-v1";
const IS_DEV_SERVICE_WORKER = new URL(self.location.href).searchParams.has("dev-sw");
const SERVICE_WORKER_PATHNAME = new URL(self.location.href).pathname;
const SCOPE_URL = new URL(self.registration.scope);
const SCOPE_PATHNAME = SCOPE_URL.pathname;
const APP_SHELL_URLS = [
  "",
  "index.html",
  "manifest.webmanifest",
  "icons/napoleon-192.png",
  "icons/napoleon-512.png",
  "icons/napoleon-maskable-512.png"
].map((path) => new URL(path, SCOPE_URL).href);
const MAX_RUNTIME_CACHE_ENTRIES = 64;

self.addEventListener("install", (event) => {
  if (IS_DEV_SERVICE_WORKER) {
    event.waitUntil(self.skipWaiting());
    return;
  }

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
          cacheNames.map((cacheName) => {
            if (IS_DEV_SERVICE_WORKER) {
              return cacheName.startsWith("napoleon-web-")
                ? caches.delete(cacheName)
                : Promise.resolve(false);
            }

            return cacheName !== CACHE_NAME
              ? caches.delete(cacheName)
              : Promise.resolve(false);
          })
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
  if (IS_DEV_SERVICE_WORKER) {
    return;
  }

  const request = event.request;

  if (request.method !== "GET") {
    return;
  }

  const url = new URL(request.url);

  if (
    url.origin !== self.location.origin ||
    !isWithinScope(url.pathname) ||
    url.pathname === SERVICE_WORKER_PATHNAME ||
    url.pathname.startsWith(new URL("api/", SCOPE_URL).pathname)
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
    return (await cache.match(request)) ?? cache.match(new URL("index.html", SCOPE_URL).href);
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

function isWithinScope(pathname) {
  return pathname === SCOPE_PATHNAME || pathname.startsWith(SCOPE_PATHNAME);
}
