/// <reference types="node" />

import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { resolveAppPath } from "./appPath";

interface WebAppManifest {
  name?: string;
  short_name?: string;
  start_url?: string;
  scope?: string;
  display?: string;
  theme_color?: string;
  background_color?: string;
  icons?: Array<{
    src?: string;
    sizes?: string;
    type?: string;
    purpose?: string;
  }>;
}

const webRoot = fileURLToPath(new URL("..", import.meta.url));
const manifestPath = fileURLToPath(
  new URL("../public/manifest.webmanifest", import.meta.url)
);
const registrationPath = fileURLToPath(new URL("./registerServiceWorker.ts", import.meta.url));
const serviceWorkerPath = fileURLToPath(new URL("../public/sw.js", import.meta.url));

describe("PWA assets", () => {
  it("defines an installable standalone web app manifest", () => {
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as WebAppManifest;

    expect(manifest).toMatchObject({
      name: "Napoleon Web",
      short_name: "Napoleon",
      start_url: ".",
      scope: ".",
      display: "standalone",
      theme_color: "#2e6b52",
      background_color: "#e7ebf0"
    });
    expect(manifest.icons).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          src: "icons/napoleon-192.png",
          sizes: "192x192",
          type: "image/png"
        }),
        expect.objectContaining({
          src: "icons/napoleon-512.png",
          sizes: "512x512",
          type: "image/png"
        }),
        expect.objectContaining({
          src: "icons/napoleon-maskable-512.png",
          sizes: "512x512",
          type: "image/png",
          purpose: "maskable"
        })
      ])
    );
  });

  it("ships every manifest icon as a PNG asset", () => {
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as WebAppManifest;

    for (const icon of manifest.icons ?? []) {
      const iconPath = fileURLToPath(new URL(`../public/${icon.src}`, import.meta.url));
      const signature = readFileSync(iconPath).subarray(0, 8);

      expect(icon.type).toBe("image/png");
      expect(existsSync(iconPath)).toBe(true);
      expect([...signature]).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
    }

    expect(existsSync(`${webRoot}/public/icons/napoleon-180.png`)).toBe(true);
  });

  it("registers an update-friendly service worker cache strategy", () => {
    const registration = readFileSync(registrationPath, "utf8");
    const serviceWorker = readFileSync(serviceWorkerPath, "utf8");

    expect(registration).toContain("import.meta.env.DEV ? \"sw.js?dev-sw\" : \"sw.js\"");
    expect(registration).toContain(
      "navigator.serviceWorker\n      .register(serviceWorkerUrl, {\n        scope: serviceWorkerScope,\n        updateViaCache: \"none\""
    );
    expect(registration).not.toContain("window.location.reload()");
    expect(serviceWorker).toContain("self.skipWaiting()");
    expect(serviceWorker).toContain("self.clients.claim()");
    expect(serviceWorker).toContain("networkFirst(request)");
    expect(serviceWorker).toContain("url.pathname === SERVICE_WORKER_PATHNAME");
    expect(serviceWorker).toContain(
      "url.pathname.startsWith(new URL(\"api/\", SCOPE_URL).pathname)"
    );
  });

  it("keeps the development service worker installable without static asset caching", () => {
    const registration = readFileSync(registrationPath, "utf8");
    const serviceWorker = readFileSync(serviceWorkerPath, "utf8");

    expect(registration).not.toContain("import.meta.env.PROD");
    expect(serviceWorker).toContain("IS_DEV_SERVICE_WORKER");
    expect(serviceWorker).toContain("searchParams.has(\"dev-sw\")");
    expect(serviceWorker).toContain("if (IS_DEV_SERVICE_WORKER) {\n    return;\n  }");
    expect(serviceWorker).toContain("cacheName.startsWith(\"napoleon-web-\")");
  });

  it("resolves app URLs under root and /napoleon/ base paths", () => {
    expect(resolveAppPath("manifest.webmanifest", "/")).toBe("/manifest.webmanifest");
    expect(resolveAppPath("sw.js?dev-sw", "/")).toBe("/sw.js?dev-sw");
    expect(resolveAppPath("api/games", "/")).toBe("/api/games");
    expect(resolveAppPath("manifest.webmanifest", "/napoleon/")).toBe(
      "/napoleon/manifest.webmanifest"
    );
    expect(resolveAppPath("sw.js?dev-sw", "/napoleon/")).toBe("/napoleon/sw.js?dev-sw");
    expect(resolveAppPath("api/games", "/napoleon/")).toBe("/napoleon/api/games");
  });

  it("keeps HTML PWA assets and entrypoint relative to Vite base", () => {
    const html = readFileSync(fileURLToPath(new URL("../index.html", import.meta.url)), "utf8");

    expect(html).toContain('href="%BASE_URL%manifest.webmanifest"');
    expect(html).toContain('href="%BASE_URL%icons/napoleon-192.png"');
    expect(html).toContain('href="%BASE_URL%icons/napoleon-180.png"');
    expect(html).toContain('src="%BASE_URL%src/main.tsx"');
  });
});
