/// <reference types="node" />

import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

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
const serviceWorkerPath = fileURLToPath(new URL("../public/sw.js", import.meta.url));

describe("PWA assets", () => {
  it("defines an installable standalone web app manifest", () => {
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as WebAppManifest;

    expect(manifest).toMatchObject({
      name: "Napoleon Web",
      short_name: "Napoleon",
      start_url: "/",
      scope: "/",
      display: "standalone",
      theme_color: "#2e6b52",
      background_color: "#e7ebf0"
    });
    expect(manifest.icons).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          src: "/icons/napoleon-192.png",
          sizes: "192x192",
          type: "image/png"
        }),
        expect.objectContaining({
          src: "/icons/napoleon-512.png",
          sizes: "512x512",
          type: "image/png"
        }),
        expect.objectContaining({
          src: "/icons/napoleon-maskable-512.png",
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
      const iconPath = fileURLToPath(new URL(`../public${icon.src}`, import.meta.url));
      const signature = readFileSync(iconPath).subarray(0, 8);

      expect(icon.type).toBe("image/png");
      expect(existsSync(iconPath)).toBe(true);
      expect([...signature]).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
    }

    expect(existsSync(`${webRoot}/public/icons/napoleon-180.png`)).toBe(true);
  });

  it("registers an update-friendly service worker cache strategy", () => {
    const serviceWorker = readFileSync(serviceWorkerPath, "utf8");

    expect(serviceWorker).toContain("self.skipWaiting()");
    expect(serviceWorker).toContain("self.clients.claim()");
    expect(serviceWorker).toContain("networkFirst(request)");
    expect(serviceWorker).toContain("url.pathname === \"/sw.js\"");
    expect(serviceWorker).toContain("url.pathname.startsWith(\"/api/\")");
  });
});
