import { describe, expect, it } from "vitest";
import type { UserConfig } from "vite";
import config, { createApiProxyConfig, prefixStrippedBasePathUrl } from "./vite.config";
import { normalizeViteBasePath } from "./viteBasePath";

const userConfig = config as UserConfig;

describe("vite development server config", () => {
  it("proxies API requests to the localhost-only Fastify server", () => {
    expect(userConfig.envDir).toBe(false);
    expect(userConfig.base).toBe("/");
    expect(userConfig.server?.host).toBe("127.0.0.1");
    expect(userConfig.server?.port).toBe(5173);
    expect(userConfig.server?.allowedHosts).not.toBe(true);
    expect(userConfig.server?.proxy).toEqual({
      "/api": {
        target: "http://127.0.0.1:3000",
        changeOrigin: true
      }
    });
  });

  it("normalizes the optional dev subpath base", () => {
    expect(normalizeViteBasePath(undefined)).toBe("/");
    expect(normalizeViteBasePath("")).toBe("/");
    expect(normalizeViteBasePath("/")).toBe("/");
    expect(normalizeViteBasePath("napoleon")).toBe("/napoleon/");
    expect(normalizeViteBasePath("/napoleon")).toBe("/napoleon/");
    expect(normalizeViteBasePath("/napoleon/")).toBe("/napoleon/");
  });

  it("proxies API requests under the optional dev subpath base", () => {
    const proxy = createApiProxyConfig("/napoleon/");
    const subpathProxy = proxy["/napoleon/api"];

    expect(proxy["/api"]).toMatchObject({
      target: "http://127.0.0.1:3000",
      changeOrigin: true
    });
    expect(subpathProxy).toMatchObject({
      target: "http://127.0.0.1:3000",
      changeOrigin: true
    });
    expect(subpathProxy.rewrite("/napoleon/api/games")).toBe("/api/games");
  });

  it("maps Tailscale-stripped backend request paths back to the Vite base path", () => {
    expect(prefixStrippedBasePathUrl("/", "/napoleon/")).toBe("/napoleon/");
    expect(prefixStrippedBasePathUrl("/@vite/client", "/napoleon/")).toBe(
      "/napoleon/@vite/client"
    );
    expect(prefixStrippedBasePathUrl("/src/main.tsx", "/napoleon/")).toBe(
      "/napoleon/src/main.tsx"
    );
    expect(prefixStrippedBasePathUrl("/api/games?seed=1", "/napoleon/")).toBe(
      "/napoleon/api/games?seed=1"
    );
    expect(prefixStrippedBasePathUrl("/napoleon/@vite/client", "/napoleon/")).toBe(
      "/napoleon/@vite/client"
    );
    expect(prefixStrippedBasePathUrl("/@vite/client", "/")).toBe("/@vite/client");
  });
});
