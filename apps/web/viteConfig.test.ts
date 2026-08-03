import { describe, expect, it } from "vitest";
import type { UserConfig } from "vite";
import config from "./vite.config";

const userConfig = config as UserConfig;

describe("vite development server config", () => {
  it("proxies API requests to the localhost-only Fastify server", () => {
    expect(userConfig.envDir).toBe(false);
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
});
