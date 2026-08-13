import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { parseAllowedHosts } from "./viteAllowedHosts";
import { normalizeViteBasePath } from "./viteBasePath";

const basePath = normalizeViteBasePath(process.env.NAPOLEON_WEB_BASE_PATH);
const proxy = createApiProxyConfig(basePath);
const strippedBasePathPlugin = createStrippedBasePathPlugin(basePath);

export function createApiProxyConfig(basePath: string) {
  const apiProxy = {
    target: "http://127.0.0.1:3000",
    changeOrigin: true
  };

  if (basePath === "/") {
    return {
      "/api": apiProxy
    };
  }

  return {
    "/api": apiProxy,
    [`${basePath}api`]: {
      ...apiProxy,
      rewrite: (path: string) => path.slice(basePath.length - 1)
    }
  };
}

export function prefixStrippedBasePathUrl(requestUrl: string, basePath: string): string {
  if (basePath === "/") {
    return requestUrl;
  }

  const url = new URL(requestUrl, "http://vite.local");
  const baseWithoutTrailingSlash = basePath.slice(0, -1);

  if (url.pathname === baseWithoutTrailingSlash || url.pathname.startsWith(basePath)) {
    return requestUrl;
  }

  url.pathname = `${baseWithoutTrailingSlash}${url.pathname}`;
  return `${url.pathname}${url.search}`;
}

function createStrippedBasePathPlugin(basePath: string): Plugin | null {
  if (basePath === "/") {
    return null;
  }

  return {
    name: "napoleon-stripped-base-path",
    apply: "serve",
    configureServer(server) {
      // Tailscale Serve mounts /napoleon/ externally, then forwards stripped
      // backend paths such as /@vite/client to the localhost Vite server.
      server.middlewares.use((request, _response, next) => {
        if (request.url !== undefined) {
          request.url = prefixStrippedBasePathUrl(request.url, basePath);
        }
        next();
      });

      server.httpServer?.prependListener("upgrade", (request) => {
        if (request.url !== undefined) {
          request.url = prefixStrippedBasePathUrl(request.url, basePath);
        }
      });
    }
  };
}

export default defineConfig({
  base: basePath,
  envDir: false,
  plugins: [react(), strippedBasePathPlugin].filter((plugin) => plugin !== null),
  resolve: {
    alias: {
      "@napoleon/game-core": fileURLToPath(
        new URL("../../packages/game-core/src/index.ts", import.meta.url)
      )
    }
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    allowedHosts: parseAllowedHosts(process.env.VITE_ALLOWED_HOSTS),
    proxy
  }
});
