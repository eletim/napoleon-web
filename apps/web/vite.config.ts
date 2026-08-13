import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { parseAllowedHosts } from "./viteAllowedHosts";
import { normalizeViteBasePath } from "./viteBasePath";

const basePath = normalizeViteBasePath(process.env.NAPOLEON_WEB_BASE_PATH);
const proxy = createApiProxyConfig(basePath);

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

export default defineConfig({
  base: basePath,
  envDir: false,
  plugins: [react()],
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
