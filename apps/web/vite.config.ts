import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { parseAllowedHosts } from "./viteAllowedHosts";

export default defineConfig({
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
    proxy: {
      "/api": {
        target: "http://127.0.0.1:3000",
        changeOrigin: true
      }
    }
  }
});
