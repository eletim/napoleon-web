import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { parseAllowedHosts } from "./viteAllowedHosts";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const allowedHosts = parseAllowedHosts(env.VITE_ALLOWED_HOSTS);

  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: 5173,
      allowedHosts
    }
  };
});
