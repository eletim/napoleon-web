import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { parseAllowedHosts } from "./viteAllowedHosts";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    allowedHosts: parseAllowedHosts(process.env.VITE_ALLOWED_HOSTS)
  }
});
