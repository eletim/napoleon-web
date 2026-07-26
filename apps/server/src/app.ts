import cors from "@fastify/cors";
import Fastify from "fastify";
import { registerRoutes } from "./routes.js";

export interface BuildAppOptions {
  logger?: boolean;
}

export async function buildApp(options: BuildAppOptions = {}) {
  const app = Fastify({
    logger: options.logger ?? false
  });

  await app.register(cors, {
    origin: ["http://127.0.0.1:5173", "http://localhost:5173"]
  });

  await registerRoutes(app);

  return app;
}
