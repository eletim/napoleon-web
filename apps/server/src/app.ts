import cors from "@fastify/cors";
import Fastify from "fastify";
import { registerRoutes } from "./routes.js";

export async function buildApp() {
  const app = Fastify({
    logger: true
  });

  await app.register(cors, {
    origin: ["http://127.0.0.1:5173", "http://localhost:5173"]
  });

  await registerRoutes(app);

  return app;
}
