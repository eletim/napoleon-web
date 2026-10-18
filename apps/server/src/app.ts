import { registerAuth, type AuthOptions } from "./auth.js";
import { createCompetitionStore, type CompetitionStore } from "./competitionStore.js";
import cors from "@fastify/cors";
import Fastify from "fastify";
import { registerRoutes } from "./routes.js";
import type { AgentRegistry } from "./agentRegistry.js";
import type { AiPresetRegistry } from "./aiPresetRegistry.js";

export interface BuildAppOptions {
  logger?: boolean;
  competitionStore?: CompetitionStore;
  matchmakingWaitMs?: number;
  now?: () => number;
  auth?: AuthOptions;
  agentRegistry?: AgentRegistry;
  aiPresetRegistry?: AiPresetRegistry;
}

export async function buildApp(options: BuildAppOptions = {}) {
  const app = Fastify({
    logger: options.logger ?? false
  });

  await app.register(cors, {
    origin: ["http://127.0.0.1:5173", "http://localhost:5173"]
  });

  const auth = await registerAuth(app, options.auth);

  await registerRoutes(app, {
    players: auth?.players,
    competitionStore: options.competitionStore ?? await createCompetitionStore(auth?.databaseUrl),
    matchmakingWaitMs: options.matchmakingWaitMs,
    now: options.now,
    agentRegistry: options.agentRegistry,
    aiPresetRegistry: options.aiPresetRegistry
  });

  return app;
}
