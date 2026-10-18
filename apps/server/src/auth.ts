import { createRemoteJWKSet, jwtVerify, type JWTVerifyGetKey } from "jose";
import type { FastifyInstance } from "fastify";
import { games } from "./store.js";
import { createPlayerRepository, type Identity, type PlayerRepository } from "./players.js";

export interface AuthConfig {
  issuer: string;
  clientId: string;
  databaseUrl: string;
  jwksUrl?: string;
}
export interface AuthOptions {
  config?: AuthConfig | null;
  players?: PlayerRepository;
  verificationKey?: JWTVerifyGetKey;
}

declare module "fastify" {
  interface FastifyRequest { identity: Identity | null; appRoles: string[] | null; }
}

export function readAuthConfig(env: NodeJS.ProcessEnv = process.env): AuthConfig | null {
  const { NAPOLEON_AUTH_ISSUER: issuer, NAPOLEON_AUTH_CLIENT_ID: clientId,
    NAPOLEON_DATABASE_URL: databaseUrl, NAPOLEON_AUTH_JWKS_URL: jwksUrl } = env;
  if (!issuer && !clientId && !databaseUrl && !jwksUrl) {
    if (env.NODE_ENV === "production") throw new Error("Authentication configuration is required in production.");
    return null;
  }
  if (!issuer || !clientId || !databaseUrl) {
    throw new Error("Set NAPOLEON_AUTH_ISSUER, NAPOLEON_AUTH_CLIENT_ID and NAPOLEON_DATABASE_URL together.");
  }
  const url = new URL(issuer);
  if (!/^https?:$/.test(url.protocol) || url.search || url.hash || url.username || url.password ||
      !/^.*\/realms\/[^/]+$/.test(url.pathname)) {
    throw new Error("NAPOLEON_AUTH_ISSUER must be a Keycloak realm URL without a trailing slash.");
  }
  if (url.protocol !== "https:" && !["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)) {
    throw new Error("Use HTTPS for a non-local Keycloak issuer.");
  }
  if (jwksUrl && !["https:", "http:"].includes(new URL(jwksUrl).protocol)) throw new Error("Invalid JWKS URL.");
  return { issuer, clientId, databaseUrl, ...(jwksUrl ? { jwksUrl } : {}) };
}

export async function registerAuth(app: FastifyInstance, options: AuthOptions = {}): Promise<{ players: PlayerRepository; databaseUrl?: string } | null> {
  const config = options.config === undefined ? readAuthConfig() : options.config;
  app.decorateRequest("identity", null);
  app.decorateRequest("appRoles", null);
  const publicPaths = new Set(["/api/health", "/api/auth/config"]);
  app.addHook("onSend", async (request, reply, payload) => {
    if (request.routeOptions.url?.startsWith("/api/")) reply.header("Cache-Control", "no-store");
    return payload;
  });
  app.get("/api/auth/config", async () => {
    if (config === null) return { enabled: false };
    const separator = config.issuer.lastIndexOf("/realms/");
    return { enabled: true, url: config.issuer.slice(0, separator),
      realm: decodeURIComponent(config.issuer.slice(separator + 8)), clientId: config.clientId };
  });
  if (config === null) return null;
  const players = options.players ?? await createPlayerRepository(config.databaseUrl);
  app.addHook("onClose", async () => { await players.close(); });
  const key = options.verificationKey ?? createRemoteJWKSet(
    new URL(config.jwksUrl ?? `${config.issuer}/protocol/openid-connect/certs`)
  );
  app.addHook("onRequest", async (request, reply) => {
    const path = request.routeOptions.url ?? "";
    if (request.method === "OPTIONS" || publicPaths.has(path) || !path.startsWith("/api/")) return;
    const match = /^Bearer (\S+)$/i.exec(request.headers.authorization ?? "");
    try {
      if (!match) throw new Error("Missing bearer token");
      const { payload } = await jwtVerify(match[1], key, {
        issuer: config.issuer, audience: config.clientId, algorithms: ["RS256"]
      });
      if (!payload.sub || payload.typ !== "Bearer" || payload.azp !== config.clientId ||
          typeof payload.exp !== "number") throw new Error("Invalid access token");
      const access = payload.realm_access as { roles?: unknown } | undefined;
      request.appRoles = Array.isArray(access?.roles)
        ? access.roles.filter((role): role is string => typeof role === "string") : [];
      request.identity = { issuer: config.issuer, subject: payload.sub,
        username: typeof payload.preferred_username === "string" && payload.preferred_username.trim()
          ? payload.preferred_username : "プレイヤー" };
    } catch {
      return reply.code(401).header("WWW-Authenticate", "Bearer").send({
        error: { code: "UNAUTHENTICATED", message: "ログインしてください。" }
      });
    }
  });
  app.addHook("preHandler", async (request, reply) => {
    const path = request.routeOptions.url ?? "";
    if (request.method === "OPTIONS" || publicPaths.has(path) || !path.startsWith("/api/")) return;
    if (!request.identity) return reply.code(401).header("WWW-Authenticate", "Bearer").send({
      error: { code: "UNAUTHENTICATED", message: "ログインしてください。" }
    });
    const isAdmin = request.appRoles?.includes("admin");
    const adminOnly = path === "/api/ai-presets" || path === "/api/agents" || path === "/api/simulations" ||
      path === "/api/games/:gameId/diagnostics" ||
      (path === "/api/ai-presets/:presetId" && request.method === "PUT");
    if ((!isAdmin && !request.appRoles?.includes("user")) || (adminOnly && !isAdmin)) {
      return reply.code(403).send({ error: { code: "FORBIDDEN", message: "この機能を利用する権限がありません。" } });
    }
    const params = request.params as { gameId?: string };
    if (params.gameId === undefined) return;
    const game = games.get(params.gameId);
    if (game !== undefined && game.ownerSubject !== request.identity?.subject) {
      return reply.code(404).send({ error: { code: "GAME_NOT_FOUND", message: "Game was not found." } });
    }
  });
  app.get("/api/me", async (request) => players.getOrCreate(request.identity!));
  app.put("/api/me", async (request, reply) => {
    const body = request.body as { displayName?: unknown } | null;
    const name = typeof body?.displayName === "string" ? body.displayName.trim() : "";
    if (!name || [...name].length > 30 || /[\u0000-\u001f\u007f]/.test(name)) {
      return reply.code(400).send({ error: { code: "INVALID_DISPLAY_NAME",
        message: "プレイヤー名は1〜30文字で入力してください。" } });
    }
    return players.rename(request.identity!, name);
  });
  return { players, databaseUrl: options.players ? undefined : config.databaseUrl };
}
