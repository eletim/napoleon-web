import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { createLocalJWKSet, exportJWK, generateKeyPair, SignJWT } from "jose";
import Fastify, { type FastifyInstance } from "fastify";
import { buildApp } from "../src/app.js";
import { readAuthConfig, registerAuth } from "../src/auth.js";
import type { Identity, PlayerProfile, PlayerRepository } from "../src/players.js";
import { games, type InternalGameState } from "../src/store.js";

const issuer = "https://auth.example.com/realms/napoleon";
const clientId = "napoleon-web";
let keys: Awaited<ReturnType<typeof generateKeyPair>>;
let verificationKey: ReturnType<typeof createLocalJWKSet>;
const apps: FastifyInstance[] = [];
beforeAll(async () => {
  keys = await generateKeyPair("RS256");
  verificationKey = createLocalJWKSet({ keys: [{ ...await exportJWK(keys.publicKey), kid: "test", alg: "RS256" }] });
});
afterEach(async () => { await Promise.all(apps.splice(0).map((app) => app.close())); games.clear(); });
async function token(sub = "alice", overrides: Record<string, unknown> = {}) {
  return new SignJWT({ typ: "Bearer", azp: clientId, preferred_username: sub, realm_access: { roles: ["user"] }, ...overrides })
    .setProtectedHeader({ alg: "RS256", kid: "test" }).setIssuer(issuer).setAudience(clientId)
    .setSubject(sub).setIssuedAt().setExpirationTime("5m").sign(keys.privateKey);
}
function memoryPlayers(): PlayerRepository {
  const records = new Map<string, PlayerProfile>();
  const get = (identity: Identity): PlayerProfile => {
    const key = `${identity.issuer}/${identity.subject}`;
    if (!records.has(key)) records.set(key, { id: identity.subject, displayName: identity.username });
    return records.get(key)!;
  };
  return { getOrCreate: async (i) => get(i), rename: async (i, name) => {
    const record = get(i); record.displayName = name; return record;
  }, close: async () => {} };
}
async function app() {
  const instance = Fastify(); apps.push(instance);
  await registerAuth(instance, { config: { issuer, clientId, databaseUrl: "unused" },
    players: memoryPlayers(), verificationKey });
  instance.get("/api/health", async () => ({ ok: true }));
  instance.get("/api/games/:gameId", async () => ({ ok: true }));
  instance.post("/api/games/:gameId/actions", async () => ({ ok: true }));
  return instance;
}
function headers(value: string) { return { authorization: `Bearer ${value}` }; }

describe("Keycloak authentication", () => {
  it("assigns ownership when creating games through the real API", async () => {
    const instance = await buildApp({ auth: { config: { issuer, clientId, databaseUrl: "unused" },
      players: memoryPlayers(), verificationKey } });
    apps.push(instance);
    const alice = headers(await token()); const bob = headers(await token("bob"));
    const created = await instance.inject({ method: "POST", url: "/api/games", headers: alice, payload: {} });
    expect(created.statusCode).toBe(201);
    const id = created.json().gameId;
    expect((await instance.inject({ url: `/api/games/${id}`, headers: alice })).statusCode).toBe(200);
    for (const suffix of ["", "/actions", "/next-trick", "/next-round"]) {
      const method = suffix === "" || suffix === "/diagnostics" ? "GET" : "POST";
      expect((await instance.inject({ method, url: `/api/games/${id}${suffix}`, headers: bob })).statusCode).toBe(404);
    }
  });
  it("enforces user/admin roles for UI-related APIs even with direct requests", async () => {
    const instance = await buildApp({ auth: { config: { issuer, clientId, databaseUrl: "unused" },
      players: memoryPlayers(), verificationKey } });
    apps.push(instance);
    const user = headers(await token());
    const admin = headers(await token("admin", { realm_access: { roles: ["admin"] } }));
    const noRole = headers(await token("none", { realm_access: { roles: [] } }));
    for (const [method, url] of [["GET", "/api/agents"], ["GET", "/api/ai-presets"], ["HEAD", "/api/ai-presets"], ["POST", "/api/simulations"],
      ["PUT", "/api/ai-presets/com-ai"], ["GET", "/api/games/missing/diagnostics"]] as const) {
      expect((await instance.inject({ method, url, headers: user })).statusCode).toBe(403);
      const response = await instance.inject({ method, url, headers: admin });
      expect(response.statusCode).not.toBe(403);
      expect(response.statusCode).not.toBe(401);
    }
    expect((await instance.inject({ url: "/api/me", headers: noRole })).statusCode).toBe(403);
    expect((await instance.inject({ method: "POST", url: "/api/games", payload: {}, headers: noRole })).statusCode).toBe(403);
    expect((await instance.inject({ url: "/api/ai-presets", headers: user })).statusCode).toBe(403);
    expect((await instance.inject({ url: "/api/me", headers: admin })).statusCode).toBe(200);
  });
  it("rejects partial configuration and non-HTTPS remote issuers", () => {
    expect(readAuthConfig({})).toBeNull();
    expect(() => readAuthConfig({ NODE_ENV: "production" })).toThrow("required in production");
    expect(() => readAuthConfig({ NAPOLEON_AUTH_CLIENT_ID: clientId })).toThrow();
    expect(() => readAuthConfig({ NAPOLEON_AUTH_ISSUER: "http://remote/realms/test",
      NAPOLEON_AUTH_CLIENT_ID: clientId, NAPOLEON_DATABASE_URL: "db" })).toThrow();
  });
  it("exposes public config without exposing the DB URL and protects API calls", async () => {
    const instance = await app();
    const config = await instance.inject({ url: "/api/auth/config" });
    expect(config.json()).toEqual({ enabled: true, url: "https://auth.example.com", realm: "napoleon", clientId });
    expect(config.headers["cache-control"]).toBe("no-store");
    expect((await instance.inject({ url: "/api/health" })).statusCode).toBe(200);
    expect((await instance.inject({ url: "/api/me" })).statusCode).toBe(401);
  });
  it("rejects tampered signatures, wrong issuer/audience, expired tokens and ID tokens", async () => {
    const instance = await app();
    const wrongIssuer = await new SignJWT({ typ: "Bearer", azp: clientId }).setProtectedHeader({ alg: "RS256", kid: "test" })
      .setIssuer("https://wrong/realms/test").setAudience(clientId).setSubject("alice").setExpirationTime("5m").sign(keys.privateKey);
    const wrongAudience = await new SignJWT({ typ: "Bearer", azp: clientId }).setProtectedHeader({ alg: "RS256", kid: "test" })
      .setIssuer(issuer).setAudience("other").setSubject("alice").setExpirationTime("5m").sign(keys.privateKey);
    const expired = await new SignJWT({ typ: "Bearer", azp: clientId }).setProtectedHeader({ alg: "RS256", kid: "test" })
      .setIssuer(issuer).setAudience(clientId).setSubject("alice").setExpirationTime(1).sign(keys.privateKey);
    const valid = await token();
    const parts = valid.split("."); parts[1] = Buffer.from('{"sub":"bob"}').toString("base64url");
    for (const value of [parts.join("."), wrongIssuer, wrongAudience, expired, await token("alice", { typ: "ID" }), await token("alice", { azp: "other" })]) {
      expect((await instance.inject({ url: "/api/me", headers: headers(value) })).statusCode).toBe(401);
    }
  });
  it("keeps profile names separate for different users and validates updates", async () => {
    const instance = await app(); const alice = headers(await token()); const bob = headers(await token("bob"));
    expect((await instance.inject({ url: "/api/me", headers: alice })).json().displayName).toBe("alice");
    expect((await instance.inject({ method: "PUT", url: "/api/me", headers: alice, payload: { displayName: " 新しい名前 " } })).json().displayName).toBe("新しい名前");
    expect((await instance.inject({ url: "/api/me", headers: alice })).json().displayName).toBe("新しい名前");
    expect((await instance.inject({ url: "/api/me", headers: bob })).json().displayName).toBe("bob");
    for (const displayName of [" ", "a".repeat(31), "name\n", 123]) {
      const response = await instance.inject({ method: "PUT", url: "/api/me", headers: alice, payload: { displayName } });
      // Trailing whitespace is trimmed before validation.
      expect(response.statusCode).toBe(displayName === "name\n" ? 200 : 400);
    }
  });
  it("blocks another user's game reads and actions, including unowned development games", async () => {
    const instance = await app();
    games.set("owned", { ownerSubject: "alice" } as InternalGameState);
    games.set("legacy", {} as InternalGameState);
    const alice = headers(await token()); const bob = headers(await token("bob"));
    expect((await instance.inject({ url: "/api/games/owned", headers: alice })).statusCode).toBe(200);
    for (const id of ["owned", "legacy"]) for (const method of ["GET", "POST"] as const) {
      expect((await instance.inject({ method, url: `/api/games/${id}${method === "POST" ? "/actions" : ""}`, headers: bob })).statusCode).toBe(404);
    }
  });
});
