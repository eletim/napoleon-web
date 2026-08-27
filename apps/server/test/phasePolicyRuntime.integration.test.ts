import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { FastifyInstance } from "fastify";
import type {
  AiPolicyComposition,
  CreateGameResponse,
  GetAgentsResponse,
  GetGamePolicyDiagnosticsResponse,
  PublicAiPhaseCallDiagnostics,
  RunAutomatedSimulationResponse
} from "@napoleon/protocol";
import { buildApp } from "../src/app.js";
import { games } from "../src/store.js";

const FULL_RULE_BASED = {
  playing: "rule-based",
  bidding: "rule-based",
  nonPlaying: "rule-based"
} as const satisfies AiPolicyComposition;

const FULL_LEARNED = {
  playing: "ppo-separated-v1000",
  bidding: "frozen-raise-v1",
  nonPlaying: "parameterized-adjutant-exchange-v1"
} as const satisfies AiPolicyComposition;

const MIXED = {
  playing: "rule-based",
  bidding: "frozen-raise-v1",
  nonPlaying: "parameterized-adjutant-exchange-v1"
} as const satisfies AiPolicyComposition;

describe.sequential("formal phase policy runtime", () => {
  let app: FastifyInstance;

  beforeAll(async () => {
    app = await buildApp();
  });

  afterAll(async () => {
    games.clear();
    await app.close();
  });

  it("publishes independently composable formal phase policies", async () => {
    const response = await app.inject({ method: "GET", url: "/api/agents" });
    const registry = response.json<GetAgentsResponse>().policyRegistry;

    expect(response.statusCode).toBe(200);
    expect(registry?.playing.map(({ id }) => id)).toEqual([
      "rule-based",
      "ppo-separated-v1000"
    ]);
    expect(registry?.bidding.map(({ id }) => id)).toEqual([
      "rule-based",
      "frozen-raise-v1"
    ]);
    expect(registry?.nonPlaying.map(({ id }) => id)).toEqual([
      "rule-based",
      "parameterized-adjutant-exchange-v1"
    ]);
    expect(registry?.nonPlaying[1]?.artifactProvenance).toMatchObject({
      featureSchemaVersion: "1",
      adjutantWeightCount: "35",
      exchangeWeightCount: "60",
      optimizerIssue: "452",
      verificationIssue: "454"
    });
  });

  it.each([
    ["Full RuleBased", FULL_RULE_BASED, 45701],
    ["Full learned", FULL_LEARNED, 45702],
    ["mixed", MIXED, 45703]
  ] as const)("completes %s without fallback or illegal actions", async (_, composition, seed) => {
    const response = await app.inject({
      method: "POST",
      url: "/api/simulations",
      payload: { seed, policyComposition: composition }
    });
    const body = response.json<RunAutomatedSimulationResponse>();

    expect(response.statusCode, response.body).toBe(200);
    expect(body.result.winner).toMatch(/^(napoleon-team|alliance)$/);
    const diagnostics = Object.values(body.policyDiagnostics ?? {});
    expect(diagnostics).toHaveLength(5);
    expect(diagnostics.every((entry) => sameComposition(entry, composition))).toBe(true);
    expect(sum(diagnostics, "biddingCalls")).toBeGreaterThan(0);
    expect(sum(diagnostics, "adjutantCalls")).toBe(1);
    expect(sum(diagnostics, "exchangeCalls")).toBe(1);
    expect(sum(diagnostics, "playingCalls")).toBe(50);
    expect(sum(diagnostics, "fallbackCount")).toBe(0);
    expect(sum(diagnostics, "illegalCount")).toBe(0);
  }, 120_000);

  it("rejects malformed or unknown phase compositions instead of falling back", async () => {
    const response = await app.inject({
      method: "POST",
      url: "/api/simulations",
      payload: {
        seed: 45704,
        policyComposition: {
          playing: "ppo-separated-v9999",
          bidding: "frozen-raise-v1",
          nonPlaying: "parameterized-adjutant-exchange-v1"
        }
      }
    });

    expect(response.statusCode).toBe(400);
    expect(response.json()).toMatchObject({
      error: { code: "INVALID_SIMULATION_REQUEST" }
    });
  });

  it("keeps seeded composed simulations deterministic", async () => {
    const request = {
      method: "POST" as const,
      url: "/api/simulations",
      payload: { seed: 45705, policyComposition: MIXED }
    };
    const first = await app.inject(request);
    const second = await app.inject(request);

    expect(first.statusCode).toBe(200);
    expect(second.json<RunAutomatedSimulationResponse>()).toEqual(
      first.json<RunAutomatedSimulationResponse>()
    );
  });

  it("accepts future UI composition objects per game seat and exposes diagnostics", async () => {
    const created = await app.inject({
      method: "POST",
      url: "/api/games",
      payload: {
        aiAgents: ["player-1", "player-2", "player-3", "player-4"].map((playerId) => ({
          playerId,
          policyComposition: FULL_LEARNED
        }))
      }
    });
    const game = created.json<CreateGameResponse>();
    const diagnosticsResponse = await app.inject({
      method: "GET",
      url: `/api/games/${game.gameId}/diagnostics`
    });
    const diagnostics = diagnosticsResponse.json<GetGamePolicyDiagnosticsResponse>();

    expect(created.statusCode, created.body).toBe(201);
    expect(diagnosticsResponse.statusCode).toBe(200);
    expect(Object.keys(diagnostics.diagnostics)).toEqual([
      "player-1",
      "player-2",
      "player-3",
      "player-4"
    ]);
    expect(Object.values(diagnostics.diagnostics).every(
      (entry) => sameComposition(entry, FULL_LEARNED)
    )).toBe(true);
  });
});

function sum(
  diagnostics: readonly PublicAiPhaseCallDiagnostics[],
  key: "playingCalls" | "biddingCalls" | "adjutantCalls" | "exchangeCalls" | "fallbackCount" | "illegalCount"
): number {
  return diagnostics.reduce((total, entry) => total + entry[key], 0);
}

function sameComposition(
  diagnostics: PublicAiPhaseCallDiagnostics,
  composition: AiPolicyComposition
): boolean {
  return diagnostics.composition.playing === composition.playing &&
    diagnostics.composition.bidding === composition.bidding &&
    diagnostics.composition.nonPlaying === composition.nonPlaying;
}
