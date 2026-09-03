import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { FastifyInstance } from "fastify";
import type {
  AiPreset,
  CreateGameResponse,
  GetAiPresetsResponse,
  GetGamePolicyDiagnosticsResponse,
  PublicAiPhaseCallDiagnostics,
  RunAutomatedSimulationResponse
} from "@napoleon/protocol";
import { buildApp } from "../src/app.js";
import { createAgentRegistry } from "../src/agentRegistry.js";
import { createPhasePolicyRegistry } from "../src/phasePolicyRegistry.js";
import { games } from "../src/store.js";

describe.sequential("AI preset runtime integration", () => {
  let app: FastifyInstance;

  beforeAll(async () => {
    app = await buildApp();
  });

  afterAll(async () => {
    games.clear();
    await app.close();
  });

  it("lists builtin presets and phase capabilities", async () => {
    const response = await app.inject({ method: "GET", url: "/api/ai-presets" });
    const body = response.json<GetAiPresetsResponse>();

    expect(response.statusCode).toBe(200);
    expect(body.presets.map(({ id, displayName }) => ({ id, displayName }))).toEqual([
      { id: "com-rule-base", displayName: "COM-RuleBase" },
      { id: "com-ai", displayName: "COM-AI" }
    ]);
    expect(body.policyRegistry.playing.map(({ id }) => id)).toContain("ppo-separated-v1000");
  });

  it.each([
    ["COM-RuleBase", "com-rule-base", 45901, {
      playing: "rule-based",
      bidding: "rule-based",
      nonPlaying: "rule-based"
    }],
    ["COM-AI", "com-ai", 45902, {
      playing: "ppo-separated-v1000",
      bidding: "frozen-raise-v1",
      nonPlaying: "parameterized-adjutant-exchange-v1"
    }]
  ] as const)("completes %s through preset resolution", async (_, presetId, seed, expected) => {
    const response = await app.inject({
      method: "POST",
      url: "/api/simulations",
      payload: { seed, aiPresetId: presetId }
    });
    const body = response.json<RunAutomatedSimulationResponse>();
    const diagnostics = Object.values(body.policyDiagnostics ?? {});

    expect(response.statusCode, response.body).toBe(200);
    expect(body.presetId).toBe(presetId);
    expect(body.result.winner).toMatch(/^(napoleon-team|alliance)$/);
    expect(diagnostics).toHaveLength(5);
    expect(diagnostics.every(({ composition }) =>
      composition.playing === expected.playing &&
      composition.bidding === expected.bidding &&
      composition.nonPlaying === expected.nonPlaying
    )).toBe(true);
    expect(sum(diagnostics, "biddingCalls")).toBeGreaterThan(0);
    expect(sum(diagnostics, "adjutantCalls")).toBe(1);
    expect(sum(diagnostics, "exchangeCalls")).toBe(1);
    expect(sum(diagnostics, "playingCalls")).toBe(50);
    expect(sum(diagnostics, "fallbackCount")).toBe(0);
    expect(sum(diagnostics, "illegalCount")).toBe(0);
  }, 120_000);

  it("applies one selected preset to all four game AI seats", async () => {
    const response = await app.inject({
      method: "POST",
      url: "/api/games",
      payload: { aiPresetId: "com-ai" }
    });
    const game = response.json<CreateGameResponse>();
    const diagnosticsResponse = await app.inject({
      method: "GET",
      url: `/api/games/${game.gameId}/diagnostics`
    });
    const body = diagnosticsResponse.json<GetGamePolicyDiagnosticsResponse>();

    expect(response.statusCode, response.body).toBe(201);
    expect(body.presetId).toBe("com-ai");
    expect(Object.keys(body.diagnostics)).toEqual([
      "player-1",
      "player-2",
      "player-3",
      "player-4"
    ]);
    expect(Object.values(body.diagnostics).every(({ composition }) =>
      composition.playing === "ppo-separated-v1000" &&
      composition.bidding === "frozen-raise-v1" &&
      composition.nonPlaying === "parameterized-adjutant-exchange-v1"
    )).toBe(true);
  });

  it("applies a different preset per seat when each seat selects independently", async () => {
    const response = await app.inject({
      method: "POST",
      url: "/api/games",
      payload: {
        aiAgents: [
          {
            playerId: "player-1",
            policyComposition: {
              playing: "rule-based",
              bidding: "rule-based",
              nonPlaying: "rule-based"
            }
          },
          {
            playerId: "player-2",
            policyComposition: {
              playing: "ppo-separated-v1000",
              bidding: "frozen-raise-v1",
              nonPlaying: "parameterized-adjutant-exchange-v1"
            }
          },
          {
            playerId: "player-3",
            policyComposition: {
              playing: "rule-based",
              bidding: "rule-based",
              nonPlaying: "rule-based"
            }
          },
          {
            playerId: "player-4",
            policyComposition: {
              playing: "ppo-separated-v1000",
              bidding: "frozen-raise-v1",
              nonPlaying: "parameterized-adjutant-exchange-v1"
            }
          }
        ]
      }
    });
    const game = response.json<CreateGameResponse>();
    const diagnosticsResponse = await app.inject({
      method: "GET",
      url: `/api/games/${game.gameId}/diagnostics`
    });
    const body = diagnosticsResponse.json<GetGamePolicyDiagnosticsResponse>();

    expect(response.statusCode, response.body).toBe(201);
    // A mixed request never resolves to a single overall preset.
    expect(body.presetId).toBeUndefined();
    expect(body.diagnostics["player-1"]?.composition.playing).toBe("rule-based");
    expect(body.diagnostics["player-2"]?.composition.playing).toBe("ppo-separated-v1000");
    // The same preset (RuleBase) can be reused on more than one seat.
    expect(body.diagnostics["player-3"]?.composition.playing).toBe("rule-based");
    expect(body.diagnostics["player-4"]?.composition.playing).toBe("ppo-separated-v1000");
  });

  it("applies the same explicitly selected preset to all four seats via per-seat selections", async () => {
    const uniformComposition = {
      playing: "ppo-separated-v1000",
      bidding: "frozen-raise-v1",
      nonPlaying: "parameterized-adjutant-exchange-v1"
    } as const;
    const response = await app.inject({
      method: "POST",
      url: "/api/games",
      payload: {
        aiAgents: ["player-1", "player-2", "player-3", "player-4"].map((playerId) => ({
          playerId,
          policyComposition: uniformComposition
        }))
      }
    });
    const game = response.json<CreateGameResponse>();
    const diagnosticsResponse = await app.inject({
      method: "GET",
      url: `/api/games/${game.gameId}/diagnostics`
    });
    const body = diagnosticsResponse.json<GetGamePolicyDiagnosticsResponse>();

    expect(response.statusCode, response.body).toBe(201);
    expect(Object.keys(body.diagnostics)).toEqual([
      "player-1",
      "player-2",
      "player-3",
      "player-4"
    ]);
    expect(Object.values(body.diagnostics).every(({ composition }) =>
      composition.playing === uniformComposition.playing &&
      composition.bidding === uniformComposition.bidding &&
      composition.nonPlaying === uniformComposition.nonPlaying
    )).toBe(true);
  });

  it("persists a settings change and uses it for subsequent games", async () => {
    const mixed = {
      playing: "rule-based",
      bidding: "frozen-raise-v1",
      nonPlaying: "parameterized-adjutant-exchange-v1"
    } as const;
    const saved = await app.inject({
      method: "PUT",
      url: "/api/ai-presets/com-ai",
      payload: { composition: mixed }
    });
    const loaded = await app.inject({ method: "GET", url: "/api/ai-presets" });
    const simulation = await app.inject({
      method: "POST",
      url: "/api/simulations",
      payload: { seed: 45903, aiPresetId: "com-ai" }
    });
    const diagnostics = Object.values(
      simulation.json<RunAutomatedSimulationResponse>().policyDiagnostics ?? {}
    );

    expect(saved.statusCode).toBe(200);
    expect(saved.json<AiPreset>().composition).toEqual(mixed);
    expect(loaded.json<GetAiPresetsResponse>().presets.find(({ id }) => id === "com-ai")
      ?.composition).toEqual(mixed);
    expect(diagnostics.every(({ composition }) => composition.playing === "rule-based")).toBe(true);
    expect(sum(diagnostics, "fallbackCount")).toBe(0);
    expect(sum(diagnostics, "illegalCount")).toBe(0);

    await app.inject({
      method: "PUT",
      url: "/api/ai-presets/com-ai",
      payload: {
        composition: {
          playing: "ppo-separated-v1000",
          bidding: "frozen-raise-v1",
          nonPlaying: "parameterized-adjutant-exchange-v1"
        }
      }
    });
  }, 120_000);

  it("rejects invalid preset and policy IDs", async () => {
    const unknownPreset = await app.inject({
      method: "POST",
      url: "/api/games",
      payload: { aiPresetId: "missing" }
    });
    const invalidPolicy = await app.inject({
      method: "PUT",
      url: "/api/ai-presets/com-ai",
      payload: {
        composition: {
          playing: "missing",
          bidding: "frozen-raise-v1",
          nonPlaying: "parameterized-adjutant-exchange-v1"
        }
      }
    });

    expect(unknownPreset.statusCode).toBe(400);
    expect(unknownPreset.json()).toMatchObject({ error: { code: "UNKNOWN_AI_PRESET_ID" } });
    expect(invalidPolicy.statusCode).toBe(400);
    expect(invalidPolicy.json()).toMatchObject({
      error: { code: "INVALID_AI_PRESET_REQUEST" }
    });
  });
});

describe("AI preset artifact preflight", () => {
  it("rejects COM-AI before game creation when a selected formal policy is unavailable", async () => {
    const phasePolicies = createPhasePolicyRegistry({
      loadPlayingPolicy: async () => {
        throw new Error("actor artifact missing");
      }
    });
    const unavailableApp = await buildApp({
      agentRegistry: createAgentRegistry({ phasePolicyRegistry: phasePolicies })
    });
    const gamesBefore = games.size;

    try {
      const response = await unavailableApp.inject({
        method: "POST",
        url: "/api/games",
        payload: { aiPresetId: "com-ai" }
      });

      expect(response.statusCode).toBe(400);
      expect(response.json()).toMatchObject({
        error: {
          code: "INVALID_AI_PRESET_COMPOSITION",
          message: expect.stringContaining("Unavailable playing policy id")
        }
      });
      expect(games.size).toBe(gamesBefore);

      const directSimulation = await unavailableApp.inject({
        method: "POST",
        url: "/api/simulations",
        payload: {
          seed: 46101,
          policyComposition: {
            playing: "ppo-separated-v1000",
            bidding: "rule-based",
            nonPlaying: "rule-based"
          }
        }
      });
      expect(directSimulation.statusCode).toBe(503);
      expect(directSimulation.json()).toMatchObject({
        error: {
          code: "AGENT_UNAVAILABLE",
          message: expect.stringContaining("ppo-separated-v1000")
        }
      });
    } finally {
      await unavailableApp.close();
    }
  });
});

function sum(
  diagnostics: readonly PublicAiPhaseCallDiagnostics[],
  key: "playingCalls" | "biddingCalls" | "adjutantCalls" | "exchangeCalls" | "fallbackCount" | "illegalCount"
): number {
  return diagnostics.reduce((total, entry) => total + entry[key], 0);
}
