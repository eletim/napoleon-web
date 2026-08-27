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

function sum(
  diagnostics: readonly PublicAiPhaseCallDiagnostics[],
  key: "playingCalls" | "biddingCalls" | "adjutantCalls" | "exchangeCalls" | "fallbackCount" | "illegalCount"
): number {
  return diagnostics.reduce((total, entry) => total + entry[key], 0);
}
