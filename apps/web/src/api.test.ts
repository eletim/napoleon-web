import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  AiPreset,
  CreateGameResponse,
  GetAiPresetsResponse,
  GetAgentsResponse,
  RunAutomatedSimulationResponse
} from "@napoleon/protocol";
import {
  createGame,
  getAgents,
  getAiPresets,
  runAutomatedSimulation,
  updateAiPreset
} from "./api";
import { resolveAppPath } from "./appPath";

describe("api client", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("creates games through a same-origin relative URL", async () => {
    const responseBody = {
      gameId: "game-1",
      playerId: "player-0",
      state: {}
    } as unknown as CreateGameResponse;
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify(responseBody), {
        status: 201,
        headers: {
          "Content-Type": "application/json"
        }
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(createGame()).resolves.toEqual(responseBody);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith("/api/games", {
      method: "POST",
      body: "{}",
      headers: {
        "Content-Type": "application/json"
      }
    });
  });

  it("gets available agents through a same-origin relative URL", async () => {
    const responseBody: GetAgentsResponse = {
      agents: [
        {
          id: "rule-based",
          displayName: "Rule-based AI",
          isAvailable: true
        }
      ]
    };
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify(responseBody), {
        status: 200,
        headers: {
          "Content-Type": "application/json"
        }
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(getAgents()).resolves.toEqual(responseBody);

    expect(fetchMock).toHaveBeenCalledWith("/api/agents", {
      headers: undefined
    });
  });

  it("sends selected AI agents when creating a game", async () => {
    const responseBody = {
      gameId: "game-1",
      playerId: "player-0",
      state: {}
    } as unknown as CreateGameResponse;
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify(responseBody), {
        status: 201,
        headers: {
          "Content-Type": "application/json"
        }
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      createGame({
        aiAgents: [
          {
            playerId: "player-1",
            agentId: "playing-policy-onnx"
          }
        ]
      })
    ).resolves.toEqual(responseBody);

    expect(fetchMock).toHaveBeenCalledWith("/api/games", {
      method: "POST",
      body: "{\"aiAgents\":[{\"playerId\":\"player-1\",\"agentId\":\"playing-policy-onnx\"}]}",
      headers: {
        "Content-Type": "application/json"
      }
    });
  });

  it("starts all AI seats from a single preset ID", async () => {
    const responseBody = {
      gameId: "game-preset",
      playerId: "player-0",
      state: {}
    } as unknown as CreateGameResponse;
    const fetchMock = vi.fn(async () => new Response(JSON.stringify(responseBody), {
      status: 201,
      headers: { "Content-Type": "application/json" }
    }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(createGame({ aiPresetId: "com-ai" })).resolves.toEqual(responseBody);
    expect(fetchMock).toHaveBeenCalledWith("/api/games", {
      method: "POST",
      body: "{\"aiPresetId\":\"com-ai\"}",
      headers: { "Content-Type": "application/json" }
    });
  });

  it("loads and updates AI presets without sending artifact paths", async () => {
    const presets = {
      presets: [createComAiPreset()],
      policyRegistry: { playing: [], bidding: [], nonPlaying: [] }
    } satisfies GetAiPresetsResponse;
    const saved = {
      ...createComAiPreset(),
      composition: {
        ...createComAiPreset().composition,
        playing: "rule-based" as const
      }
    } satisfies AiPreset;
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(presets), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      }))
      .mockResolvedValueOnce(new Response(JSON.stringify(saved), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(getAiPresets()).resolves.toEqual(presets);
    await expect(updateAiPreset("com-ai", saved.composition)).resolves.toEqual(saved);
    expect(fetchMock).toHaveBeenNthCalledWith(1, "/api/ai-presets", { headers: undefined });
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/api/ai-presets/com-ai", {
      method: "PUT",
      body: JSON.stringify({ composition: saved.composition }),
      headers: { "Content-Type": "application/json" }
    });
    expect(JSON.stringify(fetchMock.mock.calls)).not.toMatch(/onnxPath|artifactPath/);
  });

  it("runs automated simulations through a same-origin relative URL", async () => {
    const responseBody = {
      schemaVersion: 1,
      seed: 42,
      playerIds: [],
      initialHands: {},
      initialActualState: {
        hands: {},
        unusedCardIds: [],
        excludedCardIds: [],
        awardedPointCardIds: {},
        currentTrickCardIds: [],
        completedTrickCardIds: []
      },
      decisions: [],
      summary: {
        totalDecisionCount: 0,
        decisionCountByPlayer: {},
        decisionCountByPhase: {
          bidding: 0,
          exchanging: 0,
          "choosing-adjutant": 0,
          playing: 0,
          finished: 0
        },
        actionCountByType: {}
      },
      result: {
        resultType: "standard",
        winner: "alliance",
        napoleonTeamPointCards: 0,
        alliancePointCards: 0,
        targetPointCards: 13,
        napoleonPlayerId: "player-0",
        adjutantPlayerId: null
      }
    } as RunAutomatedSimulationResponse;
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify(responseBody), {
        status: 200,
        headers: {
          "Content-Type": "application/json"
        }
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(runAutomatedSimulation(42)).resolves.toEqual(responseBody);

    expect(fetchMock).toHaveBeenCalledWith("/api/simulations", {
      method: "POST",
      body: "{\"seed\":42}",
      headers: {
        "Content-Type": "application/json"
      }
    });
  });

  it("resolves API URLs under the configured app base path", () => {
    expect(resolveAppPath("api/agents", "/napoleon/")).toBe("/napoleon/api/agents");
    expect(resolveAppPath("/api/games", "napoleon")).toBe("/napoleon/api/games");
  });
});

function createComAiPreset(): AiPreset {
  return {
    id: "com-ai",
    displayName: "COM-AI",
    composition: {
      playing: "ppo-separated-v1000",
      bidding: "frozen-raise-v1",
      nonPlaying: "parameterized-adjutant-exchange-v1"
    }
  };
}
