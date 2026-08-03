import { afterEach, describe, expect, it, vi } from "vitest";
import type { CreateGameResponse, RunAutomatedSimulationResponse } from "@napoleon/protocol";
import { createGame, runAutomatedSimulation } from "./api";

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

  it("runs automated simulations through a same-origin relative URL", async () => {
    const responseBody = {
      schemaVersion: 1,
      seed: 42,
      playerIds: [],
      initialHands: {},
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
});
