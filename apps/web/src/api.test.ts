import { afterEach, describe, expect, it, vi } from "vitest";
import type { CreateGameResponse } from "@napoleon/protocol";
import { createGame } from "./api";

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
});
