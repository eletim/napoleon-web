import { describe, expect, it } from "vitest";
import { createInitialGame, createPlayerView, getLegalActions } from "@napoleon/game-core";
import { NoLegalActionsError, RandomAgent } from "../src/index.js";

describe("RandomAgent", () => {
  it("returns one of the legal actions when legal actions exist", async () => {
    const state = createInitialGame({ rng: () => 0 });
    const playerId = state.currentPlayerId;
    const view = createPlayerView(state, playerId);
    const legalActions = getLegalActions(state, playerId);
    const agent = new RandomAgent(() => 0.7);

    const action = await agent.selectAction({ playerId, view, legalActions });

    expect(legalActions).toContainEqual(action);
  });

  it("throws NoLegalActionsError when no legal actions exist", async () => {
    const state = createInitialGame({ rng: () => 0 });
    const playerId = "player-1";
    const view = createPlayerView(state, playerId);
    const agent = new RandomAgent(() => 0);

    await expect(
      agent.selectAction({ playerId, view, legalActions: [] })
    ).rejects.toBeInstanceOf(NoLegalActionsError);
  });
});
