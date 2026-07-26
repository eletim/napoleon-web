import { describe, expect, it } from "vitest";
import {
  applyAction,
  createInitialGame,
  createPlayerView,
  getLegalActions,
  type GameState
} from "@napoleon/game-core";
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

  it("selects pass or the nearest bid candidate during bidding", async () => {
    const state = createInitialGame({ rng: () => 0 });
    const playerId = state.currentPlayerId;
    const view = createPlayerView(state, playerId);
    const legalActions = getLegalActions(state, playerId);
    const agent = new RandomAgent(() => 0.99);

    const action = await agent.selectAction({ playerId, view, legalActions });

    expect(action).toEqual({
      type: "bid",
      playerId,
      suit: "clubs",
      targetPointCards: 13
    });
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

  it("returns a discard action from its own hand during exchange even without legal action enumeration", async () => {
    const state = Array.from({ length: 5 }).reduce<GameState>(
      (current) => applyAction(current, { type: "pass", playerId: current.currentPlayerId }),
      createInitialGame({ rng: () => 0 })
    );
    const playerId = state.currentPlayerId;
    const view = createPlayerView(state, playerId);
    const agent = new RandomAgent(() => 0);

    const action = await agent.selectAction({ playerId, view, legalActions: [] });

    expect(action).toEqual({
      type: "discard-cards",
      playerId,
      cardIds: view.players[0].hand?.slice(0, 3).map((card) => card.id)
    });
  });
});
