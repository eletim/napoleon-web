import { describe, expect, it } from "vitest";
import {
  applyAction,
  createDeck,
  createInitialGame,
  createPlayerView,
  getLegalActions,
  isStandardCard,
  type Card,
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
    const state = createAllPassExchangeState();
    const playerId = state.currentPlayerId;
    const view = createPlayerView(state, playerId);
    const agent = new RandomAgent(() => 0);

    const action = await agent.selectAction({ playerId, view, legalActions: [] });

    expect(view.phase).toBe("exchanging");
    expect(view.players[0].hand).toHaveLength(13);
    expect(action).toEqual({
      type: "discard-cards",
      playerId,
      cardIds: view.players[0].hand?.slice(0, 3).map((card) => card.id)
    });
  });

  it("returns a standard adjutant choice during adjutant choice without legal action enumeration", async () => {
    const choosing = createAllPassAdjutantChoiceState();
    const playerId = choosing.currentPlayerId;
    const view = createPlayerView(choosing, playerId);
    const agent = new RandomAgent(() => 0);

    const action = await agent.selectAction({ playerId, view, legalActions: [] });

    expect(view.phase).toBe("choosing-adjutant");
    expect(view.players[0].hand).toHaveLength(10);
    expect(action).toEqual({
      type: "choose-adjutant",
      playerId,
      cardId: expect.stringMatching(/^(spades|hearts|diamonds|clubs)-/)
    });
    expect(action.type === "choose-adjutant" ? action.cardId : "joker").not.toBe("joker");
  });

  it("prefers adjutant cards by rank before suit", async () => {
    const choosing = createAllPassAdjutantChoiceState();
    const playerId = choosing.currentPlayerId;
    const view = createPlayerView(choosing, playerId);
    const spadeAce: Card = {
      type: "standard",
      id: "spades-A",
      suit: "spades",
      rank: "A"
    };
    const adjustedView = {
      ...view,
      players: view.players.map((player) =>
        player.id === playerId ? { ...player, hand: [spadeAce], handCount: 1 } : player
      )
    };
    const agent = new RandomAgent(() => 0);

    const action = await agent.selectAction({
      playerId,
      view: adjustedView,
      legalActions: []
    });

    expect(action).toEqual({
      type: "choose-adjutant",
      playerId,
      cardId: "hearts-A"
    });
  });

  it("can choose the joker as adjutant when every standard candidate is in its own hand", async () => {
    const choosing = createAllPassAdjutantChoiceState();
    const playerId = choosing.currentPlayerId;
    const view = createPlayerView(choosing, playerId);
    const allStandardCards = createDeck().filter(isStandardCard);
    const adjustedView = {
      ...view,
      players: view.players.map((player) =>
        player.id === playerId
          ? { ...player, hand: allStandardCards, handCount: allStandardCards.length }
          : player
      )
    };
    const agent = new RandomAgent(() => 0);

    const action = await agent.selectAction({
      playerId,
      view: adjustedView,
      legalActions: []
    });

    expect(action).toEqual({
      type: "choose-adjutant",
      playerId,
      cardId: "joker"
    });
  });
});

function createAllPassAdjutantChoiceState(): GameState {
  return Array.from({ length: 5 }).reduce<GameState>(
    (current) => applyAction(current, { type: "pass", playerId: current.currentPlayerId }),
    createInitialGame({ rng: () => 0 })
  );
}

function createAllPassExchangeState(): GameState {
  const choosing = createAllPassAdjutantChoiceState();

  return applyAction(choosing, {
    type: "choose-adjutant",
    playerId: choosing.currentPlayerId,
    cardId: "spades-A"
  });
}
