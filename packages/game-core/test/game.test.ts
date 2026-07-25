import { describe, expect, it } from "vitest";
import {
  applyAction,
  createDeck,
  createInitialGame,
  createPlayerView,
  getLegalActions,
  type GameState
} from "../src/index.js";

const noShuffle = (): number => 0;

function playCurrentPlayerFirstCard(state: GameState): GameState {
  const action = getLegalActions(state, state.currentPlayerId).find(
    (candidate) => candidate.type === "play-card"
  );

  if (action === undefined) {
    throw new Error("Expected a legal play-card action.");
  }

  return applyAction(state, action);
}

describe("game-core", () => {
  it("creates a deck without duplicate cards", () => {
    const deck = createDeck();
    const ids = new Set(deck.map((card) => card.id));

    expect(deck).toHaveLength(52);
    expect(ids.size).toBe(52);
  });

  it("creates 5 players after initialization", () => {
    const state = createInitialGame({ rng: noShuffle });

    expect(state.players).toHaveLength(5);
    expect(state.players.every((player) => player.hand.length === 10)).toBe(true);
    expect(state.unusedCards).toHaveLength(2);
  });

  it("removes a played card from hand and adds it to the trick", () => {
    const state = createInitialGame({ rng: noShuffle });
    const player = state.players[0];
    const card = player.hand[0];
    const next = applyAction(state, {
      type: "play-card",
      playerId: player.id,
      cardId: card.id
    });

    expect(next.players[0].hand.map((candidate) => candidate.id)).not.toContain(card.id);
    expect(next.currentTrick).toEqual([{ playerId: player.id, card }]);
  });

  it("rejects a card that is not in the player's hand", () => {
    const state = createInitialGame({ rng: noShuffle });
    const player = state.players[0];
    const otherCard = state.players[1].hand[0];

    expect(() =>
      applyAction(state, {
        type: "play-card",
        playerId: player.id,
        cardId: otherCard.id
      })
    ).toThrow("not in this player's hand");
  });

  it("rejects playing on another player's turn", () => {
    const state = createInitialGame({ rng: noShuffle });
    const player = state.players[1];
    const card = player.hand[0];

    expect(() =>
      applyAction(state, {
        type: "play-card",
        playerId: player.id,
        cardId: card.id
      })
    ).toThrow("not this player's turn");
  });

  it("marks the trick complete after 5 players play", () => {
    const state = Array.from({ length: 5 }).reduce<GameState>(
      (current) => playCurrentPlayerFirstCard(current),
      createInitialGame({ rng: noShuffle })
    );

    expect(state.currentTrick).toHaveLength(5);
    expect(state.isTrickComplete).toBe(true);
    expect(state.completedTricks).toHaveLength(1);
    expect(state.completedTricks[0].winnerId).toBe("player-0");
  });

  it("clears the table for the next trick", () => {
    const completed = Array.from({ length: 5 }).reduce<GameState>(
      (current) => playCurrentPlayerFirstCard(current),
      createInitialGame({ rng: noShuffle })
    );
    const next = applyAction(completed, {
      type: "next-trick",
      playerId: completed.currentPlayerId
    });

    expect(next.currentTrick).toHaveLength(0);
    expect(next.trickNumber).toBe(2);
    expect(next.isTrickComplete).toBe(false);
  });

  it("does not leak other players' hands in createPlayerView", () => {
    const state = createInitialGame({ rng: noShuffle });
    const view = createPlayerView(state, "player-0");

    expect(view.players[0].hand).toHaveLength(10);
    expect(view.players[1].hand).toBeUndefined();
    expect(view.players[1].handCount).toBe(10);
  });
});
