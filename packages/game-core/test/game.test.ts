import { describe, expect, it } from "vitest";
import {
  advanceToNextTrick,
  applyAction,
  createDeck,
  createInitialGame,
  createPlayerView,
  determineTrickWinner,
  GameRuleError,
  getLeadSuit,
  getLegalActions,
  getPlayableCards,
  getRankValue,
  type Card,
  type GameState
} from "../src/index.js";

const noShuffle = (): number => 0;

function card(suit: Card["suit"], rank: Card["rank"]): Card {
  return {
    id: `${suit}-${rank}`,
    suit,
    rank
  };
}

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

  it("uses a distinct error when initialized with the wrong player count", () => {
    try {
      createInitialGame({ playerIds: ["a", "b", "c", "d"] });
      throw new Error("Expected createInitialGame to fail.");
    } catch (error) {
      expect(error).toBeInstanceOf(GameRuleError);
      if (error instanceof GameRuleError) {
        expect(error.code).toBe("INVALID_PLAYER_COUNT");
      }
    }
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
    expect(state.completedTricks[0].winnerId).toBe(state.currentPlayerId);
  });

  it("clears the table for the next trick", () => {
    const completed = Array.from({ length: 5 }).reduce<GameState>(
      (current) => playCurrentPlayerFirstCard(current),
      createInitialGame({ rng: noShuffle })
    );
    const next = advanceToNextTrick(completed);

    expect(next.currentTrick).toHaveLength(0);
    expect(next.trickNumber).toBe(2);
    expect(next.isTrickComplete).toBe(false);
  });

  it("only exposes play-card as a GameAction", () => {
    const state = createInitialGame({ rng: noShuffle });
    const actions = getLegalActions(state, state.currentPlayerId);

    expect(actions.length).toBeGreaterThan(0);
    expect(actions.every((action) => action.type === "play-card")).toBe(true);
  });

  it("does not expose next-trick as a legal action after a trick completes", () => {
    const completed = Array.from({ length: 5 }).reduce<GameState>(
      (current) => playCurrentPlayerFirstCard(current),
      createInitialGame({ rng: noShuffle })
    );

    expect(getLegalActions(completed, completed.currentPlayerId)).toEqual([]);
  });

  it("rejects advancing to the next trick before the trick is complete", () => {
    const state = createInitialGame({ rng: noShuffle });

    expect(() => advanceToNextTrick(state)).toThrow("not complete");
  });

  it("keeps the lead player unchanged when advancing to the next trick", () => {
    const completed = Array.from({ length: 5 }).reduce<GameState>(
      (current) => playCurrentPlayerFirstCard(current),
      createInitialGame({ rng: noShuffle })
    );
    const next = advanceToNextTrick(completed);

    expect(next.currentPlayerId).toBe(completed.currentPlayerId);
  });

  it("advances to the next trick even when an AI player is the lead player", () => {
    const base = Array.from({ length: 5 }).reduce<GameState>(
      (current) => playCurrentPlayerFirstCard(current),
      createInitialGame({ rng: noShuffle })
    );
    const completedWithAiLead: GameState = {
      ...base,
      currentPlayerId: "player-1",
      completedTricks: [
        {
          trickNumber: base.trickNumber,
          winnerId: "player-1",
          cards: base.currentTrick
        }
      ]
    };

    const next = advanceToNextTrick(completedWithAiLead);

    expect(next.currentTrick).toEqual([]);
    expect(next.currentPlayerId).toBe("player-1");
  });

  it("returns undefined lead suit for an empty trick", () => {
    expect(getLeadSuit([])).toBeUndefined();
  });

  it("uses the first played card as the lead suit", () => {
    expect(
      getLeadSuit([
        { playerId: "player-0", card: card("hearts", "7") },
        { playerId: "player-1", card: card("spades", "A") }
      ])
    ).toBe("hearts");
  });

  it("allows every card when leading a trick", () => {
    const hand = [card("hearts", "7"), card("spades", "A")];

    expect(getPlayableCards(hand, [])).toEqual(hand);
  });

  it("requires follow suit when the player has the lead suit", () => {
    const hand = [card("hearts", "7"), card("spades", "A"), card("hearts", "K")];
    const trick = [{ playerId: "player-0", card: card("hearts", "2") }];

    expect(getPlayableCards(hand, trick).map((candidate) => candidate.id)).toEqual([
      "hearts-7",
      "hearts-K"
    ]);
  });

  it("allows every card when the player cannot follow suit", () => {
    const hand = [card("clubs", "7"), card("spades", "A")];
    const trick = [{ playerId: "player-0", card: card("hearts", "2") }];

    expect(getPlayableCards(hand, trick)).toEqual(hand);
  });

  it("rejects directly playing off suit when the player can follow suit", () => {
    const state = createStateWithHands({
      hands: [
        [card("hearts", "2")],
        [card("hearts", "K"), card("spades", "A")],
        [card("clubs", "3")],
        [card("diamonds", "4")],
        [card("clubs", "5")]
      ]
    });
    const led = applyAction(state, {
      type: "play-card",
      playerId: "player-0",
      cardId: "hearts-2"
    });

    expect(() =>
      applyAction(led, {
        type: "play-card",
        playerId: "player-1",
        cardId: "spades-A"
      })
    ).toThrow("follow the lead suit");
  });

  it("allows playing off suit when the player cannot follow suit", () => {
    const state = createStateWithHands({
      hands: [
        [card("hearts", "2")],
        [card("spades", "A")],
        [card("clubs", "3")],
        [card("diamonds", "4")],
        [card("clubs", "5")]
      ]
    });
    const led = applyAction(state, {
      type: "play-card",
      playerId: "player-0",
      cardId: "hearts-2"
    });
    const next = applyAction(led, {
      type: "play-card",
      playerId: "player-1",
      cardId: "spades-A"
    });

    expect(next.currentTrick[1]).toEqual({ playerId: "player-1", card: card("spades", "A") });
  });

  it("orders ranks from 2 through ace", () => {
    expect(getRankValue("A")).toBeGreaterThan(getRankValue("K"));
    expect(getRankValue("K")).toBeGreaterThan(getRankValue("Q"));
    expect(getRankValue("Q")).toBeGreaterThan(getRankValue("J"));
    expect(getRankValue("J")).toBeGreaterThan(getRankValue("10"));
    expect(getRankValue("10")).toBeGreaterThan(getRankValue("9"));
    expect(getRankValue("3")).toBeGreaterThan(getRankValue("2"));
  });

  it("determines the highest lead-suit card as the trick winner", () => {
    const winnerId = determineTrickWinner([
      { playerId: "player-0", card: card("hearts", "7") },
      { playerId: "player-1", card: card("hearts", "K") },
      { playerId: "player-2", card: card("spades", "A") },
      { playerId: "player-3", card: card("hearts", "3") },
      { playerId: "player-4", card: card("clubs", "Q") }
    ]);

    expect(winnerId).toBe("player-1");
  });

  it("rejects determining a winner for an empty trick", () => {
    expect(() => determineTrickWinner([])).toThrow("empty trick");
  });

  it("sets the trick winner as current player and completed trick winner", () => {
    const state = createStateWithHands({
      hands: [
        [card("hearts", "7"), card("clubs", "6")],
        [card("hearts", "K"), card("clubs", "2")],
        [card("spades", "A"), card("clubs", "3")],
        [card("hearts", "3"), card("diamonds", "4")],
        [card("clubs", "Q"), card("spades", "5")]
      ]
    });
    const completed = [
      { playerId: "player-0", cardId: "hearts-7" },
      { playerId: "player-1", cardId: "hearts-K" },
      { playerId: "player-2", cardId: "spades-A" },
      { playerId: "player-3", cardId: "hearts-3" },
      { playerId: "player-4", cardId: "clubs-Q" }
    ].reduce(
      (current, action) =>
        applyAction(current, {
          type: "play-card",
          playerId: action.playerId,
          cardId: action.cardId
        }),
      state
    );
    const next = advanceToNextTrick(completed);

    expect(completed.currentPlayerId).toBe("player-1");
    expect(completed.completedTricks[0].winnerId).toBe("player-1");
    expect(next.currentPlayerId).toBe("player-1");
  });

  it("does not leak other players' hands in createPlayerView", () => {
    const state = createInitialGame({ rng: noShuffle });
    const view = createPlayerView(state, "player-0");

    expect(view.players[0].hand).toHaveLength(10);
    expect(view.players[1].hand).toBeUndefined();
    expect(view.players[1].handCount).toBe(10);
  });
});

function createStateWithHands(options: {
  hands: readonly [
    readonly Card[],
    readonly Card[],
    readonly Card[],
    readonly Card[],
    readonly Card[]
  ];
  currentPlayerId?: string;
  currentTrick?: GameState["currentTrick"];
  completedTricks?: GameState["completedTricks"];
  trickNumber?: number;
  isTrickComplete?: boolean;
  isGameOver?: boolean;
}): GameState {
  const playerIds = ["player-0", "player-1", "player-2", "player-3", "player-4"];

  return {
    players: playerIds.map((id, index) => ({
      id,
      hand: options.hands[index]
    })),
    currentPlayerId: options.currentPlayerId ?? "player-0",
    currentTrick: options.currentTrick ?? [],
    completedTricks: options.completedTricks ?? [],
    trickNumber: options.trickNumber ?? 1,
    isTrickComplete: options.isTrickComplete ?? false,
    isGameOver: options.isGameOver ?? false,
    unusedCards: []
  };
}
