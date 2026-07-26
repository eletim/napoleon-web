import { describe, expect, it } from "vitest";
import {
  advanceToNextTrick,
  applyAction,
  compareBids,
  createDeck,
  createInitialGame,
  createPlayerView,
  determineTrickWinner,
  GameRuleError,
  getLeadSuit,
  getLegalActions,
  getPlayableCards,
  getRankValue,
  getTrickCardStrength,
  isJokerCard,
  isStandardCard,
  type Bid,
  type Card,
  type GameState,
  type Rank,
  type Suit
} from "../src/index.js";

const noShuffle = (): number => 0;

const defaultContract = {
  napoleonPlayerId: "player-0",
  trumpSuit: "spades" as const,
  targetPointCards: 13
};

function card(suit: Suit, rank: Rank): Card {
  return {
    type: "standard",
    id: `${suit}-${rank}`,
    suit,
    rank
  };
}

function joker(): Card {
  return {
    type: "joker",
    id: "joker"
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

function createPlayingInitialGame(): GameState {
  return toPlayingState(createInitialGame({ rng: noShuffle }));
}

function toPlayingState(state: GameState): GameState {
  return {
    ...state,
    phase: "playing",
    trumpSuit: defaultContract.trumpSuit,
    contract: defaultContract,
    bidding: null
  };
}

function expectRuleError(work: () => void, code: string): void {
  try {
    work();
    throw new Error(`Expected ${code}.`);
  } catch (error) {
    expect(error).toBeInstanceOf(GameRuleError);
    if (error instanceof GameRuleError) {
      expect(error.code).toBe(code);
    }
  }
}

describe("game-core", () => {
  it("creates a deck without duplicate cards", () => {
    const deck = createDeck();
    const ids = new Set(deck.map((card) => card.id));

    expect(deck).toHaveLength(53);
    expect(ids.size).toBe(53);
    expect(deck.filter(isStandardCard)).toHaveLength(52);
    expect(deck.filter(isJokerCard)).toHaveLength(1);
    expect(deck.at(-1)).toEqual({ type: "joker", id: "joker" });
  });

  it("represents standard cards and the joker with discriminated card types", () => {
    const standard = card("spades", "A");
    const jokerCard = joker();

    expect(standard.type).toBe("standard");
    expect(isStandardCard(standard)).toBe(true);
    expect(isJokerCard(standard)).toBe(false);
    expect(jokerCard.type).toBe("joker");
    expect(isJokerCard(jokerCard)).toBe(true);
    expect(isStandardCard(jokerCard)).toBe(false);
    expect(hasOwn(jokerCard, "suit")).toBe(false);
    expect(hasOwn(jokerCard, "rank")).toBe(false);
  });

  it("creates 5 players after initialization", () => {
    const state = createInitialGame({ rng: noShuffle });

    expect(state.players).toHaveLength(5);
    expect(state.players.every((player) => player.hand.length === 10)).toBe(true);
    const allCards = [...state.players.flatMap((player) => player.hand), ...state.unusedCards];
    const allCardIds = new Set(allCards.map((card) => card.id));

    expect(state.unusedCards).toHaveLength(3);
    expect(allCards).toHaveLength(53);
    expect(allCardIds.size).toBe(53);
    expect(allCards.filter(isJokerCard)).toHaveLength(1);
    expect(state.phase).toBe("bidding");
    expect(state.trumpSuit).toBeNull();
    expect(state.contract).toBeNull();
    expect(state.bidding).toMatchObject({
      starterPlayerId: "player-0",
      highestBid: null,
      consecutivePassCount: 0,
      history: []
    });
    expect(state.buriedCards).toEqual([]);
    expect(state.currentPlayerId).toBe("player-0");
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

  it("orders bids by target count and then bidding suit priority", () => {
    const bid = (suit: Suit, targetPointCards: number): Bid => ({
      playerId: "player-0",
      suit,
      targetPointCards
    });

    expect(compareBids(bid("clubs", 13), bid("diamonds", 13))).toBeLessThan(0);
    expect(compareBids(bid("diamonds", 13), bid("hearts", 13))).toBeLessThan(0);
    expect(compareBids(bid("hearts", 13), bid("spades", 13))).toBeLessThan(0);
    expect(compareBids(bid("spades", 13), bid("clubs", 14))).toBeLessThan(0);
    expect(compareBids(bid("clubs", 19), bid("spades", 19))).toBeLessThan(0);
    expect(compareBids(bid("hearts", 15), bid("hearts", 15))).toBe(0);
  });

  it("lists initial bidding actions for the current player only", () => {
    const state = createInitialGame({ rng: noShuffle });
    const actions = getLegalActions(state, "player-0");

    expect(actions).toHaveLength(29);
    expect(actions[0]).toEqual({ type: "pass", playerId: "player-0" });
    expect(actions.filter((action) => action.type === "bid")).toHaveLength(28);
    expect(getLegalActions(state, "player-1")).toEqual([]);
  });

  it("filters legal bids above the current highest bid", () => {
    const state = applyAction(createInitialGame({ rng: noShuffle }), {
      type: "bid",
      playerId: "player-0",
      suit: "hearts",
      targetPointCards: 13
    });
    const actions = getLegalActions(state, "player-1");

    expect(actions).toContainEqual({ type: "pass", playerId: "player-1" });
    expect(actions).toContainEqual({
      type: "bid",
      playerId: "player-1",
      suit: "spades",
      targetPointCards: 13
    });
    expect(actions).toContainEqual({
      type: "bid",
      playerId: "player-1",
      suit: "clubs",
      targetPointCards: 14
    });
    expect(actions).not.toContainEqual({
      type: "bid",
      playerId: "player-1",
      suit: "hearts",
      targetPointCards: 13
    });
    expect(actions).not.toContainEqual({
      type: "bid",
      playerId: "player-1",
      suit: "diamonds",
      targetPointCards: 13
    });
    expect(actions).not.toContainEqual({
      type: "bid",
      playerId: "player-1",
      suit: "clubs",
      targetPointCards: 13
    });
  });

  it("rejects invalid bids and phase violations", () => {
    const bidding = createInitialGame({ rng: noShuffle });
    const highest = applyAction(bidding, {
      type: "bid",
      playerId: "player-0",
      suit: "hearts",
      targetPointCards: 13
    });
    const playing = createPlayingInitialGame();
    const finished = createStateWithHands({
      hands: [[card("spades", "A")], [], [], [], []],
      phase: "finished",
      isGameOver: true
    });

    expectRuleError(
      () =>
        applyAction(bidding, {
          type: "bid",
          playerId: "player-0",
          suit: "spades",
          targetPointCards: 12
        }),
      "INVALID_BID"
    );
    expectRuleError(
      () =>
        applyAction(bidding, {
          type: "bid",
          playerId: "player-0",
          suit: "spades",
          targetPointCards: 20
        }),
      "INVALID_BID"
    );
    expectRuleError(
      () =>
        applyAction(highest, {
          type: "bid",
          playerId: "player-1",
          suit: "hearts",
          targetPointCards: 13
        }),
      "BID_TOO_LOW"
    );
    expectRuleError(
      () =>
        applyAction(highest, {
          type: "bid",
          playerId: "player-1",
          suit: "diamonds",
          targetPointCards: 13
        }),
      "BID_TOO_LOW"
    );
    expectRuleError(
      () =>
        applyAction(bidding, {
          type: "bid",
          playerId: "player-1",
          suit: "spades",
          targetPointCards: 13
        }),
      "NOT_PLAYERS_TURN"
    );
    expectRuleError(
      () =>
        applyAction(playing, {
          type: "bid",
          playerId: "player-0",
          suit: "spades",
          targetPointCards: 13
        }),
      "INVALID_ACTION_FOR_PHASE"
    );
    expectRuleError(
      () => applyAction(playing, { type: "pass", playerId: "player-0" }),
      "INVALID_ACTION_FOR_PHASE"
    );
    expectRuleError(
      () => applyAction(finished, { type: "pass", playerId: "player-0" }),
      "GAME_OVER"
    );
  });

  it("allows a player to bid again after passing when the highest bid changes", () => {
    const state = [
      { type: "bid" as const, playerId: "player-0", suit: "hearts" as const, targetPointCards: 13 },
      { type: "pass" as const, playerId: "player-1" },
      { type: "bid" as const, playerId: "player-2", suit: "spades" as const, targetPointCards: 13 },
      { type: "pass" as const, playerId: "player-3" },
      { type: "pass" as const, playerId: "player-4" },
      { type: "pass" as const, playerId: "player-0" },
      { type: "bid" as const, playerId: "player-1", suit: "clubs" as const, targetPointCards: 14 }
    ].reduce<GameState>((current, action) => applyAction(current, action), createInitialGame());

    expect(state.phase).toBe("bidding");
    expect(state.currentPlayerId).toBe("player-2");
    expect(state.bidding?.highestBid).toMatchObject({
      playerId: "player-1",
      suit: "clubs",
      targetPointCards: 14
    });
    expect(state.bidding?.consecutivePassCount).toBe(0);
  });

  it("finalizes a normal contract after four passes outside the latest bidder", () => {
    const state = [
      { type: "bid" as const, playerId: "player-0", suit: "hearts" as const, targetPointCards: 13 },
      { type: "pass" as const, playerId: "player-1" },
      { type: "bid" as const, playerId: "player-2", suit: "spades" as const, targetPointCards: 13 },
      { type: "pass" as const, playerId: "player-3" },
      { type: "pass" as const, playerId: "player-4" },
      { type: "pass" as const, playerId: "player-0" },
      { type: "pass" as const, playerId: "player-1" }
    ].reduce<GameState>((current, action) => applyAction(current, action), createInitialGame());

    expect(state.phase).toBe("exchanging");
    expect(state.bidding).toBeNull();
    expect(state.contract).toEqual({
      napoleonPlayerId: "player-2",
      trumpSuit: "spades",
      targetPointCards: 13
    });
    expect(state.trumpSuit).toBe("spades");
    expect(state.currentPlayerId).toBe("player-2");
    expect(state.players.find((player) => player.id === "player-2")?.hand).toHaveLength(13);
    expect(
      state.players
        .filter((player) => player.id !== "player-2")
        .every((player) => player.hand.length === 10)
    ).toBe(true);
    expect(state.unusedCards).toHaveLength(0);
    expect(state.buriedCards).toHaveLength(0);
    expect(countKnownCards(state)).toBe(53);
    expect(getLegalActions(state, "player-2")).toEqual([]);
  });

  it("creates a special spades-12 contract when everyone passes", () => {
    const state = Array.from({ length: 5 }).reduce<GameState>(
      (current) => applyAction(current, { type: "pass", playerId: current.currentPlayerId }),
      createInitialGame({ rng: noShuffle })
    );

    expect(state.phase).toBe("exchanging");
    expect(state.contract).toEqual({
      napoleonPlayerId: "player-0",
      trumpSuit: "spades",
      targetPointCards: 12
    });
    expect(state.trumpSuit).toBe("spades");
    expect(state.currentPlayerId).toBe("player-0");
    expect(state.players[0].hand).toHaveLength(13);
    expect(state.unusedCards).toHaveLength(0);
  });

  it("moves three discarded cards into buriedCards and starts playing", () => {
    const exchanging = createAllPassExchangeState();
    const discardIds = exchanging.players[0].hand.slice(0, 3).map((card) => card.id);
    const next = applyAction(exchanging, {
      type: "discard-cards",
      playerId: "player-0",
      cardIds: discardIds
    });

    expect(next.phase).toBe("playing");
    expect(next.currentPlayerId).toBe("player-0");
    expect(next.players[0].hand).toHaveLength(10);
    expect(next.players[0].hand.some((card) => discardIds.includes(card.id))).toBe(false);
    expect(next.buriedCards.map((card) => card.id)).toEqual(discardIds);
    expect(next.currentTrick).toEqual([]);
    expect(next.trickNumber).toBe(1);
    expect(next.isTrickComplete).toBe(false);
    expect(next.isGameOver).toBe(false);
    expect(countKnownCards(next)).toBe(53);
    expect(getLegalActions(next, "player-0").some((action) => action.type === "play-card")).toBe(
      true
    );
  });

  it("allows discarding the joker", () => {
    const exchanging = createAllPassExchangeState();
    const jokerCard = exchanging.players[0].hand.find(isJokerCard);

    expect(jokerCard).toBeDefined();
    if (jokerCard === undefined) {
      throw new Error("Expected joker in player-0 hand after all-pass exchange with no shuffle.");
    }

    const discardIds = [
      jokerCard.id,
      exchanging.players[0].hand[0].id,
      exchanging.players[0].hand[1].id
    ];
    const next = applyAction(exchanging, {
      type: "discard-cards",
      playerId: "player-0",
      cardIds: discardIds
    });

    expect(next.phase).toBe("playing");
    expect(next.players[0].hand.some(isJokerCard)).toBe(false);
    expect(next.buriedCards.some(isJokerCard)).toBe(true);
  });

  it("rejects invalid discard counts, duplicate ids, and cards outside Napoleon's hand", () => {
    const exchanging = createAllPassExchangeState();
    const ids = exchanging.players[0].hand.map((card) => card.id);

    for (const cardIds of [[], [ids[0]], ids.slice(0, 2), ids.slice(0, 4), ids]) {
      expectRuleError(
        () =>
          applyAction(exchanging, {
            type: "discard-cards",
            playerId: "player-0",
            cardIds
          }),
        "INVALID_DISCARD_COUNT"
      );
    }

    expectRuleError(
      () =>
        applyAction(exchanging, {
          type: "discard-cards",
          playerId: "player-0",
          cardIds: [ids[0], ids[0], ids[1]]
        }),
      "DUPLICATE_CARD_ID"
    );
    expectRuleError(
      () =>
        applyAction(exchanging, {
          type: "discard-cards",
          playerId: "player-0",
          cardIds: [ids[0], ids[1], "not-in-hand"]
        }),
      "CARD_NOT_IN_HAND"
    );
    expectRuleError(
      () =>
        applyAction(exchanging, {
          type: "discard-cards",
          playerId: "player-0",
          cardIds: [ids[0], ids[1], exchanging.players[1].hand[0].id]
        }),
      "CARD_NOT_IN_HAND"
    );
  });

  it("rejects discard-cards from invalid actors or phases", () => {
    const exchanging = createAllPassExchangeState();
    const ids = exchanging.players[0].hand.slice(0, 3).map((card) => card.id);
    const playing = applyAction(exchanging, {
      type: "discard-cards",
      playerId: "player-0",
      cardIds: ids
    });
    const finished = { ...playing, phase: "finished" as const, isGameOver: true };

    expectRuleError(
      () =>
        applyAction(exchanging, {
          type: "discard-cards",
          playerId: "player-1",
          cardIds: exchanging.players[1].hand.slice(0, 3).map((card) => card.id)
        }),
      "NOT_NAPOLEON"
    );
    expectRuleError(
      () => applyAction(createInitialGame({ rng: noShuffle }), { type: "discard-cards", playerId: "player-0", cardIds: ids }),
      "INVALID_ACTION_FOR_PHASE"
    );
    expectRuleError(
      () => applyAction(playing, { type: "discard-cards", playerId: "player-0", cardIds: ids }),
      "INVALID_ACTION_FOR_PHASE"
    );
    expectRuleError(
      () => applyAction(finished, { type: "discard-cards", playerId: "player-0", cardIds: ids }),
      "GAME_OVER"
    );
    expectRuleError(
      () =>
        applyAction(exchanging, {
          type: "play-card",
          playerId: "player-0",
          cardId: ids[0]
        }),
      "INVALID_ACTION_FOR_PHASE"
    );
    expectRuleError(
      () => applyAction(exchanging, { type: "pass", playerId: "player-0" }),
      "INVALID_ACTION_FOR_PHASE"
    );
    expectRuleError(
      () =>
        applyAction(exchanging, {
          type: "bid",
          playerId: "player-0",
          suit: "spades",
          targetPointCards: 13
        }),
      "INVALID_ACTION_FOR_PHASE"
    );
    expectRuleError(() => advanceToNextTrick(exchanging), "INVALID_ACTION_FOR_PHASE");
  });

  it("rejects card play during bidding and next-trick before playing", () => {
    const state = createInitialGame({ rng: noShuffle });

    expectRuleError(
      () =>
        applyAction(state, {
          type: "play-card",
          playerId: "player-0",
          cardId: state.players[0].hand[0].id
        }),
      "INVALID_ACTION_FOR_PHASE"
    );
    expectRuleError(() => advanceToNextTrick(state), "INVALID_ACTION_FOR_PHASE");
  });

  it("removes a played card from hand and adds it to the trick", () => {
    const state = createPlayingInitialGame();
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
    const state = createPlayingInitialGame();
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
    const state = createPlayingInitialGame();
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
      createPlayingInitialGame()
    );

    expect(state.currentTrick).toHaveLength(5);
    expect(state.isTrickComplete).toBe(true);
    expect(state.completedTricks).toHaveLength(1);
    expect(state.completedTricks[0].winnerId).toBe(state.currentPlayerId);
  });

  it("clears the table for the next trick", () => {
    const completed = Array.from({ length: 5 }).reduce<GameState>(
      (current) => playCurrentPlayerFirstCard(current),
      createPlayingInitialGame()
    );
    const next = advanceToNextTrick(completed);

    expect(next.currentTrick).toHaveLength(0);
    expect(next.trickNumber).toBe(2);
    expect(next.isTrickComplete).toBe(false);
  });

  it("only exposes play-card as a GameAction", () => {
    const state = createPlayingInitialGame();
    const actions = getLegalActions(state, state.currentPlayerId);

    expect(actions.length).toBeGreaterThan(0);
    expect(actions.every((action) => action.type === "play-card")).toBe(true);
  });

  it("does not expose next-trick as a legal action after a trick completes", () => {
    const completed = Array.from({ length: 5 }).reduce<GameState>(
      (current) => playCurrentPlayerFirstCard(current),
      createPlayingInitialGame()
    );

    expect(getLegalActions(completed, completed.currentPlayerId)).toEqual([]);
  });

  it("rejects advancing to the next trick before the trick is complete", () => {
    const state = createPlayingInitialGame();

    expect(() => advanceToNextTrick(state)).toThrow("not complete");
  });

  it("keeps the lead player unchanged when advancing to the next trick", () => {
    const completed = Array.from({ length: 5 }).reduce<GameState>(
      (current) => playCurrentPlayerFirstCard(current),
      createPlayingInitialGame()
    );
    const next = advanceToNextTrick(completed);

    expect(next.currentPlayerId).toBe(completed.currentPlayerId);
  });

  it("advances to the next trick even when an AI player is the lead player", () => {
    const base = Array.from({ length: 5 }).reduce<GameState>(
      (current) => playCurrentPlayerFirstCard(current),
      createPlayingInitialGame()
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
    expect(getLeadSuit([], { trumpSuit: "spades" })).toBeUndefined();
  });

  it("uses the first played card as the lead suit", () => {
    expect(
      getLeadSuit([
        { playerId: "player-0", card: card("hearts", "7") },
        { playerId: "player-1", card: card("spades", "A") }
      ], { trumpSuit: "spades" })
    ).toBe("hearts");
  });

  it("uses trump as the lead suit when joker leads", () => {
    const trick = [{ playerId: "player-0", card: joker() }];

    expect(getLeadSuit(trick, { trumpSuit: "spades" })).toBe("spades");
    expect(getLeadSuit(trick, { trumpSuit: "hearts" })).toBe("hearts");
    expectRuleError(() => getLeadSuit(trick, { trumpSuit: null }), "TRUMP_NOT_SET");
  });

  it("allows every card when leading a trick", () => {
    const hand = [card("hearts", "7"), card("spades", "A"), joker()];

    expect(getPlayableCards(hand, [], { trumpSuit: "spades" })).toEqual(hand);
  });

  it("requires follow suit when the player has the lead suit", () => {
    const hand = [card("hearts", "7"), card("spades", "A"), joker(), card("hearts", "K")];
    const trick = [{ playerId: "player-0", card: card("hearts", "2") }];

    expect(getPlayableCards(hand, trick, { trumpSuit: "spades" }).map((candidate) => candidate.id)).toEqual([
      "hearts-7",
      "joker",
      "hearts-K"
    ]);
  });

  it("allows every card when the player cannot follow suit", () => {
    const hand = [card("clubs", "7"), card("spades", "A"), joker()];
    const trick = [{ playerId: "player-0", card: card("hearts", "2") }];

    expect(getPlayableCards(hand, trick, { trumpSuit: "spades" })).toEqual(hand);
  });

  it("uses trump as follow suit when joker leads", () => {
    const hand = [card("spades", "2"), card("hearts", "A"), joker()];
    const trick = [{ playerId: "player-0", card: joker() }];

    expect(getPlayableCards(hand, trick, { trumpSuit: "spades" }).map((candidate) => candidate.id)).toEqual([
      "spades-2",
      "joker"
    ]);
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

  it("allows directly playing joker even when the player can follow suit", () => {
    const state = createStateWithHands({
      hands: [
        [card("hearts", "2")],
        [card("hearts", "K"), card("spades", "A"), joker()],
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
      cardId: "joker"
    });

    expect(next.currentTrick[1]).toEqual({ playerId: "player-1", card: joker() });
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
    const winnerId = determineTrickWinner(
      [
        { playerId: "player-0", card: card("hearts", "7") },
        { playerId: "player-1", card: card("hearts", "K") },
        { playerId: "player-2", card: card("spades", "A") },
        { playerId: "player-3", card: card("hearts", "3") },
        { playerId: "player-4", card: card("clubs", "Q") }
      ],
      { trumpSuit: null }
    );

    expect(winnerId).toBe("player-1");
  });

  it("rejects determining a winner for an empty trick", () => {
    expect(() => determineTrickWinner([], { trumpSuit: "spades" })).toThrow("empty trick");
  });

  it("sets the trick winner as current player and completed trick winner", () => {
    const state = createStateWithHands({
      hands: [
        [card("hearts", "7"), card("clubs", "6")],
        [card("hearts", "K"), card("clubs", "2")],
        [card("spades", "A"), card("clubs", "3")],
        [card("hearts", "3"), card("diamonds", "4")],
        [card("clubs", "Q"), card("spades", "5")]
      ],
      trumpSuit: "diamonds"
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

  it("categorizes trump, lead, and other trick cards", () => {
    expect(
      getTrickCardStrength(card("spades", "2"), "hearts", { trumpSuit: "spades" })
    ).toMatchObject({
      category: "trump",
      categoryPriority: 2
    });
    expect(
      getTrickCardStrength(card("hearts", "A"), "hearts", { trumpSuit: "spades" })
    ).toMatchObject({
      category: "lead",
      categoryPriority: 1
    });
    expect(
      getTrickCardStrength(card("clubs", "A"), "hearts", { trumpSuit: "spades" })
    ).toMatchObject({
      category: "other",
      categoryPriority: 0
    });
    expect(
      getTrickCardStrength(card("spades", "A"), "spades", { trumpSuit: "spades" })
    ).toMatchObject({
      category: "trump",
      categoryPriority: 2
    });
    expect(
      getTrickCardStrength(card("spades", "A"), "hearts", { trumpSuit: null })
    ).toMatchObject({
      category: "other",
      categoryPriority: 0
    });
  });

  it("lets a low trump beat a high lead-suit card", () => {
    const winnerId = determineTrickWinner(
      [
        { playerId: "player-0", card: card("hearts", "A") },
        { playerId: "player-1", card: card("spades", "2") },
        { playerId: "player-2", card: card("hearts", "K") },
        { playerId: "player-3", card: card("clubs", "A") },
        { playerId: "player-4", card: card("diamonds", "A") }
      ],
      { trumpSuit: "spades" }
    );

    expect(winnerId).toBe("player-1");
  });

  it("uses the highest rank among trumps when multiple trumps are played", () => {
    const winnerId = determineTrickWinner(
      [
        { playerId: "player-0", card: card("hearts", "A") },
        { playerId: "player-1", card: card("spades", "2") },
        { playerId: "player-2", card: card("spades", "K") },
        { playerId: "player-3", card: card("hearts", "K") },
        { playerId: "player-4", card: card("clubs", "A") }
      ],
      { trumpSuit: "spades" }
    );

    expect(winnerId).toBe("player-2");
  });

  it("lets a leading joker win when no trump is played", () => {
    expect(
      determineTrickWinner(
        [
          { playerId: "player-0", card: joker() },
          { playerId: "player-1", card: card("hearts", "A") },
          { playerId: "player-2", card: card("clubs", "A") },
          { playerId: "player-3", card: card("diamonds", "A") },
          { playerId: "player-4", card: card("hearts", "K") }
        ],
        { trumpSuit: "spades" }
      )
    ).toBe("player-0");
    expect(
      determineTrickWinner(
        [
          { playerId: "player-0", card: joker() },
          { playerId: "player-1", card: card("spades", "A") },
          { playerId: "player-2", card: card("clubs", "A") },
          { playerId: "player-3", card: card("diamonds", "A") },
          { playerId: "player-4", card: card("spades", "K") }
        ],
        { trumpSuit: "hearts" }
      )
    ).toBe("player-0");
  });

  it("lets the weakest standard trump beat a leading joker", () => {
    expect(
      determineTrickWinner(
        [
          { playerId: "player-0", card: joker() },
          { playerId: "player-1", card: card("spades", "2") },
          { playerId: "player-2", card: card("hearts", "A") },
          { playerId: "player-3", card: card("clubs", "A") },
          { playerId: "player-4", card: card("diamonds", "A") }
        ],
        { trumpSuit: "spades" }
      )
    ).toBe("player-1");
    expect(
      determineTrickWinner(
        [
          { playerId: "player-0", card: joker() },
          { playerId: "player-1", card: card("hearts", "2") },
          { playerId: "player-2", card: card("spades", "A") },
          { playerId: "player-3", card: card("clubs", "A") },
          { playerId: "player-4", card: card("diamonds", "A") }
        ],
        { trumpSuit: "hearts" }
      )
    ).toBe("player-1");
  });

  it("treats a later joker as weaker than standard lead-suit cards", () => {
    expect(
      determineTrickWinner(
        [
          { playerId: "player-0", card: card("hearts", "3") },
          { playerId: "player-1", card: joker() },
          { playerId: "player-2", card: card("clubs", "A") },
          { playerId: "player-3", card: card("diamonds", "A") },
          { playerId: "player-4", card: card("hearts", "2") }
        ],
        { trumpSuit: "spades" }
      )
    ).toBe("player-0");
    expect(
      determineTrickWinner(
        [
          { playerId: "player-0", card: card("hearts", "2") },
          { playerId: "player-1", card: joker() },
          { playerId: "player-2", card: card("clubs", "A") },
          { playerId: "player-3", card: card("diamonds", "A") },
          { playerId: "player-4", card: card("clubs", "K") }
        ],
        { trumpSuit: "spades" }
      )
    ).toBe("player-0");
  });

  it("still lets trump beat a later joker and lead-suit cards", () => {
    const winnerId = determineTrickWinner(
      [
        { playerId: "player-0", card: card("hearts", "A") },
        { playerId: "player-1", card: joker() },
        { playerId: "player-2", card: card("spades", "2") },
        { playerId: "player-3", card: card("clubs", "A") },
        { playerId: "player-4", card: card("diamonds", "A") }
      ],
      { trumpSuit: "spades" }
    );

    expect(winnerId).toBe("player-2");
  });

  it("does not let an off-suit ace win unless it is trump", () => {
    const winnerId = determineTrickWinner(
      [
        { playerId: "player-0", card: card("hearts", "K") },
        { playerId: "player-1", card: card("clubs", "A") },
        { playerId: "player-2", card: card("hearts", "3") },
        { playerId: "player-3", card: card("diamonds", "A") },
        { playerId: "player-4", card: card("clubs", "Q") }
      ],
      { trumpSuit: "spades" }
    );

    expect(winnerId).toBe("player-0");
  });

  it("uses the highest trump when trump is led", () => {
    const winnerId = determineTrickWinner(
      [
        { playerId: "player-0", card: card("spades", "3") },
        { playerId: "player-1", card: card("spades", "A") },
        { playerId: "player-2", card: card("hearts", "A") },
        { playerId: "player-3", card: card("spades", "K") },
        { playerId: "player-4", card: card("clubs", "A") }
      ],
      { trumpSuit: "spades" }
    );

    expect(winnerId).toBe("player-1");
  });

  it("keeps follow-suit obligations independent of trump", () => {
    const trick = [{ playerId: "player-0", card: card("hearts", "K") }];

    expect(
      getPlayableCards([card("hearts", "3"), card("spades", "A")], trick, {
        trumpSuit: "spades"
      }).map((candidate) => candidate.id)
    ).toEqual(["hearts-3"]);
    expect(
      getPlayableCards([card("spades", "A"), card("clubs", "3")], trick, {
        trumpSuit: "spades"
      }).map((candidate) => candidate.id)
    ).toEqual(["spades-A", "clubs-3"]);
  });

  it("rejects directly playing trump when the player can follow lead suit", () => {
    const state = createStateWithHands({
      hands: [
        [card("hearts", "2")],
        [card("hearts", "3"), card("spades", "A")],
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

  it("allows leading with trump", () => {
    const state = createStateWithHands({
      hands: [
        [card("spades", "A")],
        [card("hearts", "3")],
        [card("clubs", "3")],
        [card("diamonds", "4")],
        [card("clubs", "5")]
      ]
    });
    const next = applyAction(state, {
      type: "play-card",
      playerId: "player-0",
      cardId: "spades-A"
    });

    expect(next.currentTrick[0]).toEqual({ playerId: "player-0", card: card("spades", "A") });
  });

  it("sets an AI trump winner as next lead player", () => {
    const state = createStateWithHands({
      hands: [
        [card("hearts", "A"), card("clubs", "6")],
        [card("spades", "2"), card("clubs", "2")],
        [card("hearts", "K"), card("clubs", "3")],
        [card("clubs", "A"), card("diamonds", "4")],
        [card("diamonds", "A"), card("spades", "5")]
      ]
    });
    const completed = [
      { playerId: "player-0", cardId: "hearts-A" },
      { playerId: "player-1", cardId: "spades-2" },
      { playerId: "player-2", cardId: "hearts-K" },
      { playerId: "player-3", cardId: "clubs-A" },
      { playerId: "player-4", cardId: "diamonds-A" }
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

    expect(completed.completedTricks[0].winnerId).toBe("player-1");
    expect(completed.currentPlayerId).toBe("player-1");
    expect(next.currentPlayerId).toBe("player-1");
  });

  it("sets phase to finished when the final trick completes", () => {
    const state = createStateWithHands({
      hands: [
        [card("spades", "A")],
        [card("spades", "K")],
        [card("spades", "Q")],
        [card("spades", "J")],
        [card("spades", "10")]
      ]
    });
    const completed = Array.from({ length: 5 }).reduce<GameState>(
      (current) => playCurrentPlayerFirstCard(current),
      state
    );

    expect(completed.isGameOver).toBe(true);
    expect(completed.phase).toBe("finished");
    expect(getLegalActions(completed, completed.currentPlayerId)).toEqual([]);
    expectRuleError(
      () => applyAction(completed, { type: "pass", playerId: completed.currentPlayerId }),
      "GAME_OVER"
    );
    expectRuleError(
      () =>
        applyAction(completed, {
          type: "play-card",
          playerId: completed.currentPlayerId,
          cardId: "spades-A"
        }),
      "GAME_OVER"
    );
  });

  it("does not leak other players' hands in createPlayerView", () => {
    const state = createInitialGame({ rng: noShuffle });
    const view = createPlayerView(state, "player-0");

    expect(view.phase).toBe("bidding");
    expect(view.trumpSuit).toBeNull();
    expect(view.contract).toBeNull();
    expect(view.bidding?.starterPlayerId).toBe("player-0");
    expect(view.players[0].hand).toHaveLength(10);
    expect(view.players[1].hand).toBeUndefined();
    expect(view.players[1].handCount).toBe(10);
  });

  it("exposes public bidding state and only the player's legal bidding actions", () => {
    const state = applyAction(createInitialGame({ rng: noShuffle }), {
      type: "bid",
      playerId: "player-0",
      suit: "hearts",
      targetPointCards: 13
    });
    const view = createPlayerView(state, "player-1");

    expect(view.phase).toBe("bidding");
    expect(view.trumpSuit).toBeNull();
    expect(view.contract).toBeNull();
    expect(view.bidding).toMatchObject({
      starterPlayerId: "player-0",
      consecutivePassCount: 0,
      highestBid: {
        playerId: "player-0",
        suit: "hearts",
        targetPointCards: 13
      }
    });
    expect(view.bidding?.history).toEqual([
      {
        type: "bid",
        playerId: "player-0",
        suit: "hearts",
        targetPointCards: 13
      }
    ]);
    expect(view.players[1].hand).toHaveLength(10);
    expect(view.players[0].hand).toBeUndefined();
    expect(view.legalActions.every((action) => action.playerId === "player-1")).toBe(true);
    expect(view.legalActions.some((action) => action.type === "pass")).toBe(true);
    expect(view.legalActions.some((action) => action.type === "bid")).toBe(true);
  });

  it("exposes the finalized contract through createPlayerView", () => {
    const state = createAllPassExchangeState();
    const view = createPlayerView(state, "player-0");

    expect(view.phase).toBe("exchanging");
    expect(view.contract).toEqual({
      napoleonPlayerId: "player-0",
      trumpSuit: "spades",
      targetPointCards: 12
    });
    expect(view.bidding).toBeNull();
    expect(view.trumpSuit).toBe("spades");
    expect(view.exchangeRequirement).toEqual({ discardCount: 3 });
  });

  it("keeps exchange information private for non-Napoleon views", () => {
    const state = createAllPassExchangeState();
    const napoleonView = createPlayerView(state, "player-0");
    const opponentView = createPlayerView(state, "player-1");

    expect(napoleonView.phase).toBe("exchanging");
    expect(napoleonView.players[0].hand).toHaveLength(13);
    expect(napoleonView.exchangeRequirement).toEqual({ discardCount: 3 });
    expect(opponentView.phase).toBe("exchanging");
    expect(opponentView.players[0].handCount).toBe(13);
    expect(opponentView.players[0].hand).toBeUndefined();
    expect(opponentView.players[1].hand).toHaveLength(10);
    expect(opponentView.exchangeRequirement).toBeNull();
    expect(hasOwn(opponentView, "buriedCards")).toBe(false);
  });

  it("clears exchange requirements after exchange completes", () => {
    const exchanging = createAllPassExchangeState();
    const next = applyAction(exchanging, {
      type: "discard-cards",
      playerId: "player-0",
      cardIds: exchanging.players[0].hand.slice(0, 3).map((card) => card.id)
    });
    const view = createPlayerView(next, "player-1");

    expect(view.phase).toBe("playing");
    expect(view.exchangeRequirement).toBeNull();
    expect(view.players[0].handCount).toBe(10);
    expect(hasOwn(view, "buriedCards")).toBe(false);
  });

  it("exposes null trump suit through createPlayerView", () => {
    const state = createStateWithHands({
      hands: [
        [card("hearts", "2")],
        [card("clubs", "3")],
        [card("diamonds", "4")],
        [card("hearts", "5")],
        [card("clubs", "6")]
      ],
      trumpSuit: null
    });
    const view = createPlayerView(state, "player-0");

    expect(view.trumpSuit).toBeNull();
  });

  it("exposes the same trump suit to every player view without leaking hands", () => {
    const state = createPlayingInitialGame();

    for (const player of state.players) {
      const view = createPlayerView(state, player.id);

      expect(view.trumpSuit).toBe(state.trumpSuit);
      expect(view.players.find((viewPlayer) => viewPlayer.id === player.id)?.hand).toHaveLength(10);
      expect(
        view.players
          .filter((viewPlayer) => viewPlayer.id !== player.id)
          .every((viewPlayer) => viewPlayer.hand === undefined && viewPlayer.handCount === 10)
      ).toBe(true);
      expect(view.legalActions.every((action) => action.playerId === player.id)).toBe(true);
    }
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
  trumpSuit?: GameState["trumpSuit"];
  phase?: GameState["phase"];
  contract?: GameState["contract"];
  bidding?: GameState["bidding"];
  buriedCards?: GameState["buriedCards"];
}): GameState {
  const playerIds = ["player-0", "player-1", "player-2", "player-3", "player-4"];

  return {
    players: playerIds.map((id, index) => ({
      id,
      hand: options.hands[index]
    })),
    phase: options.phase ?? (options.isGameOver === true ? "finished" : "playing"),
    currentPlayerId: options.currentPlayerId ?? "player-0",
    currentTrick: options.currentTrick ?? [],
    completedTricks: options.completedTricks ?? [],
    trumpSuit: options.trumpSuit !== undefined ? options.trumpSuit : "spades",
    contract:
      options.contract !== undefined
        ? options.contract
        : {
            napoleonPlayerId: "player-0",
            trumpSuit: options.trumpSuit ?? "spades",
            targetPointCards: 13
    },
    bidding: options.bidding ?? null,
    buriedCards: options.buriedCards ?? [],
    trickNumber: options.trickNumber ?? 1,
    isTrickComplete: options.isTrickComplete ?? false,
    isGameOver: options.isGameOver ?? false,
    unusedCards: []
  };
}

function hasOwn(value: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function createAllPassExchangeState(): GameState {
  return Array.from({ length: 5 }).reduce<GameState>(
    (current) => applyAction(current, { type: "pass", playerId: current.currentPlayerId }),
    createInitialGame({ rng: noShuffle })
  );
}

function countKnownCards(state: GameState): number {
  return [
    ...state.players.flatMap((player) => player.hand),
    ...state.buriedCards,
    ...state.unusedCards,
    ...state.currentTrick.map((played) => played.card),
    ...state.completedTricks.flatMap((trick) => trick.cards.map((played) => played.card))
  ].length;
}
