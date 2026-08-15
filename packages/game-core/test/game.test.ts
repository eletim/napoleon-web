import { describe, expect, it } from "vitest";
import {
  advanceToNextTrick,
  applyAction,
  calculateGameResult,
  compareBids,
  createDeck,
  createInitialGame,
  createPlayerView,
  determineCurrentWinningPlayer,
  determineTrickWinner,
  GameRuleError,
  getSeiJackCardId,
  getLeadSuit,
  getLegalActions,
  getPlayableCards,
  getRankValue,
  getTrickCardStrength,
  getUraJackCardId,
  isAdjutantCardId,
  isJokerCard,
  isOrumaCard,
  isPointCard,
  isSeiJackCard,
  isStandardCard,
  isUraJackCard,
  isYoromekiCard,
  jokerCardId,
  orumaCardId,
  type Bid,
  type Card,
  type GameState,
  type Rank,
  type Suit,
  yoromekiCardId
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

function played(playerId: string, playedCard: Card): GameState["currentTrick"][number] {
  return {
    playerId,
    card: playedCard
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
    adjutant: null,
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
    expect(jokerCardId).toBe("joker");
    expect(isAdjutantCardId("spades-A")).toBe(true);
    expect(isAdjutantCardId("joker")).toBe(true);
    expect(isAdjutantCardId("spades-1")).toBe(false);
  });

  it("identifies point cards as each suit's 10, J, Q, K, and A", () => {
    const pointRanks: readonly Rank[] = ["10", "J", "Q", "K", "A"];
    const nonPointRanks: readonly Rank[] = ["2", "3", "4", "5", "6", "7", "8", "9"];
    const suits: readonly Suit[] = ["spades", "hearts", "diamonds", "clubs"];

    for (const suit of suits) {
      for (const rank of pointRanks) {
        expect(isPointCard(card(suit, rank))).toBe(true);
      }

      for (const rank of nonPointRanks) {
        expect(isPointCard(card(suit, rank))).toBe(false);
      }
    }

    expect(isPointCard(joker())).toBe(false);
    expect(createDeck().filter(isPointCard)).toHaveLength(20);
  });

  it("counts sei jack and ura jack as point cards without extra points", () => {
    const pointCards = [card("spades", "J"), card("clubs", "J")];

    expect(pointCards.every(isPointCard)).toBe(true);
    expect(pointCards.filter(isPointCard)).toHaveLength(2);
  });

  it("defines fixed oruma and yoromeki cards", () => {
    expect(orumaCardId).toBe("spades-A");
    expect(yoromekiCardId).toBe("hearts-Q");
    expect(isOrumaCard(card("spades", "A"))).toBe(true);
    expect(isOrumaCard(card("clubs", "A"))).toBe(false);
    expect(isYoromekiCard(card("hearts", "Q"))).toBe(true);
    expect(isYoromekiCard(card("hearts", "K"))).toBe(false);
  });

  it("calculates sei jack and ura jack card ids from trump suit", () => {
    expect(getSeiJackCardId("spades")).toBe("spades-J");
    expect(getSeiJackCardId("clubs")).toBe("clubs-J");
    expect(getSeiJackCardId("hearts")).toBe("hearts-J");
    expect(getSeiJackCardId("diamonds")).toBe("diamonds-J");
    expect(getUraJackCardId("spades")).toBe("clubs-J");
    expect(getUraJackCardId("clubs")).toBe("spades-J");
    expect(getUraJackCardId("hearts")).toBe("diamonds-J");
    expect(getUraJackCardId("diamonds")).toBe("hearts-J");
    expect(isSeiJackCard(card("spades", "J"), "spades")).toBe(true);
    expect(isUraJackCard(card("clubs", "J"), "spades")).toBe(true);
    expect(isUraJackCard(card("clubs", "J"), "clubs")).toBe(false);
  });

  it("lets oruma beat trump, lead cards, and joker unless yoromeki is present", () => {
    expect(
      determineTrickWinner(
        [
          played("player-0", card("spades", "A")),
          played("player-1", card("hearts", "A")),
          played("player-2", card("clubs", "K")),
          played("player-3", card("diamonds", "K")),
          played("player-4", joker())
        ],
        { trumpSuit: "hearts" }
      )
    ).toBe("player-0");
    expect(
      determineTrickWinner(
        [
          played("player-0", card("spades", "A")),
          played("player-1", card("spades", "K")),
          played("player-2", card("clubs", "A")),
          played("player-3", card("hearts", "A")),
          played("player-4", card("diamonds", "A"))
        ],
        { trumpSuit: "spades" }
      )
    ).toBe("player-0");
    expect(
      determineTrickWinner(
        [
          played("player-0", card("spades", "A")),
          played("player-1", card("clubs", "A")),
          played("player-2", card("clubs", "K")),
          played("player-3", card("hearts", "A")),
          played("player-4", card("diamonds", "A"))
        ],
        { trumpSuit: "clubs" }
      )
    ).toBe("player-0");
    expect(
      determineTrickWinner(
        [
          played("player-0", joker()),
          played("player-1", card("spades", "A")),
          played("player-2", card("clubs", "2"))
        ],
        { trumpSuit: "hearts" }
      )
    ).toBe("player-1");
  });

  it("determines the current winning player for a partial trick using the same rules", () => {
    expect(
      determineCurrentWinningPlayer(
        [
          played("player-0", card("hearts", "A")),
          played("player-1", card("spades", "2"))
        ],
        { trumpSuit: "spades" },
        { trickNumber: 2 }
      )
    ).toBe("player-1");
  });

  it("lets yoromeki beat oruma only when both are in the same trick", () => {
    for (const trumpSuit of ["spades", "hearts", "diamonds", "clubs"] satisfies readonly Suit[]) {
      expect(
        determineTrickWinner(
          [
            played("player-0", card("spades", "A")),
            played("player-1", card("hearts", "Q")),
            played("player-2", joker()),
            played("player-3", card("clubs", "A")),
            played("player-4", card("diamonds", "A"))
          ],
          { trumpSuit }
        )
      ).toBe("player-1");
    }
  });

  it("lets sei jack beat ura jack and normal trump cards for every trump suit", () => {
    const cases: readonly Array<{
      trumpSuit: Suit;
      seiJack: Card;
      uraJack: Card;
      highTrump: Card;
    }> = [
      {
        trumpSuit: "spades",
        seiJack: card("spades", "J"),
        uraJack: card("clubs", "J"),
        highTrump: card("spades", "K")
      },
      {
        trumpSuit: "clubs",
        seiJack: card("clubs", "J"),
        uraJack: card("spades", "J"),
        highTrump: card("clubs", "A")
      },
      {
        trumpSuit: "hearts",
        seiJack: card("hearts", "J"),
        uraJack: card("diamonds", "J"),
        highTrump: card("hearts", "A")
      },
      {
        trumpSuit: "diamonds",
        seiJack: card("diamonds", "J"),
        uraJack: card("hearts", "J"),
        highTrump: card("diamonds", "A")
      }
    ];

    for (const testCase of cases) {
      expect(
        determineTrickWinner(
          [
            played("player-0", testCase.seiJack),
            played("player-1", testCase.uraJack),
            played("player-2", testCase.highTrump),
            played("player-3", card("clubs", "A")),
            played("player-4", joker())
          ],
          { trumpSuit: testCase.trumpSuit }
        )
      ).toBe("player-0");
    }
  });

  it("lets ura jack beat normal trump cards when sei jack and oruma are absent", () => {
    const cases: readonly Array<{ trumpSuit: Suit; uraJack: Card; highTrump: Card }> = [
      { trumpSuit: "spades", uraJack: card("clubs", "J"), highTrump: card("spades", "K") },
      { trumpSuit: "clubs", uraJack: card("spades", "J"), highTrump: card("clubs", "A") },
      { trumpSuit: "hearts", uraJack: card("diamonds", "J"), highTrump: card("hearts", "A") },
      { trumpSuit: "diamonds", uraJack: card("hearts", "J"), highTrump: card("diamonds", "A") }
    ];

    for (const testCase of cases) {
      expect(
        determineTrickWinner(
          [
            played("player-0", testCase.uraJack),
            played("player-1", testCase.highTrump),
            played("player-2", card("clubs", "A")),
            played("player-3", card("diamonds", "K"))
          ],
          { trumpSuit: testCase.trumpSuit }
        )
      ).toBe("player-0");
    }
  });

  it("keeps oruma and yoromeki above sei jack and ura jack", () => {
    expect(
      determineTrickWinner(
        [
          played("player-0", card("spades", "A")),
          played("player-1", card("hearts", "J")),
          played("player-2", card("diamonds", "J"))
        ],
        { trumpSuit: "hearts" }
      )
    ).toBe("player-0");
    expect(
      determineTrickWinner(
        [
          played("player-0", card("spades", "A")),
          played("player-1", card("hearts", "Q")),
          played("player-2", card("clubs", "J")),
          played("player-3", card("spades", "J"))
        ],
        { trumpSuit: "clubs" }
      )
    ).toBe("player-1");
  });

  it("lets sei jack and ura jack beat joker even when joker leads", () => {
    expect(
      determineTrickWinner(
        [
          played("player-0", joker()),
          played("player-1", card("hearts", "J")),
          played("player-2", card("diamonds", "J")),
          played("player-3", card("hearts", "A"))
        ],
        { trumpSuit: "hearts" }
      )
    ).toBe("player-1");
    expect(
      determineTrickWinner(
        [
          played("player-0", joker()),
          played("player-1", card("diamonds", "J")),
          played("player-2", card("hearts", "A")),
          played("player-3", card("clubs", "A"))
        ],
        { trumpSuit: "hearts" }
      )
    ).toBe("player-1");
  });

  it("treats hearts-Q as normal when oruma is absent even with special jacks", () => {
    expect(
      determineTrickWinner(
        [
          played("player-0", card("hearts", "Q")),
          played("player-1", card("clubs", "J")),
          played("player-2", card("spades", "J")),
          played("player-3", card("clubs", "A"))
        ],
        { trumpSuit: "clubs" }
      )
    ).toBe("player-1");
  });

  it("rejects special jack winner determination when trump is not set", () => {
    expect(() =>
      determineTrickWinner(
        [
          played("player-0", card("clubs", "J")),
          played("player-1", card("clubs", "A"))
        ],
        { trumpSuit: null }
      )
    ).toThrow(GameRuleError);
  });

  it("treats hearts-Q as a normal card when oruma is absent and clubs-A is not oruma", () => {
    expect(
      determineTrickWinner(
        [
          played("player-0", card("spades", "K")),
          played("player-1", card("hearts", "Q")),
          played("player-2", card("spades", "10"))
        ],
        { trumpSuit: "spades" }
      )
    ).toBe("player-0");
    expect(
      determineTrickWinner(
        [
          played("player-0", card("clubs", "A")),
          played("player-1", card("hearts", "Q")),
          played("player-2", card("spades", "K"))
        ],
        { trumpSuit: "spades" }
      )
    ).toBe("player-2");
  });

  it("lets same two win from the second trick when five standard cards share a suit and include the 2", () => {
    expect(
      determineTrickWinner(
        [
          played("player-0", card("hearts", "A")),
          played("player-1", card("hearts", "K")),
          played("player-2", card("hearts", "Q")),
          played("player-3", card("hearts", "7")),
          played("player-4", card("hearts", "2"))
        ],
        { trumpSuit: "spades" },
        { trickNumber: 2 }
      )
    ).toBe("player-4");
    expect(
      determineTrickWinner(
        [
          played("player-0", card("clubs", "A")),
          played("player-1", card("clubs", "K")),
          played("player-2", card("clubs", "Q")),
          played("player-3", card("clubs", "7")),
          played("player-4", card("clubs", "2"))
        ],
        { trumpSuit: "hearts" },
        { trickNumber: 3 }
      )
    ).toBe("player-4");
  });

  it("lets same two win in trump and spade suits when blocking special cards are absent", () => {
    expect(
      determineTrickWinner(
        [
          played("player-0", card("hearts", "A")),
          played("player-1", card("hearts", "K")),
          played("player-2", card("hearts", "10")),
          played("player-3", card("hearts", "7")),
          played("player-4", card("hearts", "2"))
        ],
        { trumpSuit: "hearts" },
        { trickNumber: 2 }
      )
    ).toBe("player-4");
    expect(
      determineTrickWinner(
        [
          played("player-0", card("spades", "K")),
          played("player-1", card("spades", "Q")),
          played("player-2", card("spades", "10")),
          played("player-3", card("spades", "7")),
          played("player-4", card("spades", "2"))
        ],
        { trumpSuit: "hearts" },
        { trickNumber: 2 }
      )
    ).toBe("player-4");
  });

  it("does not apply same two during the first trick", () => {
    expect(
      determineTrickWinner(
        [
          played("player-0", card("clubs", "A")),
          played("player-1", card("clubs", "K")),
          played("player-2", card("clubs", "Q")),
          played("player-3", card("clubs", "7")),
          played("player-4", card("clubs", "2"))
        ],
        { trumpSuit: "hearts" },
        { trickNumber: 1 }
      )
    ).toBe("player-0");
  });

  it("falls back when same two lacks one suit or the matching 2", () => {
    expect(
      determineTrickWinner(
        [
          played("player-0", card("diamonds", "A")),
          played("player-1", card("diamonds", "2")),
          played("player-2", card("diamonds", "K")),
          played("player-3", card("diamonds", "7"))
        ],
        { trumpSuit: "hearts" },
        { trickNumber: 2 }
      )
    ).toBe("player-0");
    expect(
      determineTrickWinner(
        [
          played("player-0", card("diamonds", "A")),
          played("player-1", card("diamonds", "K")),
          played("player-2", card("diamonds", "7")),
          played("player-3", card("diamonds", "2")),
          played("player-4", card("clubs", "3"))
        ],
        { trumpSuit: "hearts" },
        { trickNumber: 2 }
      )
    ).toBe("player-0");
    expect(
      determineTrickWinner(
        [
          played("player-0", card("clubs", "A")),
          played("player-1", card("clubs", "K")),
          played("player-2", card("clubs", "Q")),
          played("player-3", card("clubs", "7")),
          played("player-4", card("clubs", "3"))
        ],
        { trumpSuit: "hearts" },
        { trickNumber: 2 }
      )
    ).toBe("player-0");
  });

  it("does not apply same two when oruma, sei jack, ura jack, or joker is present", () => {
    expect(
      determineTrickWinner(
        [
          played("player-0", card("spades", "A")),
          played("player-1", card("spades", "K")),
          played("player-2", card("spades", "Q")),
          played("player-3", card("spades", "7")),
          played("player-4", card("spades", "2"))
        ],
        { trumpSuit: "hearts" },
        { trickNumber: 2 }
      )
    ).toBe("player-0");
    expect(
      determineTrickWinner(
        [
          played("player-0", card("hearts", "J")),
          played("player-1", card("hearts", "K")),
          played("player-2", card("hearts", "10")),
          played("player-3", card("hearts", "7")),
          played("player-4", card("hearts", "2"))
        ],
        { trumpSuit: "hearts" },
        { trickNumber: 2 }
      )
    ).toBe("player-0");
    expect(
      determineTrickWinner(
        [
          played("player-0", card("clubs", "J")),
          played("player-1", card("clubs", "K")),
          played("player-2", card("clubs", "10")),
          played("player-3", card("clubs", "7")),
          played("player-4", card("clubs", "2"))
        ],
        { trumpSuit: "spades" },
        { trickNumber: 2 }
      )
    ).toBe("player-0");
    expect(
      determineTrickWinner(
        [
          played("player-0", card("clubs", "A")),
          played("player-1", card("clubs", "K")),
          played("player-2", card("clubs", "7")),
          played("player-3", card("clubs", "2")),
          played("player-4", joker())
        ],
        { trumpSuit: "spades" },
        { trickNumber: 2 }
      )
    ).toBe("player-0");
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
    expect(state.adjutant).toBeNull();
    expect(state.result).toBeNull();
    expect(state.bidding).toMatchObject({
      starterPlayerId: "player-0",
      highestBid: null,
      consecutivePassCount: 0,
      history: []
    });
    expect(state.awardedPointCards).toEqual([]);
    expect(state.excludedCards).toEqual([]);
    expect(state.latestEvent).toBeNull();
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

    expect(actions).toHaveLength(41);
    expect(actions[0]).toEqual({ type: "pass", playerId: "player-0" });
    expect(actions.filter((action) => action.type === "bid")).toHaveLength(40);
    expect(actions).toContainEqual({
      type: "bid",
      playerId: "player-0",
      suit: "clubs",
      targetPointCards: 10
    });
    expect(actions).not.toContainEqual({
      type: "bid",
      playerId: "player-0",
      suit: "spades",
      targetPointCards: 9
    });
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
          targetPointCards: 9
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

    expect(state.phase).toBe("choosing-adjutant");
    expect(state.bidding).toBeNull();
    expect(state.contract).toEqual({
      napoleonPlayerId: "player-2",
      trumpSuit: "spades",
      targetPointCards: 13
    });
    expect(state.trumpSuit).toBe("spades");
    expect(state.currentPlayerId).toBe("player-2");
    expect(state.players.find((player) => player.id === "player-2")?.hand).toHaveLength(10);
    expect(
      state.players
        .filter((player) => player.id !== "player-2")
        .every((player) => player.hand.length === 10)
    ).toBe(true);
    expect(state.unusedCards).toHaveLength(3);
    expect(state.awardedPointCards).toHaveLength(0);
    expect(state.excludedCards).toHaveLength(0);
    expect(countKnownCards(state)).toBe(53);
    expect(createPlayerView(state, "player-2").adjutantChoiceRequirement).toEqual({
      jokerAllowed: true
    });
    expect(createPlayerView(state, "player-2").exchangeRequirement).toBeNull();
    expect(getLegalActions(state, "player-2")).toHaveLength(53);
    expect(getLegalActions(state, "player-2")).toContainEqual({
      type: "choose-adjutant",
      playerId: "player-2",
      cardId: "joker"
    });
  });

  it("creates a special spades-9 contract when everyone passes", () => {
    const state = Array.from({ length: 5 }).reduce<GameState>(
      (current) => applyAction(current, { type: "pass", playerId: current.currentPlayerId }),
      createInitialGame({ rng: noShuffle })
    );

    expect(state.phase).toBe("choosing-adjutant");
    expect(state.contract).toEqual({
      napoleonPlayerId: "player-0",
      trumpSuit: "spades",
      targetPointCards: 9
    });
    expect(state.trumpSuit).toBe("spades");
    expect(state.currentPlayerId).toBe("player-0");
    expect(state.players[0].hand).toHaveLength(10);
    expect(state.unusedCards).toHaveLength(3);
  });

  it("resolves discarded buried cards and starts play", () => {
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
    expect([
      ...next.awardedPointCards.flatMap((award) => award.cards.map((card) => card.id)),
      ...next.excludedCards.map((card) => card.id)
    ].sort()).toEqual([...discardIds].sort());
    expect(next.latestEvent).toMatchObject({
      type: "buried-cards-resolved",
      napoleonPlayerId: "player-0"
    });
    expect(next.currentTrick).toEqual([]);
    expect(next.trickNumber).toBe(1);
    expect(next.isTrickComplete).toBe(false);
    expect(next.isGameOver).toBe(false);
    expect(next.adjutant).toEqual(exchanging.adjutant);
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
    expect(next.excludedCards.some(isJokerCard)).toBe(true);
  });

  it("exposes the buried-card resolution event without non-point card contents", () => {
    const exchanging = createAllPassExchangeState();
    const pointCard = exchanging.players[0].hand.find(isPointCard);
    const nonPointStandardCard = exchanging.players[0].hand.find(
      (candidate) => isStandardCard(candidate) && !isPointCard(candidate)
    );
    const jokerCard = exchanging.players[0].hand.find(isJokerCard);

    expect(pointCard).toBeDefined();
    expect(nonPointStandardCard).toBeDefined();
    expect(jokerCard).toBeDefined();
    if (pointCard === undefined || nonPointStandardCard === undefined || jokerCard === undefined) {
      throw new Error("Expected point, non-point, and joker cards in the exchange hand.");
    }

    const discardIds = [pointCard.id, nonPointStandardCard.id, jokerCard.id];
    const next = applyAction(exchanging, {
      type: "discard-cards",
      playerId: "player-0",
      cardIds: discardIds
    });
    const view = createPlayerView(next, "player-1");

    expect(view.latestEvent).toEqual({
      type: "buried-cards-resolved",
      napoleonPlayerId: "player-0",
      awardedPointCards: [pointCard],
      hiddenNonPointCardCount: 2
    });
    expect(JSON.stringify(view.latestEvent)).not.toContain(nonPointStandardCard.id);
    expect(JSON.stringify(view.latestEvent)).not.toContain(jokerCard.id);
  });

  it("lets Napoleon choose a standard adjutant card before play starts", () => {
    const choosing = createAllPassAdjutantChoiceState();
    const opponentCard = choosing.players[1].hand.find(isStandardCard);

    expect(opponentCard).toBeDefined();
    if (opponentCard === undefined) {
      throw new Error("Expected player-1 to have a standard card.");
    }

    const exchanging = chooseAdjutant(choosing, opponentCard.id);
    const napoleonView = createPlayerView(exchanging, "player-0");
    const opponentView = createPlayerView(exchanging, "player-1");

    expect(exchanging.phase).toBe("exchanging");
    expect(exchanging.currentPlayerId).toBe("player-0");
    expect(exchanging.players[0].hand).toHaveLength(13);
    expect(exchanging.unusedCards).toHaveLength(0);
    expect(exchanging.adjutant).toEqual({
      calledCardId: opponentCard.id,
      playerId: "player-1",
      revealed: false
    });
    expect(getLegalActions(exchanging, "player-0")).toEqual([]);
    expect(napoleonView.adjutant).toEqual({
      calledCardId: opponentCard.id,
      revealedPlayerId: null
    });
    expect(opponentView.adjutant).toEqual(napoleonView.adjutant);
    expect(napoleonView.adjutantChoiceRequirement).toBeNull();
  });

  it("treats self-held and buried adjutant cards as absent adjutants", () => {
    const choosing = createAllPassAdjutantChoiceState();
    const selfCard = choosing.players[0].hand.find(isStandardCard);
    const resolvedBuriedCard = choosing.unusedCards.find(isStandardCard);

    expect(selfCard).toBeDefined();
    expect(resolvedBuriedCard).toBeDefined();
    if (selfCard === undefined || resolvedBuriedCard === undefined) {
      throw new Error("Expected standard cards in hand and resolved buried cards.");
    }

    expect(chooseAdjutant(choosing, selfCard.id).adjutant).toEqual({
      calledCardId: selfCard.id,
      playerId: null,
      revealed: false
    });
    expect(chooseAdjutant(choosing, resolvedBuriedCard.id).adjutant).toEqual({
      calledCardId: resolvedBuriedCard.id,
      playerId: null,
      revealed: false
    });
  });

  it("lets Napoleon choose the joker as an adjutant card", () => {
    const choosing = createStateWithHands({
      phase: "choosing-adjutant",
      currentPlayerId: "player-0",
      hands: [
        [
          card("spades", "2"),
          card("spades", "3"),
          card("spades", "4"),
          card("spades", "5"),
          card("spades", "6"),
          card("spades", "7"),
          card("spades", "8"),
          card("spades", "9"),
          card("spades", "10"),
          card("spades", "J")
        ],
        [joker()],
        [card("clubs", "3")],
        [card("diamonds", "4")],
        [card("hearts", "5")]
      ],
      unusedCards: [card("clubs", "2"), card("clubs", "4"), card("diamonds", "3")]
    });

    const exchanging = chooseAdjutant(choosing, "joker");

    expect(exchanging.phase).toBe("exchanging");
    expect(exchanging.adjutant).toEqual({
      calledCardId: "joker",
      playerId: "player-1",
      revealed: false
    });
    expect(createPlayerView(exchanging, "player-0").adjutant).toEqual({
      calledCardId: "joker",
      revealedPlayerId: null
    });
  });

  it("treats self-held and buried joker adjutant choices as absent adjutants", () => {
    const selfHeld = createStateWithHands({
      phase: "choosing-adjutant",
      currentPlayerId: "player-0",
      hands: [
        [
          joker(),
          card("spades", "2"),
          card("spades", "3"),
          card("spades", "4"),
          card("spades", "5"),
          card("spades", "6"),
          card("spades", "7"),
          card("spades", "8"),
          card("spades", "9"),
          card("spades", "10")
        ],
        [card("clubs", "A")],
        [card("clubs", "3")],
        [card("diamonds", "4")],
        [card("hearts", "5")]
      ],
      unusedCards: [card("clubs", "2"), card("clubs", "4"), card("diamonds", "3")]
    });
    const buried = createStateWithHands({
      phase: "choosing-adjutant",
      currentPlayerId: "player-0",
      hands: [
        [
          card("spades", "2"),
          card("spades", "3"),
          card("spades", "4"),
          card("spades", "5"),
          card("spades", "6"),
          card("spades", "7"),
          card("spades", "8"),
          card("spades", "9"),
          card("spades", "10"),
          card("spades", "J")
        ],
        [card("clubs", "3")],
        [card("diamonds", "4")],
        [card("hearts", "5")],
        [card("clubs", "A")]
      ],
      unusedCards: [joker(), card("clubs", "2"), card("clubs", "4")]
    });

    expect(chooseAdjutant(selfHeld, "joker").adjutant).toEqual({
      calledCardId: "joker",
      playerId: null,
      revealed: false
    });
    expect(chooseAdjutant(buried, "joker").adjutant).toEqual({
      calledCardId: "joker",
      playerId: null,
      revealed: false
    });
  });

  it("allows oruma, yoromeki, sei jack, and ura jack to be chosen as adjutant cards", () => {
    const choosing = createAllPassAdjutantChoiceState();

    expect(chooseAdjutant(choosing, orumaCardId).adjutant?.calledCardId).toBe(orumaCardId);
    expect(chooseAdjutant(choosing, yoromekiCardId).adjutant?.calledCardId).toBe(yoromekiCardId);
    expect(chooseAdjutant(choosing, "spades-J").adjutant?.calledCardId).toBe("spades-J");
    expect(chooseAdjutant(choosing, "clubs-J").adjutant?.calledCardId).toBe("clubs-J");
  });

  it("rejects invalid adjutant choices and phase violations", () => {
    const choosing = createAllPassAdjutantChoiceState();
    const exchanging = chooseAdjutant(choosing, "spades-A");

    expectRuleError(
      () =>
        applyAction(choosing, {
          type: "choose-adjutant",
          playerId: "player-0",
          cardId: "spades-1"
        }),
      "INVALID_ADJUTANT_CARD"
    );
    expectRuleError(
      () =>
        applyAction(choosing, {
          type: "choose-adjutant",
          playerId: "player-1",
          cardId: "spades-A"
        }),
      "NOT_NAPOLEON"
    );
    expectRuleError(
      () =>
        applyAction(exchanging, {
          type: "choose-adjutant",
          playerId: "player-0",
          cardId: "spades-K"
        }),
      "INVALID_ACTION_FOR_PHASE"
    );
    expectRuleError(
      () => applyAction(choosing, { type: "pass", playerId: "player-0" }),
      "INVALID_ACTION_FOR_PHASE"
    );
    expectRuleError(
      () =>
        applyAction(choosing, {
          type: "bid",
          playerId: "player-0",
          suit: "spades",
          targetPointCards: 13
        }),
      "INVALID_ACTION_FOR_PHASE"
    );
  });

  it("reveals the adjutant only when the called card is played by its owner", () => {
    const state = createStateWithHands({
      hands: [
        [card("clubs", "2")],
        [card("hearts", "A")],
        [card("clubs", "3")],
        [card("clubs", "4")],
        [card("clubs", "5")]
      ],
      currentPlayerId: "player-1",
      adjutant: {
        calledCardId: "hearts-A",
        playerId: "player-1",
        revealed: false
      }
    });
    const next = applyAction(state, {
      type: "play-card",
      playerId: "player-1",
      cardId: "hearts-A"
    });

    expect(next.adjutant).toEqual({
      calledCardId: "hearts-A",
      playerId: "player-1",
      revealed: true
    });
    expect(createPlayerView(next, "player-0").adjutant).toEqual({
      calledCardId: "hearts-A",
      revealedPlayerId: "player-1"
    });
  });

  it("reveals a joker adjutant only when the owner plays the joker", () => {
    const state = createStateWithHands({
      hands: [
        [card("clubs", "2")],
        [joker()],
        [card("clubs", "3")],
        [card("clubs", "4")],
        [card("clubs", "5")]
      ],
      currentPlayerId: "player-1",
      adjutant: {
        calledCardId: "joker",
        playerId: "player-1",
        revealed: false
      }
    });
    const next = applyAction(state, {
      type: "play-card",
      playerId: "player-1",
      cardId: "joker"
    });

    expect(next.adjutant).toEqual({
      calledCardId: "joker",
      playerId: "player-1",
      revealed: true
    });
    expect(createPlayerView(next, "player-0").adjutant).toEqual({
      calledCardId: "joker",
      revealedPlayerId: "player-1"
    });
  });

  it("does not reveal a joker adjutant when the joker has no adjutant owner", () => {
    const state = createStateWithHands({
      hands: [
        [card("clubs", "2")],
        [joker()],
        [card("clubs", "3")],
        [card("clubs", "4")],
        [card("clubs", "5")]
      ],
      currentPlayerId: "player-1",
      adjutant: {
        calledCardId: "joker",
        playerId: null,
        revealed: false
      }
    });
    const next = applyAction(state, {
      type: "play-card",
      playerId: "player-1",
      cardId: "joker"
    });

    expect(next.adjutant).toEqual({
      calledCardId: "joker",
      playerId: null,
      revealed: false
    });
    expect(createPlayerView(next, "player-0").adjutant).toEqual({
      calledCardId: "joker",
      revealedPlayerId: null
    });
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
    const choosing = createAllPassAdjutantChoiceState();
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
    expectRuleError(
      () =>
        applyAction(exchanging, {
          type: "choose-adjutant",
          playerId: "player-0",
          cardId: "spades-A"
        }),
      "INVALID_ACTION_FOR_PHASE"
    );
    expectRuleError(
      () =>
        applyAction(choosing, {
          type: "play-card",
          playerId: "player-0",
          cardId: choosing.players[0].hand[0].id
        }),
      "INVALID_ACTION_FOR_PHASE"
    );
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

  it("uses oruma and yoromeki lead suits as their normal suits", () => {
    expect(getLeadSuit([played("player-0", card("spades", "A"))], { trumpSuit: "hearts" })).toBe(
      "spades"
    );
    expect(getLeadSuit([played("player-0", card("hearts", "Q"))], { trumpSuit: "spades" })).toBe(
      "hearts"
    );
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

  it("keeps oruma and yoromeki under normal follow-suit obligations", () => {
    expect(
      getPlayableCards(
        [card("spades", "A"), card("hearts", "5")],
        [played("player-0", card("spades", "2"))],
        { trumpSuit: "clubs" }
      ).map((candidate) => candidate.id)
    ).toEqual(["spades-A"]);
    expect(
      getPlayableCards(
        [card("hearts", "Q"), card("clubs", "5")],
        [played("player-0", card("hearts", "2"))],
        { trumpSuit: "spades" }
      ).map((candidate) => candidate.id)
    ).toEqual(["hearts-Q"]);
    expect(
      getPlayableCards(
        [card("spades", "A"), card("hearts", "5")],
        [played("player-0", card("hearts", "2"))],
        { trumpSuit: "clubs" }
      ).map((candidate) => candidate.id)
    ).toEqual(["hearts-5"]);
    expect(
      getPlayableCards(
        [card("hearts", "Q"), card("clubs", "5")],
        [played("player-0", card("clubs", "2"))],
        { trumpSuit: "spades" }
      ).map((candidate) => candidate.id)
    ).toEqual(["clubs-5"]);
  });

  it("keeps sei jack and ura jack under normal follow-suit obligations", () => {
    expect(
      getPlayableCards(
        [card("spades", "J"), card("hearts", "5")],
        [played("player-0", card("spades", "2"))],
        { trumpSuit: "spades" }
      ).map((candidate) => candidate.id)
    ).toEqual(["spades-J"]);
    expect(
      getPlayableCards(
        [card("clubs", "J"), card("hearts", "5")],
        [played("player-0", card("clubs", "2"))],
        { trumpSuit: "spades" }
      ).map((candidate) => candidate.id)
    ).toEqual(["clubs-J"]);
    expect(
      getPlayableCards(
        [card("clubs", "J"), card("spades", "5")],
        [played("player-0", card("spades", "2"))],
        { trumpSuit: "spades" }
      ).map((candidate) => candidate.id)
    ).toEqual(["spades-5"]);
    expect(
      getPlayableCards(
        [card("hearts", "J"), card("clubs", "5")],
        [played("player-0", card("clubs", "2"))],
        { trumpSuit: "hearts" }
      ).map((candidate) => candidate.id)
    ).toEqual(["clubs-5"]);
    expect(
      getPlayableCards(
        [card("diamonds", "J"), card("clubs", "5")],
        [played("player-0", card("clubs", "2"))],
        { trumpSuit: "hearts" }
      ).map((candidate) => candidate.id)
    ).toEqual(["clubs-5"]);
  });

  it("uses sei jack and ura jack source suits as lead suits", () => {
    expect(getLeadSuit([played("player-0", card("spades", "J"))], { trumpSuit: "spades" })).toBe(
      "spades"
    );
    expect(getLeadSuit([played("player-0", card("clubs", "J"))], { trumpSuit: "spades" })).toBe(
      "clubs"
    );
  });

  it("keeps same two candidate cards under normal follow-suit and lead-suit rules", () => {
    expect(
      getPlayableCards(
        [card("clubs", "2"), card("hearts", "5")],
        [played("player-0", card("hearts", "A"))],
        { trumpSuit: "spades" }
      ).map((candidate) => candidate.id)
    ).toEqual(["hearts-5"]);
    expect(
      getPlayableCards(
        [card("clubs", "2"), card("hearts", "2")],
        [played("player-0", card("hearts", "A"))],
        { trumpSuit: "spades" }
      ).map((candidate) => candidate.id)
    ).toEqual(["hearts-2"]);
    expect(getLeadSuit([played("player-0", card("diamonds", "2"))], { trumpSuit: "spades" })).toBe(
      "diamonds"
    );
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

  it("rejects directly playing sei jack or ura jack when another suit must be followed", () => {
    const seiJackState = createStateWithHands({
      hands: [
        [card("clubs", "2")],
        [card("hearts", "J"), card("clubs", "5")],
        [],
        [],
        []
      ],
      trumpSuit: "hearts"
    });
    const uraJackState = createStateWithHands({
      hands: [
        [card("clubs", "2")],
        [card("diamonds", "J"), card("clubs", "5")],
        [],
        [],
        []
      ],
      trumpSuit: "hearts"
    });
    const ledForSeiJack = applyAction(seiJackState, {
      type: "play-card",
      playerId: "player-0",
      cardId: "clubs-2"
    });
    const ledForUraJack = applyAction(uraJackState, {
      type: "play-card",
      playerId: "player-0",
      cardId: "clubs-2"
    });

    expectRuleError(
      () =>
        applyAction(ledForSeiJack, {
          type: "play-card",
          playerId: "player-1",
          cardId: "hearts-J"
        }),
      "MUST_FOLLOW_SUIT"
    );
    expectRuleError(
      () =>
        applyAction(ledForUraJack, {
          type: "play-card",
          playerId: "player-1",
          cardId: "diamonds-J"
        }),
      "MUST_FOLLOW_SUIT"
    );
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
        { playerId: "player-2", card: card("spades", "K") },
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
        [card("spades", "K"), card("clubs", "3")],
        [card("hearts", "3"), card("diamonds", "4")],
        [card("clubs", "Q"), card("spades", "5")]
      ],
      trumpSuit: "diamonds"
    });
    const completed = [
      { playerId: "player-0", cardId: "hearts-7" },
      { playerId: "player-1", cardId: "hearts-K" },
      { playerId: "player-2", cardId: "spades-K" },
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

  it("sets the same two winner as the next lead player and records the completed trick", () => {
    const state = createStateWithHands({
      hands: [
        [card("hearts", "A"), card("clubs", "6")],
        [card("hearts", "K"), card("clubs", "2")],
        [card("hearts", "Q"), card("clubs", "3")],
        [card("hearts", "7"), card("diamonds", "4")],
        [card("hearts", "2"), card("spades", "5")]
      ],
      trumpSuit: "spades",
      trickNumber: 2
    });
    const completed = [
      { playerId: "player-0", cardId: "hearts-A" },
      { playerId: "player-1", cardId: "hearts-K" },
      { playerId: "player-2", cardId: "hearts-Q" },
      { playerId: "player-3", cardId: "hearts-7" },
      { playerId: "player-4", cardId: "hearts-2" }
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

    expect(completed.currentPlayerId).toBe("player-4");
    expect(completed.completedTricks[0].winnerId).toBe("player-4");
    expect(completed.completedTricks[0].cards).toHaveLength(5);
    expect(
      completed.completedTricks[0].cards.filter((playedCard) => isPointCard(playedCard.card))
    ).toHaveLength(3);
    expect(next.currentPlayerId).toBe("player-4");
    expect(next.trickNumber).toBe(3);
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
          { playerId: "player-1", card: card("spades", "K") },
          { playerId: "player-2", card: card("clubs", "A") },
          { playerId: "player-3", card: card("diamonds", "A") },
          { playerId: "player-4", card: card("spades", "Q") }
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
          { playerId: "player-2", card: card("spades", "K") },
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

  it("calculates point cards by the trick winner's team and uses awarded buried point cards", () => {
    const state = createStateWithHands({
      hands: [[], [], [], [], []],
      phase: "finished",
      isGameOver: true,
      completedTricks: [
        createCompletedTrick(1, "player-0", [
          card("spades", "A"),
          card("spades", "K"),
          card("spades", "Q"),
          card("spades", "J"),
          card("spades", "10")
        ]),
        createCompletedTrick(2, "player-1", [
          card("hearts", "A"),
          card("hearts", "K"),
          card("hearts", "Q"),
          card("hearts", "J"),
          card("clubs", "2")
        ]),
        createCompletedTrick(3, "player-2", [
          card("diamonds", "A"),
          card("diamonds", "K"),
          card("diamonds", "Q"),
          card("diamonds", "J"),
          card("diamonds", "10")
        ]),
        createCompletedTrick(4, "player-3", [
          card("clubs", "K"),
          card("clubs", "Q"),
          card("clubs", "J"),
          card("clubs", "10"),
          card("clubs", "3")
        ]),
        ...createCompletedTricksFromPointCounts([0, 0, 0, 0, 0, 0], 5)
      ],
      awardedPointCards: [
        { playerId: "player-0", cards: [card("clubs", "A"), card("hearts", "10")] }
      ],
      excludedCards: [joker()],
      adjutant: {
        calledCardId: "hearts-A",
        playerId: "player-1",
        revealed: false
      },
      contract: {
        napoleonPlayerId: "player-0",
        trumpSuit: "spades",
        targetPointCards: 11
      }
    });

    expect(calculateGameResult(state)).toEqual({
      winner: "napoleon-team",
      napoleonTeamPointCards: 11,
      alliancePointCards: 9,
      targetPointCards: 11,
      napoleonPlayerId: "player-0",
      adjutantPlayerId: "player-1"
    });
  });

  it("uses only Napoleon as the Napoleon team when no adjutant exists", () => {
    const state = createResultStateWithNapoleonPointCards(15, 15, {
      calledCardId: "spades-A",
      playerId: null,
      revealed: false
    });

    expect(calculateGameResult(state)).toMatchObject({
      winner: "napoleon-team",
      napoleonTeamPointCards: 15,
      alliancePointCards: 5,
      adjutantPlayerId: null
    });
  });

  it("includes a joker adjutant player in Napoleon team scoring", () => {
    const state = createResultStateWithNapoleonPointCards(15, 15, {
      calledCardId: "joker",
      playerId: "player-1",
      revealed: false
    });

    expect(calculateGameResult(state)).toMatchObject({
      winner: "napoleon-team",
      napoleonTeamPointCards: 15,
      alliancePointCards: 5,
      adjutantPlayerId: "player-1"
    });
  });

  it("compares Napoleon team point cards against the contract target", () => {
    expect(createResultStateWithNapoleonPointCards(15, 15).result).toBeNull();
    expect(calculateGameResult(createResultStateWithNapoleonPointCards(15, 15)).winner).toBe(
      "napoleon-team"
    );
    expect(calculateGameResult(createResultStateWithNapoleonPointCards(14, 15)).winner).toBe(
      "alliance"
    );
  });

  it("rejects result calculation when the result state is inconsistent", () => {
    const invalid = createResultStateWithNapoleonPointCards(14, 15);

    expectRuleError(
      () =>
        calculateGameResult({
          ...invalid,
          completedTricks: invalid.completedTricks.slice(0, 9)
        }),
      "INVALID_RESULT_STATE"
    );
    expectRuleError(
      () =>
        calculateGameResult({
          ...invalid,
          completedTricks: invalid.completedTricks.map((trick, index) =>
            index === 0
              ? {
                  ...trick,
                  cards: trick.cards.map((playedCard, cardIndex) =>
                    cardIndex === 0
                      ? { ...playedCard, card: card("spades", "2") }
                      : playedCard
                  )
                }
              : trick
          )
        }),
      "POINT_CARD_COUNT_MISMATCH"
    );
  });

  it("sets phase to finished when the final trick completes", () => {
    const state = createStateWithHands({
      hands: [
        [card("spades", "A")],
        [card("spades", "K")],
        [card("spades", "Q")],
        [card("spades", "J")],
        [card("spades", "10")]
      ],
      completedTricks: createCompletedTricksFromPointCounts([
        5,
        5,
        5,
        0,
        0,
        0,
        0,
        0,
        0
      ]),
      trickNumber: 10,
      excludedCards: [card("clubs", "2"), card("clubs", "3"), joker()],
      adjutant: {
        calledCardId: "hearts-A",
        playerId: "player-1",
        revealed: false
      }
    });
    const completed = Array.from({ length: 5 }).reduce<GameState>(
      (current) => playCurrentPlayerFirstCard(current),
      state
    );

    expect(completed.isGameOver).toBe(true);
    expect(completed.phase).toBe("finished");
    expect(completed.result).toMatchObject({
      napoleonTeamPointCards: 20,
      alliancePointCards: 0
    });
    expect(completed.completedTricks).toHaveLength(10);
    expect(completed.players.every((player) => player.hand.length === 0)).toBe(true);
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
    expect(view.players[1].capturedPointCards).toEqual([]);
  });

  it("derives captured point cards by completed trick winner in createPlayerView", () => {
    const state = createStateWithHands({
      hands: [[], [], [], [], []],
      completedTricks: [
        createCompletedTrick(1, "player-1", [
          card("hearts", "A"),
          card("hearts", "K"),
          card("hearts", "Q"),
          card("clubs", "2"),
          joker()
        ]),
        createCompletedTrick(2, "player-1", [
          card("diamonds", "10"),
          card("clubs", "3"),
          card("clubs", "4"),
          card("clubs", "5"),
          card("clubs", "6")
        ]),
        createCompletedTrick(3, "player-3", [
          card("spades", "J"),
          card("diamonds", "2"),
          card("diamonds", "3"),
          card("diamonds", "4"),
          card("diamonds", "5")
        ])
      ],
      awardedPointCards: [{ playerId: "player-0", cards: [card("clubs", "A")] }],
      excludedCards: [card("spades", "2"), joker()]
    });
    const view = createPlayerView(state, "player-0");

    expect(
      view.players.find((player) => player.id === "player-0")?.capturedPointCards.map((card) => card.id)
    ).toEqual(["clubs-A"]);
    expect(
      view.players.find((player) => player.id === "player-1")?.capturedPointCards.map((card) => card.id)
    ).toEqual(["hearts-A", "hearts-K", "hearts-Q", "diamonds-10"]);
    expect(
      view.players.find((player) => player.id === "player-3")?.capturedPointCards.map((card) => card.id)
    ).toEqual(["spades-J"]);
    expect(
      view.players.find((player) => player.id === "player-2")?.capturedPointCards
    ).toEqual([]);
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
    const state = createAllPassAdjutantChoiceState();
    const view = createPlayerView(state, "player-0");

    expect(view.phase).toBe("choosing-adjutant");
    expect(view.contract).toEqual({
      napoleonPlayerId: "player-0",
      trumpSuit: "spades",
      targetPointCards: 9
    });
    expect(view.bidding).toBeNull();
    expect(view.trumpSuit).toBe("spades");
    expect(view.exchangeRequirement).toBeNull();
    expect(view.adjutant).toBeNull();
    expect(view.adjutantChoiceRequirement).toEqual({ jokerAllowed: true });
    expect(view.players[0].hand).toHaveLength(10);
  });

  it("keeps exchange information private for non-Napoleon views", () => {
    const state = createAllPassAdjutantChoiceState();
    const napoleonView = createPlayerView(state, "player-0");
    const opponentView = createPlayerView(state, "player-1");

    expect(napoleonView.phase).toBe("choosing-adjutant");
    expect(napoleonView.players[0].hand).toHaveLength(10);
    expect(napoleonView.exchangeRequirement).toBeNull();
    expect(napoleonView.adjutantChoiceRequirement).toEqual({ jokerAllowed: true });
    expect(opponentView.phase).toBe("choosing-adjutant");
    expect(opponentView.players[0].handCount).toBe(10);
    expect(opponentView.players[0].hand).toBeUndefined();
    expect(opponentView.players[1].hand).toHaveLength(10);
    expect(opponentView.exchangeRequirement).toBeNull();
    expect(opponentView.adjutantChoiceRequirement).toBeNull();
    expect(opponentView.latestEvent).toBeNull();
    expect(opponentView.result).toBeNull();
  });

  it("exposes adjutant choice requirement only to Napoleon during adjutant choice", () => {
    const choosing = createAllPassAdjutantChoiceState();
    const napoleonView = createPlayerView(choosing, "player-0");
    const opponentView = createPlayerView(choosing, "player-1");

    expect(napoleonView.phase).toBe("choosing-adjutant");
    expect(napoleonView.exchangeRequirement).toBeNull();
    expect(napoleonView.adjutantChoiceRequirement).toEqual({ jokerAllowed: true });
    expect(napoleonView.players[0].hand).toHaveLength(10);
    expect(opponentView.phase).toBe("choosing-adjutant");
    expect(opponentView.players[0].handCount).toBe(10);
    expect(opponentView.players[0].hand).toBeUndefined();
    expect(opponentView.adjutantChoiceRequirement).toBeNull();
    expect(opponentView.adjutant).toBeNull();
    expect(opponentView.latestEvent).toBeNull();
    expect(JSON.stringify(opponentView.latestEvent)).not.toContain("joker");
  });

  it("clears exchange and adjutant requirements after adjutant choice completes", () => {
    const choosing = createAllPassAdjutantChoiceState();
    const next = chooseAdjutant(choosing, "spades-A");
    const view = createPlayerView(next, "player-1");

    expect(view.phase).toBe("exchanging");
    expect(view.exchangeRequirement).toBeNull();
    expect(view.adjutantChoiceRequirement).toBeNull();
    expect(view.players[0].handCount).toBe(13);
    expect(view.adjutant).toEqual({
      calledCardId: "spades-A",
      revealedPlayerId: null
    });
    expect(view.latestEvent).toBeNull();
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
    expect(view.specialCards).toEqual({
      orumaCardId,
      yoromekiCardId,
      seiJackCardId: null,
      uraJackCardId: null
    });
  });

  it("exposes the same trump suit to every player view without leaking hands", () => {
    const state = createPlayingInitialGame();

    for (const player of state.players) {
      const view = createPlayerView(state, player.id);

      expect(view.trumpSuit).toBe(state.trumpSuit);
      expect(view.specialCards).toEqual({
        orumaCardId: "spades-A",
        yoromekiCardId: "hearts-Q",
        seiJackCardId: "spades-J",
        uraJackCardId: "clubs-J"
      });
      expect(view.players.find((viewPlayer) => viewPlayer.id === player.id)?.hand).toHaveLength(10);
      expect(
        view.players
          .filter((viewPlayer) => viewPlayer.id !== player.id)
          .every((viewPlayer) => viewPlayer.hand === undefined && viewPlayer.handCount === 10)
      ).toBe(true);
      expect(view.legalActions.every((action) => action.playerId === player.id)).toBe(true);
    }
  });

  it("exposes fixed oruma and yoromeki ids with trump-derived sei jack and ura jack ids", () => {
    const cases: readonly Array<{ trumpSuit: Suit; seiJackCardId: string; uraJackCardId: string }> = [
      { trumpSuit: "spades", seiJackCardId: "spades-J", uraJackCardId: "clubs-J" },
      { trumpSuit: "clubs", seiJackCardId: "clubs-J", uraJackCardId: "spades-J" },
      { trumpSuit: "hearts", seiJackCardId: "hearts-J", uraJackCardId: "diamonds-J" },
      { trumpSuit: "diamonds", seiJackCardId: "diamonds-J", uraJackCardId: "hearts-J" }
    ];

    for (const testCase of cases) {
      const view = createPlayerView(createStateWithHands({
        hands: [[card("clubs", "2")], [], [], [], []],
        trumpSuit: testCase.trumpSuit,
        contract: {
          napoleonPlayerId: "player-0",
          trumpSuit: testCase.trumpSuit,
          targetPointCards: 13
        }
      }), "player-0");

      expect(view.specialCards).toEqual({
        orumaCardId,
        yoromekiCardId,
        seiJackCardId: testCase.seiJackCardId,
        uraJackCardId: testCase.uraJackCardId
      });
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
  adjutant?: GameState["adjutant"];
  bidding?: GameState["bidding"];
  awardedPointCards?: GameState["awardedPointCards"];
  excludedCards?: GameState["excludedCards"];
  unusedCards?: GameState["unusedCards"];
  latestEvent?: GameState["latestEvent"];
  result?: GameState["result"];
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
    adjutant: options.adjutant ?? null,
    bidding: options.bidding ?? null,
    awardedPointCards: options.awardedPointCards ?? [],
    excludedCards: options.excludedCards ?? [],
    latestEvent: options.latestEvent ?? null,
    result: options.result ?? null,
    trickNumber: options.trickNumber ?? 1,
    isTrickComplete: options.isTrickComplete ?? false,
    isGameOver: options.isGameOver ?? false,
    unusedCards: options.unusedCards ?? []
  };
}

function hasOwn(value: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

function createAllPassAdjutantChoiceState(): GameState {
  return Array.from({ length: 5 }).reduce<GameState>(
    (current) => applyAction(current, { type: "pass", playerId: current.currentPlayerId }),
    createInitialGame({ rng: noShuffle })
  );
}

function createAllPassExchangeState(cardId = "spades-A"): GameState {
  return chooseAdjutant(createAllPassAdjutantChoiceState(), cardId);
}

function chooseAdjutant(state: GameState, cardId = "spades-A"): GameState {
  return applyAction(state, {
    type: "choose-adjutant",
    playerId: state.contract?.napoleonPlayerId ?? state.currentPlayerId,
    cardId
  });
}

function createResultStateWithNapoleonPointCards(
  napoleonPointCards: number,
  targetPointCards: number,
  adjutant: NonNullable<GameState["adjutant"]> = {
    calledCardId: "hearts-A",
    playerId: "player-1",
    revealed: false
  }
): GameState {
  const completedTricks = createCompletedTricksForTeamPointSplit(napoleonPointCards);

  return createStateWithHands({
    hands: [[], [], [], [], []],
    phase: "finished",
    isGameOver: true,
    completedTricks,
    excludedCards: [card("clubs", "2"), card("clubs", "3"), joker()],
    contract: {
      napoleonPlayerId: "player-0",
      trumpSuit: "spades",
      targetPointCards
    },
    adjutant
  });
}

function createCompletedTricksForTeamPointSplit(
  napoleonPointCards: number
): GameState["completedTricks"] {
  const trickPointCounts: number[] = [];
  let remainingNapoleonPoints = napoleonPointCards;
  let remainingAlliancePoints = 20 - napoleonPointCards;

  while (remainingNapoleonPoints > 0) {
    const count = Math.min(5, remainingNapoleonPoints);
    trickPointCounts.push(count);
    remainingNapoleonPoints -= count;
  }

  while (remainingAlliancePoints > 0) {
    const count = Math.min(5, remainingAlliancePoints);
    trickPointCounts.push(-count);
    remainingAlliancePoints -= count;
  }

  while (trickPointCounts.length < 10) {
    trickPointCounts.push(0);
  }

  return trickPointCounts.map((pointCount, index) =>
    createCompletedTrickWithPointCount(
      index + 1,
      pointCount >= 0 ? "player-0" : "player-2",
      Math.abs(pointCount),
      index * 5
    )
  );
}

function createCompletedTricksFromPointCounts(
  pointCounts: readonly number[],
  firstTrickNumber = 1
): GameState["completedTricks"] {
  return pointCounts.map((pointCount, index) =>
    createCompletedTrickWithPointCount(
      firstTrickNumber + index,
      "player-0",
      pointCount,
      index * 5
    )
  );
}

function createCompletedTrickWithPointCount(
  trickNumber: number,
  winnerId: string,
  pointCount: number,
  cardOffset: number
): GameState["completedTricks"][number] {
  const pointCards = createDeck().filter(isPointCard);
  const nonPointCards = createDeck().filter((candidate) => !isPointCard(candidate));
  const cards = [
    ...Array.from(
      { length: pointCount },
      (_, index) => pointCards[(cardOffset + index) % pointCards.length]
    ),
    ...Array.from(
      { length: 5 - pointCount },
      (_, index) => nonPointCards[(cardOffset + index) % nonPointCards.length]
    )
  ];

  return createCompletedTrick(trickNumber, winnerId, cards);
}

function createCompletedTrick(
  trickNumber: number,
  winnerId: string,
  cards: readonly Card[]
): GameState["completedTricks"][number] {
  const playerIds = ["player-0", "player-1", "player-2", "player-3", "player-4"];

  return {
    trickNumber,
    winnerId,
    cards: playerIds.map((playerId, index) => ({
      playerId,
      card: cards[index]
    }))
  };
}

function countKnownCards(state: GameState): number {
  return [
    ...state.players.flatMap((player) => player.hand),
    ...state.awardedPointCards.flatMap((award) => award.cards),
    ...state.excludedCards,
    ...state.unusedCards,
    ...state.currentTrick.map((played) => played.card),
    ...state.completedTricks.flatMap((trick) => trick.cards.map((played) => played.card))
  ].length;
}
