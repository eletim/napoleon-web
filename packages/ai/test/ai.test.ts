import { describe, expect, it } from "vitest";
import {
  applyAction,
  createDeck,
  createInitialGame,
  createPlayerView,
  getLegalActions,
  isPointCard,
  isStandardCard,
  type Card,
  type GameAction,
  type GameState,
  type PlayerView,
  type Rank,
  type StandardCard,
  type Suit
} from "@napoleon/game-core";
import {
  adjustTeamWinProbability,
  aiRankValues,
  calculateExpectedPointCardsInTrick,
  calculateUsedCardValue,
  collectKnownCardIdsForPlayEvaluation,
  ConservativeBiddingAgent,
  createSpecialCardsForTrump,
  estimateLeadWinProbability,
  evaluateCardForTrump,
  getBidLimitForScore,
  NoLegalActionsError,
  RandomAgent,
  RuleBasedAgent,
  runAutomatedGame,
  selectAdjutantCardId,
  selectDiscardCardIds,
  type Agent
} from "../src/index.js";

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

describe("card evaluation", () => {
  it("uses shared AI rank values", () => {
    expect(aiRankValues.A).toBe(20);
    expect(aiRankValues.K).toBe(13);
    expect(aiRankValues.Q).toBe(12);
    expect(aiRankValues.J).toBe(11);
    expect(aiRankValues["10"]).toBe(10);
    expect(aiRankValues["2"]).toBe(12);
  });

  it("evaluates trump and special cards without double-counting trump bonuses", () => {
    const spadeSpecials = createSpecialCardsForTrump("spades");

    expect(evaluateCardForTrump(card("spades", "K"), "spades", spadeSpecials)).toBe(43);
    expect(evaluateCardForTrump(card("hearts", "K"), "spades", spadeSpecials)).toBe(13);
    expect(evaluateCardForTrump(card("spades", "A"), "spades", spadeSpecials)).toBe(60);
    expect(evaluateCardForTrump(card("spades", "J"), "spades", spadeSpecials)).toBe(55);
    expect(evaluateCardForTrump(card("clubs", "J"), "spades", spadeSpecials)).toBe(50);
    expect(evaluateCardForTrump(card("hearts", "Q"), "spades", spadeSpecials)).toBe(30);
    expect(
      evaluateCardForTrump(card("hearts", "Q"), "hearts", createSpecialCardsForTrump("hearts"))
    ).toBe(45);
    expect(evaluateCardForTrump(joker(), "spades", spadeSpecials)).toBe(20);
  });

  it("derives sei jack and ura jack ids for each trump suit", () => {
    expect(createSpecialCardsForTrump("spades")).toMatchObject({
      seiJackCardId: "spades-J",
      uraJackCardId: "clubs-J"
    });
    expect(createSpecialCardsForTrump("clubs")).toMatchObject({
      seiJackCardId: "clubs-J",
      uraJackCardId: "spades-J"
    });
    expect(createSpecialCardsForTrump("hearts")).toMatchObject({
      seiJackCardId: "hearts-J",
      uraJackCardId: "diamonds-J"
    });
    expect(createSpecialCardsForTrump("diamonds")).toMatchObject({
      seiJackCardId: "diamonds-J",
      uraJackCardId: "hearts-J"
    });
  });
});

describe("RuleBasedAgent bidding", () => {
  it("converts hand scores to bid limits", () => {
    expect(getBidLimitForScore(199)).toBeNull();
    expect(getBidLimitForScore(200)).toBe(13);
    expect(getBidLimitForScore(230)).toBe(14);
    expect(getBidLimitForScore(260)).toBe(15);
    expect(getBidLimitForScore(290)).toBe(16);
    expect(getBidLimitForScore(410)).toBe(20);
  });

  it("bids the minimum legal target for the highest-scoring suit", async () => {
    const state = createInitialGame({ rng: () => 0 });
    const playerId = state.currentPlayerId;
    const view = withSelfHand(createPlayerView(state, playerId), playerId, strongSpadeHand());
    const legalActions: readonly GameAction[] = [
      { type: "pass", playerId },
      { type: "bid", playerId, suit: "spades", targetPointCards: 13 },
      { type: "bid", playerId, suit: "hearts", targetPointCards: 13 }
    ];
    const agent = new RuleBasedAgent(() => 0);

    await expect(agent.selectAction({ playerId, view, legalActions })).resolves.toEqual({
      type: "bid",
      playerId,
      suit: "spades",
      targetPointCards: 13
    });
  });

  it("raises by one target when below its limit and passes at the limit", async () => {
    const state = createInitialGame({ rng: () => 0 });
    const playerId = state.currentPlayerId;
    const baseView = withSelfHand(createPlayerView(state, playerId), playerId, strongSpadeHand());
    const agent = new RuleBasedAgent(() => 0);

    await expect(
      agent.selectAction({
        playerId,
        view: {
          ...baseView,
          bidding: {
            starterPlayerId: playerId,
            highestBid: { playerId: "player-1", suit: "hearts", targetPointCards: 13 },
            consecutivePassCount: 0,
            history: []
          }
        },
        legalActions: [
          { type: "pass", playerId },
          { type: "bid", playerId, suit: "spades", targetPointCards: 14 }
        ]
      })
    ).resolves.toEqual({ type: "bid", playerId, suit: "spades", targetPointCards: 14 });

    await expect(
      agent.selectAction({
        playerId,
        view: {
          ...baseView,
          bidding: {
            starterPlayerId: playerId,
            highestBid: { playerId: "player-1", suit: "hearts", targetPointCards: 20 },
            consecutivePassCount: 0,
            history: []
          }
        },
        legalActions: [{ type: "pass", playerId }]
      })
    ).resolves.toEqual({ type: "pass", playerId });
  });

  it("passes below threshold or when the desired bid is not legal", async () => {
    const state = createInitialGame({ rng: () => 0 });
    const playerId = state.currentPlayerId;
    const agent = new RuleBasedAgent(() => 0);

    await expect(
      agent.selectAction({
        playerId,
        view: withSelfHand(createPlayerView(state, playerId), playerId, [
          card("clubs", "3"),
          card("diamonds", "4"),
          card("hearts", "5")
        ]),
        legalActions: [
          { type: "pass", playerId },
          { type: "bid", playerId, suit: "clubs", targetPointCards: 13 }
        ]
      })
    ).resolves.toEqual({ type: "pass", playerId });

    await expect(
      agent.selectAction({
        playerId,
        view: withSelfHand(createPlayerView(state, playerId), playerId, strongSpadeHand()),
        legalActions: [
          { type: "pass", playerId },
          { type: "bid", playerId, suit: "hearts", targetPointCards: 13 }
        ]
      })
    ).resolves.toEqual({ type: "pass", playerId });
  });

  it("uses a conservative bidding baseline with a higher fixed-seed pass rate", async () => {
    const ruleBased = await countBiddingActions((rng) => new RuleBasedAgent(rng));
    const conservative = await countBiddingActions((rng) => new ConservativeBiddingAgent(rng));

    expect(conservative.bidCount).toBeGreaterThan(0);
    expect(conservative.passRate).toBeGreaterThan(ruleBased.passRate + 0.1);
  });
});

describe("RuleBasedAgent adjutant and exchange", () => {
  it("chooses the highest-valued adjutant candidate not in its own hand", async () => {
    const choosing = createAllPassAdjutantChoiceState();
    const playerId = choosing.currentPlayerId;
    const view = createPlayerView(choosing, playerId);
    const legalActions = getLegalActions(choosing, playerId);
    const agent = new RuleBasedAgent(() => 0);

    await expect(agent.selectAction({ playerId, view, legalActions })).resolves.toEqual({
      type: "choose-adjutant",
      playerId,
      cardId: "clubs-J"
    });
  });

  it("can prefer ura jack over sei jack when the trump count bonus is high", () => {
    const selfHand = [
      card("spades", "A"),
      card("spades", "2"),
      card("spades", "3"),
      card("spades", "4"),
      card("spades", "5")
    ];

    expect(
      selectAdjutantCardId(
        selfHand,
        "spades",
        { specialCards: createSpecialCardsForTrump("spades") },
        () => 0
      )
    ).toBe("clubs-J");
  });

  it("returns null when normal adjutant candidates are unavailable", () => {
    expect(
      selectAdjutantCardId(
        createDeck(),
        "spades",
        { specialCards: createSpecialCardsForTrump("spades") },
        () => 0.999
      )
    ).toBeNull();
  });

  it("uses only legal adjutant actions for fallback when normal candidates are unavailable", async () => {
    const choosing = createAllPassAdjutantChoiceState();
    const playerId = choosing.currentPlayerId;
    const view = withSelfHand(createPlayerView(choosing, playerId), playerId, createDeck());
    const legalActions: readonly GameAction[] = [
      { type: "pass", playerId },
      { type: "choose-adjutant", playerId, cardId: "hearts-3" },
      { type: "choose-adjutant", playerId, cardId: "joker" }
    ];
    const agent = new RuleBasedAgent(() => 0.75);

    await expect(agent.selectAction({ playerId, view, legalActions })).resolves.toEqual({
      type: "choose-adjutant",
      playerId,
      cardId: "joker"
    });
  });

  it("can choose the joker when it is the only legal adjutant fallback", async () => {
    const choosing = createAllPassAdjutantChoiceState();
    const playerId = choosing.currentPlayerId;
    const view = createPlayerView(choosing, playerId);
    const legalActions: readonly GameAction[] = [
      { type: "choose-adjutant", playerId, cardId: "joker" }
    ];
    const agent = new RuleBasedAgent(() => 0);

    await expect(agent.selectAction({ playerId, view, legalActions })).resolves.toEqual({
      type: "choose-adjutant",
      playerId,
      cardId: "joker"
    });
  });

  it("does not return an adjutant card id outside legalActions", async () => {
    const choosing = createAllPassAdjutantChoiceState();
    const playerId = choosing.currentPlayerId;
    const view = createPlayerView(choosing, playerId);
    const legalActions: readonly GameAction[] = [
      { type: "choose-adjutant", playerId, cardId: "hearts-3" }
    ];
    const agent = new RuleBasedAgent(() => 0);

    await expect(agent.selectAction({ playerId, view, legalActions })).resolves.toEqual({
      type: "choose-adjutant",
      playerId,
      cardId: "hearts-3"
    });
  });

  it("throws NoLegalActionsError when fallback has no legal adjutant actions", async () => {
    const choosing = createAllPassAdjutantChoiceState();
    const playerId = choosing.currentPlayerId;
    const view = createPlayerView(choosing, playerId);
    const agent = new RuleBasedAgent(() => 0);

    await expect(
      agent.selectAction({ playerId, view, legalActions: [{ type: "pass", playerId }] })
    ).rejects.toBeInstanceOf(NoLegalActionsError);
  });

  it("uses injected rng to choose among legal adjutant fallback actions", async () => {
    const choosing = createAllPassAdjutantChoiceState();
    const playerId = choosing.currentPlayerId;
    const view = withSelfHand(createPlayerView(choosing, playerId), playerId, createDeck());
    const legalActions: readonly GameAction[] = [
      { type: "choose-adjutant", playerId, cardId: "hearts-3" },
      { type: "choose-adjutant", playerId, cardId: "joker" },
      { type: "choose-adjutant", playerId, cardId: "diamonds-4" }
    ];
    const agent = new RuleBasedAgent(() => 0.4);

    await expect(agent.selectAction({ playerId, view, legalActions })).resolves.toEqual({
      type: "choose-adjutant",
      playerId,
      cardId: "joker"
    });
  });

  it("selects the discard combination that maximizes the remaining hand score", () => {
    const hand = [
      card("spades", "A"),
      card("spades", "K"),
      card("spades", "Q"),
      card("spades", "10"),
      card("spades", "9"),
      card("spades", "8"),
      card("spades", "7"),
      card("hearts", "3"),
      card("diamonds", "4"),
      card("clubs", "5"),
      card("hearts", "4"),
      card("diamonds", "5"),
      card("clubs", "6")
    ];

    expect(
      selectDiscardCardIds(
        hand,
        3,
        "spades",
        { specialCards: createSpecialCardsForTrump("spades") },
        () => 0
      )
    ).toEqual(["hearts-3", "diamonds-4", "hearts-4"]);
  });

  it("returns three distinct discard ids from its own 13-card observation", async () => {
    const exchange = createAllPassExchangeState();
    const playerId = exchange.currentPlayerId;
    const view = createPlayerView(exchange, playerId);
    const agent = new RuleBasedAgent(() => 0);
    const action = await agent.selectAction({ playerId, view, legalActions: [] });

    expect(action.type).toBe("discard-cards");
    if (action.type === "discard-cards") {
      expect(action.cardIds).toHaveLength(3);
      expect(new Set(action.cardIds).size).toBe(3);
      expect(view.players[0].hand?.map((card) => card.id)).toEqual(
        expect.arrayContaining([...action.cardIds])
      );
    }
  });
});

describe("RuleBasedAgent trick play", () => {
  it("uses the configured win probability and card cost formulas", () => {
    expect(estimateLeadWinProbability(0)).toBe(0.05);
    expect(estimateLeadWinProbability(10)).toBe(0.1);
    expect(estimateLeadWinProbability(20)).toBe(0.2);
    expect(estimateLeadWinProbability(30)).toBe(0.35);
    expect(estimateLeadWinProbability(40)).toBe(0.55);
    expect(estimateLeadWinProbability(50)).toBe(0.7);
    expect(estimateLeadWinProbability(55)).toBe(0.8);
    expect(estimateLeadWinProbability(60)).toBe(0.9);
    expect(adjustTeamWinProbability(0.2, 4, true)).toBeCloseTo(0.2);
    expect(adjustTeamWinProbability(0.2, 0, true)).toBe(1);
    expect(adjustTeamWinProbability(0.2, 0, false)).toBe(0);
    expect(calculateUsedCardValue(5)).toBeCloseTo(-0.5);
    expect(calculateUsedCardValue(20)).toBeCloseTo(0);
    expect(calculateUsedCardValue(50)).toBeCloseTo(1);
    expect(calculateUsedCardValue(60)).toBeCloseTo(1.333333);
  });

  it("excludes completed non-point cards from the expected point denominator", () => {
    const candidateCard = card("spades", "2");
    const withoutCompletedTricks = createPointExpectationView();
    const withCompletedNonPointCards = createPointExpectationView({
      completedTricks: [
        completedTrick(1, "player-1", [
          card("hearts", "3"),
          card("diamonds", "4"),
          card("clubs", "5")
        ])
      ]
    });
    const totalPointCardCount = createDeck().filter(isPointCard).length;
    const expectedWithoutCompletedTricks = (4 * totalPointCardCount) / 52;
    const expectedWithCompletedNonPointCards = (4 * totalPointCardCount) / 49;

    expect(
      calculateExpectedPointCardsInTrick(withoutCompletedTricks, candidateCard)
    ).toBeCloseTo(expectedWithoutCompletedTricks);
    expect(
      calculateExpectedPointCardsInTrick(withCompletedNonPointCards, candidateCard)
    ).toBeCloseTo(expectedWithCompletedNonPointCards);
    expect(
      calculateExpectedPointCardsInTrick(withCompletedNonPointCards, candidateCard)
    ).toBeGreaterThan(calculateExpectedPointCardsInTrick(withoutCompletedTricks, candidateCard));
  });

  it("does not double-count completed point cards also visible as captured point cards", () => {
    const candidateCard = card("spades", "2");
    const capturedPointCard = card("hearts", "A");
    const view = createPointExpectationView({
      completedTricks: [completedTrick(1, "player-1", [capturedPointCard])],
      capturedPointCards: {
        "player-1": [capturedPointCard]
      }
    });
    const knownCardIds = collectKnownCardIdsForPlayEvaluation(view, candidateCard);
    const totalPointCardCount = createDeck().filter(isPointCard).length;
    const expected = (4 * (totalPointCardCount - 1)) / 51;

    expect(knownCardIds.has(candidateCard.id)).toBe(true);
    expect(knownCardIds.has(capturedPointCard.id)).toBe(true);
    expect(knownCardIds.size).toBe(2);
    expect(calculateExpectedPointCardsInTrick(view, candidateCard)).toBeCloseTo(expected);
  });

  it("deduplicates cards overlapping with the current trick", () => {
    const candidateCard = card("spades", "2");
    const currentPointCard = card("hearts", "A");
    const view = createPointExpectationView({
      currentTrick: [{ playerId: "player-2", card: currentPointCard }],
      completedTricks: [completedTrick(1, "player-1", [currentPointCard])],
      capturedPointCards: {
        "player-1": [currentPointCard]
      }
    });
    const knownCardIds = collectKnownCardIdsForPlayEvaluation(view, candidateCard);
    const totalPointCardCount = createDeck().filter(isPointCard).length;
    const expected = 1 + (3 * (totalPointCardCount - 1)) / 51;

    expect(knownCardIds.has(candidateCard.id)).toBe(true);
    expect(knownCardIds.has(currentPointCard.id)).toBe(true);
    expect(knownCardIds.size).toBe(2);
    expect(calculateExpectedPointCardsInTrick(view, candidateCard)).toBeCloseTo(expected);
  });

  it("does not include hidden non-point buried card ids in known cards", () => {
    const candidateCard = card("spades", "2");
    const hiddenBuriedCards = [card("clubs", "3"), card("diamonds", "4")];
    const state: GameState = {
      ...toPlayingState(createInitialGame({ rng: () => 0 })),
      excludedCards: hiddenBuriedCards,
      unusedCards: hiddenBuriedCards
    };
    const view = withSelfHand(createPlayerView(state, "player-0"), "player-0", [candidateCard]);
    const knownCardIds = collectKnownCardIdsForPlayEvaluation(view, candidateCard);
    const totalPointCardCount = createDeck().filter(isPointCard).length;

    expect(knownCardIds.has(hiddenBuriedCards[0].id)).toBe(false);
    expect(knownCardIds.has(hiddenBuriedCards[1].id)).toBe(false);
    expect(calculateExpectedPointCardsInTrick(view, candidateCard)).toBeCloseTo(
      (4 * totalPointCardCount) / 52
    );
  });

  it("uses game-core provisional winner logic, including same two on the fifth card", async () => {
    const state = toPlayingState(createInitialGame({ rng: () => 0 }));
    const playerId = "player-0";
    const view = withSelfHand(
      {
        ...createPlayerView(state, playerId),
        phase: "playing",
        trickNumber: 2,
        currentTrick: [
          { playerId: "player-1", card: card("hearts", "A") },
          { playerId: "player-2", card: card("hearts", "K") },
          { playerId: "player-3", card: card("hearts", "Q") },
          { playerId: "player-4", card: card("hearts", "7") }
        ],
        contract: {
          napoleonPlayerId: "player-1",
          trumpSuit: "spades",
          targetPointCards: 13
        },
        trumpSuit: "spades",
        specialCards: createSpecialCardsForTrump("spades")
      },
      playerId,
      [card("hearts", "2"), card("hearts", "3")]
    );
    const agent = new RuleBasedAgent(() => 0);

    await expect(
      agent.selectAction({
        playerId,
        view,
        legalActions: [
          { type: "play-card", playerId, cardId: "hearts-2" },
          { type: "play-card", playerId, cardId: "hearts-3" }
        ]
      })
    ).resolves.toEqual({ type: "play-card", playerId, cardId: "hearts-2" });
  });

  it("throws when there are no legal play-card actions in playing phase", async () => {
    const state = toPlayingState(createInitialGame({ rng: () => 0 }));
    const playerId = "player-0";
    const view = createPlayerView(state, playerId);
    const agent = new RuleBasedAgent(() => 0);

    await expect(agent.selectAction({ playerId, view, legalActions: [] })).rejects.toBeInstanceOf(
      NoLegalActionsError
    );
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

function strongSpadeHand(): readonly Card[] {
  return [
    card("spades", "A"),
    card("spades", "K"),
    card("spades", "Q"),
    card("spades", "J"),
    card("spades", "10"),
    joker(),
    card("hearts", "A"),
    card("clubs", "A"),
    card("diamonds", "A"),
    card("hearts", "K")
  ];
}

function withSelfHand(view: PlayerView, playerId: string, hand: readonly Card[]): PlayerView {
  return {
    ...view,
    players: view.players.map((player) =>
      player.id === playerId ? { ...player, hand, handCount: hand.length } : player
    )
  };
}

async function countBiddingActions(
  createAgent: (rng: () => number) => Agent
): Promise<{ passCount: number; bidCount: number; passRate: number }> {
  let passCount = 0;
  let bidCount = 0;

  for (let seed = 1000; seed < 1100; seed += 1) {
    const record = await runAutomatedGame({
      seed,
      createAgent: ({ rng }) => createAgent(rng)
    });

    for (const decision of record.decisions) {
      if (decision.phase !== "bidding") {
        continue;
      }
      if (decision.action.type === "pass") {
        passCount += 1;
      } else if (decision.action.type === "bid") {
        bidCount += 1;
      }
    }
  }

  return {
    passCount,
    bidCount,
    passRate: passCount / (passCount + bidCount)
  };
}

function createPointExpectationView(options: {
  currentTrick?: PlayerView["currentTrick"];
  completedTricks?: PlayerView["completedTricks"];
  capturedPointCards?: Partial<Record<string, readonly StandardCard[]>>;
} = {}): PlayerView {
  const playerId = "player-0";
  const completedTricks = options.completedTricks ?? [];
  const view = withSelfHand(
    createPlayerView(toPlayingState(createInitialGame({ rng: () => 0 })), playerId),
    playerId,
    [card("spades", "2")]
  );

  return {
    ...view,
    currentTrick: options.currentTrick ?? [],
    completedTricks,
    completedTrickCount: completedTricks.length,
    players: view.players.map((player) => ({
      ...player,
      capturedPointCards: options.capturedPointCards?.[player.id] ?? []
    }))
  };
}

function completedTrick(
  trickNumber: number,
  winnerId: string,
  cards: readonly Card[]
): PlayerView["completedTricks"][number] {
  return {
    trickNumber,
    winnerId,
    cards: cards.map((trickCard, index) => ({
      playerId: `player-${index}`,
      card: trickCard
    }))
  };
}

function toPlayingState(state: GameState): GameState {
  return {
    ...state,
    phase: "playing",
    trumpSuit: "spades",
    contract: {
      napoleonPlayerId: "player-0",
      trumpSuit: "spades",
      targetPointCards: 13
    },
    bidding: null,
    adjutant: null
  };
}
