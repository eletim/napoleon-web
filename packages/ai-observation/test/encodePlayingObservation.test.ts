import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import { createDeck, createPlayerView, getLegalActions } from "@napoleon/game-core";
import type { AdjutantState, GameState, PlayerId } from "@napoleon/game-core";
import {
  BIDDING_ACTION_TYPE_BID,
  BIDDING_ACTION_TYPE_PASS,
  CARD_COUNT,
  CARDS_PER_TRICK,
  EMPTY_BIDDING_SUIT_INDEX,
  EMPTY_CARD_INDEX,
  EMPTY_PLAYER_INDEX,
  PLAYER_COUNT,
  PLAYING_ENCODER_SCHEMA_VERSION,
  SELF_ROLE_ORDER,
  TRICK_COUNT,
  createEmptyEncodedBiddingHistory,
  createPlayingTrainingSample,
  encodePlayAction,
  encodePlayingObservation,
  getCardIndex,
  validateEncodedPlayingObservation
} from "../src/index.js";
import type { EncodedPlayingObservation } from "../src/index.js";

describe("encodePlayingObservation", () => {
  it("encodes a playing observation into fixed shapes", async () => {
    const { record, decision } = await getFirstPlayingDecision(12345);
    const encoded = encodePlayingObservation(
      decision.observation,
      record.playerIds,
      createEmptyEncodedBiddingHistory()
    );

    expect(encoded.schemaVersion).toBe(PLAYING_ENCODER_SCHEMA_VERSION);
    expect(sum(encoded.biddingHistory.actionMask)).toBe(0);
    expect(encoded.relativePlayerIds[0]).toBe(decision.playerId);
    expect(encoded.trumpSuitOneHot).toHaveLength(4);
    expect(encoded.napoleonPlayerOneHot).toHaveLength(PLAYER_COUNT);
    expect(encoded.revealedAdjutantPlayerOneHot).toHaveLength(PLAYER_COUNT + 1);
    expect(encoded.selfRoleOneHot).toHaveLength(SELF_ROLE_ORDER.length);
    expect(encoded.calledAdjutantCardMask).toHaveLength(CARD_COUNT);
    expect(encoded.selfHandMask).toHaveLength(CARD_COUNT);
    expect(encoded.legalPlayMask).toHaveLength(CARD_COUNT);
    expect(encoded.handCountByPlayer).toHaveLength(PLAYER_COUNT);
    expect(encoded.capturedPointCardMaskByPlayer).toHaveLength(PLAYER_COUNT);
    expect(encoded.currentTrickCardIndices).toHaveLength(CARDS_PER_TRICK);
    expect(encoded.currentTrickPlayerIndices).toHaveLength(CARDS_PER_TRICK);
    expect(encoded.currentTrickSlotMask).toHaveLength(CARDS_PER_TRICK);
    expect(encoded.completedTrickCardIndices).toHaveLength(TRICK_COUNT * CARDS_PER_TRICK);
    expect(encoded.completedTrickPlayerIndices).toHaveLength(TRICK_COUNT * CARDS_PER_TRICK);
    expect(encoded.completedTrickSlotMask).toHaveLength(TRICK_COUNT * CARDS_PER_TRICK);
    expect(encoded.completedTrickWinnerIndices).toHaveLength(TRICK_COUNT);
    expect(encoded.completedTrickMask).toHaveLength(TRICK_COUNT);
    expect(encoded.latestBuriedEventPointCardMask).toHaveLength(CARD_COUNT);
    validateEncodedPlayingObservation(encoded);
  });

  it("encodes exactly one stable self role without exposing opponent hidden ownership", () => {
    const cases: readonly {
      name: string;
      playerId: PlayerId;
      adjutant: AdjutantState;
      expectedRole: typeof SELF_ROLE_ORDER[number];
    }[] = [
      {
        name: "Napoleon",
        playerId: "player-0",
        adjutant: { calledCardId: "spades-A", playerId: "player-1", revealed: false },
        expectedRole: "napoleon"
      },
      {
        name: "Adjutant before reveal",
        playerId: "player-1",
        adjutant: { calledCardId: "spades-A", playerId: "player-1", revealed: false },
        expectedRole: "adjutant"
      },
      {
        name: "Adjutant after reveal",
        playerId: "player-1",
        adjutant: { calledCardId: "spades-A", playerId: "player-1", revealed: true },
        expectedRole: "adjutant"
      },
      {
        name: "Alliance",
        playerId: "player-2",
        adjutant: { calledCardId: "spades-A", playerId: "player-1", revealed: false },
        expectedRole: "alliance"
      },
      {
        name: "Napoleon-solo",
        playerId: "player-0",
        adjutant: { calledCardId: "spades-A", playerId: null, revealed: false },
        expectedRole: "napoleon-solo"
      }
    ];

    for (const testCase of cases) {
      const encoded = encodeFixtureObservation(testCase.playerId, testCase.adjutant);
      const expectedIndex = SELF_ROLE_ORDER.indexOf(testCase.expectedRole);

      expect(encoded.selfRoleOneHot, testCase.name).toEqual(
        SELF_ROLE_ORDER.map((_, index) => (index === expectedIndex ? 1 : 0))
      );
      expect(sum(encoded.selfRoleOneHot), testCase.name).toBe(1);
    }

    const beforeReveal = encodeFixtureObservation("player-1", {
      calledCardId: "spades-A",
      playerId: "player-1",
      revealed: false
    });
    const afterReveal = encodeFixtureObservation("player-1", {
      calledCardId: "spades-A",
      playerId: "player-1",
      revealed: true
    });
    expect(afterReveal.selfRoleOneHot).toEqual(beforeReveal.selfRoleOneHot);
  });

  it("preserves current trick order and empty slots", async () => {
    const record = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = record.decisions.find(
      (candidate) => candidate.phase === "playing" && candidate.observation.view.currentTrick.length > 0
    );

    if (decision === undefined) {
      throw new Error("Expected a playing decision with a non-empty current trick.");
    }

    const encoded = encodePlayingObservation(
      decision.observation,
      record.playerIds,
      createEmptyEncodedBiddingHistory()
    );
    const trick = decision.observation.view.currentTrick;

    expect(encoded.currentTrickSlotMask.slice(0, trick.length)).toEqual(
      Array(trick.length).fill(1)
    );
    expect(encoded.currentTrickSlotMask.slice(trick.length)).toEqual(
      Array(CARDS_PER_TRICK - trick.length).fill(0)
    );
    expect(encoded.currentTrickCardIndices.slice(0, trick.length)).toEqual(
      trick.map((playedCard) => getCardIndex(playedCard.card.id))
    );
    expect(encoded.currentTrickCardIndices.slice(trick.length)).toEqual(
      Array(CARDS_PER_TRICK - trick.length).fill(EMPTY_CARD_INDEX)
    );
    expect(encoded.currentTrickPlayerIndices.slice(trick.length)).toEqual(
      Array(CARDS_PER_TRICK - trick.length).fill(EMPTY_PLAYER_INDEX)
    );
  });

  it("encodes completed tricks as flattened fixed slots", async () => {
    const record = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = record.decisions.find(
      (candidate) => candidate.phase === "playing" && candidate.observation.view.completedTricks.length > 0
    );

    if (decision === undefined) {
      throw new Error("Expected a playing decision with completed tricks.");
    }

    const encoded = encodePlayingObservation(
      decision.observation,
      record.playerIds,
      createEmptyEncodedBiddingHistory()
    );
    const completedSlotCount =
      decision.observation.view.completedTricks.length * CARDS_PER_TRICK;

    expect(encoded.completedTrickSlotMask.slice(0, completedSlotCount)).toEqual(
      Array(completedSlotCount).fill(1)
    );
    expect(encoded.completedTrickSlotMask.slice(completedSlotCount)).toEqual(
      Array(TRICK_COUNT * CARDS_PER_TRICK - completedSlotCount).fill(0)
    );
    expect(encoded.completedTrickMask.slice(0, decision.observation.view.completedTricks.length)).toEqual(
      Array(decision.observation.view.completedTricks.length).fill(1)
    );
  });

  it("encodes the latest buried-card event without hidden card ids", async () => {
    const record = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = record.decisions.find(
      (candidate) =>
        candidate.phase === "playing" &&
        candidate.observation.view.latestEvent?.type === "buried-cards-resolved"
    );

    if (decision === undefined) {
      throw new Error("Expected a playing decision with a buried-card event.");
    }

    const encoded = encodePlayingObservation(
      decision.observation,
      record.playerIds,
      createEmptyEncodedBiddingHistory()
    );
    const event = decision.observation.view.latestEvent;

    if (event?.type !== "buried-cards-resolved") {
      throw new Error("Expected buried-card event.");
    }

    expect(encoded.latestBuriedEventPresent).toBe(1);
    expect(encoded.latestBuriedEventHiddenNonPointCount).toBe(event.hiddenNonPointCardCount);
    expect(sum(encoded.latestBuriedEventPointCardMask)).toBe(event.awardedPointCards.length);
  });

  it("rejects non-playing observations and non-play legal actions", async () => {
    const record = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const biddingDecision = record.decisions.find((candidate) => candidate.phase === "bidding");

    if (biddingDecision === undefined) {
      throw new Error("Expected a bidding decision.");
    }

    expect(() =>
      encodePlayingObservation(
        biddingDecision.observation,
        record.playerIds,
        createEmptyEncodedBiddingHistory()
      )
    ).toThrow("requires a playing observation");

    const { decision } = await getFirstPlayingDecision(12345);
    const invalidObservation = {
      ...decision.observation,
      legalActions: [
        ...decision.observation.legalActions,
        { type: "pass" as const, playerId: decision.playerId }
      ]
    };

    expect(() =>
      encodePlayingObservation(
        invalidObservation,
        record.playerIds,
        createEmptyEncodedBiddingHistory()
      )
    ).toThrow("only play-card actions");
  });

  it("encodes selected play-card actions and validates the legal mask", async () => {
    const { record, decision } = await getFirstPlayingDecision(12345);
    const observation = encodePlayingObservation(
      decision.observation,
      record.playerIds,
      createEmptyEncodedBiddingHistory()
    );

    expect(encodePlayAction(decision.action, observation.legalPlayMask)).toEqual({
      selectedCardIndex: getCardIndex(decision.action.type === "play-card" ? decision.action.cardId : "")
    });

    expect(() =>
      encodePlayAction(
        { type: "pass", playerId: decision.playerId },
        observation.legalPlayMask
      )
    ).toThrow("requires a play-card action");
    expect(() =>
      encodePlayAction(
        { type: "play-card", playerId: decision.playerId, cardId: "joker" },
        Array(CARD_COUNT).fill(0)
      )
    ).toThrow("Selected card is not legal");
  });

  it("rejects invalid scalar, index, one-hot, mask, slot, and event values", async () => {
    const encoded = await getEncodedPlayingObservation(12345);

    expectInvalid(encoded, (draft) => {
      draft.currentTrickSlotMask = replaceAt(draft.currentTrickSlotMask, 0, 1);
      draft.currentTrickCardIndices = replaceAt(draft.currentTrickCardIndices, 0, 53);
      draft.currentTrickPlayerIndices = replaceAt(draft.currentTrickPlayerIndices, 0, 0);
    }, "currentTrickCardIndices");
    expectInvalid(encoded, (draft) => {
      draft.currentTrickSlotMask = replaceAt(draft.currentTrickSlotMask, 0, 1);
      draft.currentTrickCardIndices = replaceAt(draft.currentTrickCardIndices, 0, -2);
      draft.currentTrickPlayerIndices = replaceAt(draft.currentTrickPlayerIndices, 0, 0);
    }, "currentTrickCardIndices");
    expectInvalid(encoded, (draft) => {
      draft.currentTrickSlotMask = replaceAt(draft.currentTrickSlotMask, 0, 1);
      draft.currentTrickCardIndices = replaceAt(draft.currentTrickCardIndices, 0, 0);
      draft.currentTrickPlayerIndices = replaceAt(draft.currentTrickPlayerIndices, 0, 5);
    }, "currentTrickPlayerIndices");
    expectInvalid(encoded, (draft) => {
      draft.currentTrickSlotMask = replaceAt(draft.currentTrickSlotMask, 0, 1);
      draft.currentTrickCardIndices = replaceAt(draft.currentTrickCardIndices, 0, 0);
      draft.currentTrickPlayerIndices = replaceAt(draft.currentTrickPlayerIndices, 0, -2);
    }, "currentTrickPlayerIndices");
    expectInvalid(encoded, (draft) => {
      draft.specialCardIndices = { ...draft.specialCardIndices, oruma: 1.5 };
    }, "finite integer");
    expectInvalid(encoded, (draft) => {
      draft.specialCardIndices = { ...draft.specialCardIndices, yoromeki: Number.NaN };
    }, "finite integer");
    expectInvalid(encoded, (draft) => {
      draft.specialCardIndices = { ...draft.specialCardIndices, oruma: Number.POSITIVE_INFINITY };
    }, "finite integer");
    expectInvalid(encoded, (draft) => {
      draft.trumpSuitOneHot = [0, 0, 0, 0];
    }, "trumpSuitOneHot must sum to 1");
    expectInvalid(encoded, (draft) => {
      draft.trumpSuitOneHot = [1, 1, 0, 0];
    }, "trumpSuitOneHot must sum to 1");
    expectInvalid(encoded, (draft) => {
      const index = draft.selfHandMask.findIndex((value) => value === 0);
      draft.legalPlayMask = replaceAt(draft.legalPlayMask, index, 1);
    }, "legalPlayMask");
    expectInvalid(encoded, (draft) => {
      const emptyIndex = draft.currentTrickSlotMask.findIndex((value) => value === 0);
      draft.currentTrickCardIndices = replaceAt(draft.currentTrickCardIndices, emptyIndex, 0);
    }, "empty card and player indices");
    expectInvalid(encoded, (draft) => {
      draft.currentTrickSlotMask = replaceAt(draft.currentTrickSlotMask, 0, 1);
      draft.currentTrickCardIndices = replaceAt(draft.currentTrickCardIndices, 0, EMPTY_CARD_INDEX);
      draft.currentTrickPlayerIndices = replaceAt(draft.currentTrickPlayerIndices, 0, 0);
    }, "cardIndex");
    expectInvalid(encoded, (draft) => {
      draft.completedTrickCount = draft.completedTrickCount + 1;
    }, "completedTrickCount must equal completedTrickMask sum");
    expectInvalid(encoded, (draft) => {
      draft.currentTrickSlotMask = [1, 0, 1, 0, 0];
    }, "contiguous");
    expectInvalid(encoded, (draft) => {
      draft.latestBuriedEventPresent = 0;
      draft.latestBuriedEventPointCardMask = replaceAt(draft.latestBuriedEventPointCardMask, 0, 1);
    }, "latestBuriedEvent fields must be empty");
  });

  it("rejects invalid bidding history values", async () => {
    const encoded = await getEncodedPlayingObservation(12345);
    const firstBiddingSlot = encoded.biddingHistory.actionMask.findIndex((value) => value === 1);

    expect(firstBiddingSlot).toBeGreaterThanOrEqual(0);
    expectInvalid(encoded, (draft) => {
      draft.biddingHistory = {
        ...draft.biddingHistory,
        suitIndices: replaceAt(
          draft.biddingHistory.suitIndices,
          firstBiddingSlot,
          0
        ),
        actionTypeIndices: replaceAt(
          draft.biddingHistory.actionTypeIndices,
          firstBiddingSlot,
          BIDDING_ACTION_TYPE_PASS
        ),
        targetPointCards: replaceAt(draft.biddingHistory.targetPointCards, firstBiddingSlot, 0)
      };
    }, "pass must use suit -1 and target 0");
    expectInvalid(encoded, (draft) => {
      draft.biddingHistory = {
        ...draft.biddingHistory,
        actionTypeIndices: replaceAt(
          draft.biddingHistory.actionTypeIndices,
          firstBiddingSlot,
          BIDDING_ACTION_TYPE_BID
        ),
        suitIndices: replaceAt(
          draft.biddingHistory.suitIndices,
          firstBiddingSlot,
          EMPTY_BIDDING_SUIT_INDEX + 1
        ),
        targetPointCards: replaceAt(draft.biddingHistory.targetPointCards, firstBiddingSlot, 0)
      };
    }, "targetPointCards");
  });
});

function encodeFixtureObservation(
  playerId: PlayerId,
  adjutant: AdjutantState
): EncodedPlayingObservation {
  const state = createPlayingFixtureState(playerId, adjutant);
  const view = createPlayerView(state, playerId);
  return encodePlayingObservation(
    {
      playerId,
      view,
      legalActions: getLegalActions(state, playerId)
    },
    state.players.map((player) => player.id),
    createEmptyEncodedBiddingHistory()
  );
}

function createPlayingFixtureState(currentPlayerId: PlayerId, adjutant: AdjutantState): GameState {
  const deck = createDeck();
  const playerIds = ["player-0", "player-1", "player-2", "player-3", "player-4"];
  const players = playerIds.map((id, index) => ({
    id,
    hand: deck.slice(index * 10, index * 10 + 10)
  }));

  return {
    players,
    phase: "playing",
    currentPlayerId,
    currentTrick: [],
    completedTricks: [],
    trumpSuit: "spades",
    contract: {
      napoleonPlayerId: "player-0",
      trumpSuit: "spades",
      targetPointCards: 12
    },
    adjutant,
    bidding: null,
    awardedPointCards: [],
    excludedCards: deck.slice(50, 53),
    latestEvent: null,
    result: null,
    trickNumber: 1,
    isTrickComplete: false,
    isGameOver: false,
    unusedCards: []
  };
}

async function getFirstPlayingDecision(seed: number) {
  const record = await runAutomatedGame({
    seed,
    createAgent: ({ rng }) => new RuleBasedAgent(rng)
  });
  const decision = record.decisions.find((candidate) => candidate.phase === "playing");

  if (decision === undefined) {
    throw new Error("Expected at least one playing decision.");
  }

  return { record, decision };
}

async function getEncodedPlayingObservation(seed: number): Promise<EncodedPlayingObservation> {
  const { record, decision } = await getFirstPlayingDecision(seed);
  const sample = createPlayingTrainingSample(record, decision);

  if (sample === null) {
    throw new Error("Expected a playing sample.");
  }

  return sample.observation;
}

function expectInvalid(
  encoded: EncodedPlayingObservation,
  mutate: (draft: EncodedPlayingObservation) => void,
  message: string
): void {
  const draft = structuredClone(encoded);
  mutate(draft);
  expect(() => validateEncodedPlayingObservation(draft)).toThrow(message);
}

function replaceAt(values: readonly number[], index: number, value: number): readonly number[] {
  if (index < 0) {
    throw new Error("Expected a valid replacement index.");
  }

  const copy = [...values];
  copy[index] = value;
  return copy;
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
