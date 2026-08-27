import {
  createDeck,
  isPointCard,
  isStandardCard
} from "@napoleon/game-core";
import type {
  Card,
  ChooseAdjutantAction,
  DiscardCardsAction,
  GameAction,
  PlayerView,
  Rank,
  Suit
} from "@napoleon/game-core";
import { getAiRankValue } from "./cardEvaluation.js";
import { NoLegalActionsError } from "./errors.js";
import type { Agent, PlayerObservation } from "./types.js";

export const PARAMETERIZED_ADJUTANT_FEATURE_COUNT = 35;
export const PARAMETERIZED_EXCHANGE_FEATURE_COUNT = 60;
export const PARAMETERIZED_NON_PLAYING_PARAMETER_COUNT = 95;
export const PARAMETERIZED_NON_PLAYING_FEATURE_SCHEMA_VERSION = 1;

const suits: readonly Suit[] = ["spades", "hearts", "diamonds", "clubs"];
const highRanks = new Set<Rank>(["A", "K", "Q", "J"]);
const cardOrder = new Map(createDeck().map((card, index) => [card.id, index]));
const epsilon = 1e-12;

export interface ParameterizedNonPlayingParameters {
  featureSchemaVersion: 1;
  adjutantWeights: readonly number[];
  exchangeWeights: readonly number[];
}

export interface ParameterizedNonPlayingSelection<T extends GameAction> {
  action: T;
  score: number;
  features: readonly number[];
}

export class ParameterizedNonPlayingAgent implements Agent {
  private originalHandCardIds: ReadonlySet<string> | null = null;

  constructor(private readonly parameters: ParameterizedNonPlayingParameters) {
    validateParameterizedNonPlayingParameters(parameters);
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    switch (observation.view.phase) {
      case "choosing-adjutant": {
        const hand = getSelfHand(observation);
        if (hand.length !== 10) {
          throw new Error(`Parameterized adjutant policy expected 10 visible cards, got ${hand.length}.`);
        }
        this.originalHandCardIds = new Set(hand.map((card) => card.id));
        return selectParameterizedAdjutant(observation, this.parameters.adjutantWeights).action;
      }
      case "exchanging": {
        if (this.originalHandCardIds === null) {
          throw new Error(
            "Parameterized exchange policy is missing the visible pre-kitty hand captured during adjutant selection."
          );
        }
        const hand = getSelfHand(observation);
        const kittyCardIds = new Set(
          hand
            .filter((card) => !this.originalHandCardIds?.has(card.id))
            .map((card) => card.id)
        );
        if (kittyCardIds.size !== 3) {
          throw new Error(`Parameterized exchange policy expected 3 visible kitty cards, got ${kittyCardIds.size}.`);
        }
        return selectParameterizedExchange(
          observation,
          kittyCardIds,
          this.parameters.exchangeWeights
        ).action;
      }
      default:
        throw new Error(
          `Parameterized non-playing policy cannot handle phase ${observation.view.phase}.`
        );
    }
  }
}

export function validateParameterizedNonPlayingParameters(
  parameters: ParameterizedNonPlayingParameters
): void {
  if (parameters.featureSchemaVersion !== PARAMETERIZED_NON_PLAYING_FEATURE_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported parameterized non-playing feature schema version: ${parameters.featureSchemaVersion}.`
    );
  }
  assertFiniteWeights(
    "adjutant",
    parameters.adjutantWeights,
    PARAMETERIZED_ADJUTANT_FEATURE_COUNT
  );
  assertFiniteWeights(
    "exchange",
    parameters.exchangeWeights,
    PARAMETERIZED_EXCHANGE_FEATURE_COUNT
  );
}

export function selectParameterizedAdjutant(
  observation: PlayerObservation,
  weights: readonly number[]
): ParameterizedNonPlayingSelection<ChooseAdjutantAction> {
  assertFiniteWeights("adjutant", weights, PARAMETERIZED_ADJUTANT_FEATURE_COUNT);
  if (observation.view.phase !== "choosing-adjutant") {
    throw new Error("Parameterized adjutant selection requires choosing-adjutant phase.");
  }
  const candidates = observation.legalActions.filter(
    (action): action is ChooseAdjutantAction => action.type === "choose-adjutant"
  );
  let best: ParameterizedNonPlayingSelection<ChooseAdjutantAction> | null = null;
  for (const action of candidates) {
    const card = createDeck().find((candidate) => candidate.id === action.cardId);
    if (card === undefined) {
      throw new Error(`Unknown legal adjutant card id: ${action.cardId}.`);
    }
    const features = extractParameterizedAdjutantFeatures(observation, card);
    const score = dot(features, weights);
    if (
      best === null ||
      score > best.score + epsilon ||
      (Math.abs(score - best.score) <= epsilon && compareCardIds(action.cardId, best.action.cardId) < 0)
    ) {
      best = { action, score, features };
    }
  }
  if (best === null) {
    throw new NoLegalActionsError(observation.playerId);
  }
  return best;
}

export function selectParameterizedExchange(
  observation: PlayerObservation,
  kittyCardIds: ReadonlySet<string>,
  weights: readonly number[]
): ParameterizedNonPlayingSelection<DiscardCardsAction> {
  assertFiniteWeights("exchange", weights, PARAMETERIZED_EXCHANGE_FEATURE_COUNT);
  if (observation.view.phase !== "exchanging") {
    throw new Error("Parameterized exchange selection requires exchanging phase.");
  }
  const hand = getSelfHand(observation);
  const discardCount = observation.view.exchangeRequirement?.discardCount;
  if (discardCount !== 3 || hand.length !== 13 || kittyCardIds.size !== 3) {
    throw new Error(
      `Parameterized exchange requires 13 visible cards, 3 discards, and 3 kitty ids; got hand=${hand.length}, discards=${discardCount ?? "missing"}, kitty=${kittyCardIds.size}.`
    );
  }
  let best: ParameterizedNonPlayingSelection<DiscardCardsAction> | null = null;
  for (const discarded of combinations(hand, 3)) {
    const features = extractParameterizedExchangeFeatures(
      observation.view,
      discarded,
      kittyCardIds
    );
    const score = dot(features, weights);
    if (best === null || score > best.score + epsilon) {
      best = {
        action: {
          type: "discard-cards",
          playerId: observation.playerId,
          cardIds: discarded.map((card) => card.id)
        },
        score,
        features
      };
    }
  }
  if (best === null) {
    throw new NoLegalActionsError(observation.playerId);
  }
  return best;
}

export function extractParameterizedAdjutantFeatures(
  observation: Pick<PlayerObservation, "playerId" | "view" | "publicActionHistory">,
  candidate: Card
): readonly number[] {
  const { view } = observation;
  if (view.phase !== "choosing-adjutant" || view.contract === null) {
    throw new Error("Parameterized adjutant features require choosing-adjutant contract state.");
  }
  const hand = getSelfHand(observation);
  const trump = view.contract.trumpSuit;
  const candidateSuit = isStandardCard(candidate) ? candidate.suit : null;
  const lengths = suitLengths(hand);
  const candidateLength = candidateSuit === null ? 0 : lengths[candidateSuit];
  const held = hand.some((card) => card.id === candidate.id);
  const oruma = candidate.id === view.specialCards.orumaCardId;
  const sei = candidate.id === view.specialCards.seiJackCardId;
  const ura = candidate.id === view.specialCards.uraJackCardId;
  const yoro = candidate.id === view.specialCards.yoromekiCardId;
  const isTrump = isStandardCard(candidate) && candidate.suit === trump;
  const eligible = isStandardCard(candidate) && !held && !yoro && (oruma || sei || ura || isTrump);
  const generic = eligible && !oruma && !sei && !ura;
  const candidateCards = candidateSuit === null
    ? []
    : hand.filter(isStandardCard).filter((card) => card.suit === candidateSuit);
  const candidatePoints = candidateCards.filter(isPointCard).length;
  const candidateHighs = candidateCards.filter(isHighCard).length;
  const handPoints = hand.filter(isPointCard).length;
  const handHighs = hand.filter(isHighCard).length;
  const voids = suits.filter((suit) => lengths[suit] === 0).length;
  const singletons = suits.filter((suit) => lengths[suit] === 1).length;
  const trumpLength = lengths[trump];
  const target = (view.contract.targetPointCards - 13) / 6;
  const bids = (observation.publicActionHistory ?? []).filter(
    (record) => record.action.type === "bid"
  );
  const features = [
    1,
    flag(oruma),
    flag(sei),
    flag(ura),
    flag(candidate.type === "joker"),
    flag(yoro),
    flag(isTrump),
    flag(isPointCard(candidate)),
    flag(held),
    flag(eligible),
    flag(generic),
    aiRank(candidate) / 20,
    generic ? aiRank(candidate) / 20 : 0,
    candidateLength / 10,
    candidatePoints / 5,
    candidateHighs / 4,
    flag(candidateCards.some((card) => card.rank === "A")),
    flag(candidateCards.some((card) => card.rank === "K")),
    flag(candidateSuit !== null && candidateLength === 0),
    flag(candidateSuit !== null && candidateLength === 1),
    flag(candidateSuit !== null && candidateLength === 2),
    trumpLength / 10,
    handPoints / 10,
    handHighs / 10,
    voids / 4,
    singletons / 4,
    target,
    target * flag(oruma),
    target * flag(sei),
    target * flag(ura),
    target * flag(isTrump),
    ura ? trumpLength / 10 : 0,
    bids.length / 10,
    bids.filter((record) => record.playerId === observation.playerId).length / 4,
    target * trumpLength / 10
  ];
  assertFeatureCount("adjutant", features, PARAMETERIZED_ADJUTANT_FEATURE_COUNT);
  return features;
}

export function extractParameterizedExchangeFeatures(
  view: PlayerView,
  discarded: readonly Card[],
  kittyCardIds: ReadonlySet<string>
): readonly number[] {
  if (view.phase !== "exchanging" || view.contract === null || view.adjutant === null) {
    throw new Error("Parameterized exchange features require exchanging contract/adjutant state.");
  }
  if (discarded.length !== 3) {
    throw new Error(`Parameterized exchange features require 3 discards, got ${discarded.length}.`);
  }
  const self = view.players.find((player) => player.id === view.selfId);
  const hand = self?.hand;
  if (hand === undefined) {
    throw new Error("Parameterized exchange features require the visible self hand.");
  }
  const discardedIds = new Set(discarded.map((card) => card.id));
  const retained = hand.filter((card) => !discardedIds.has(card.id));
  if (retained.length !== 10) {
    throw new Error(`Parameterized exchange candidate must retain 10 cards, got ${retained.length}.`);
  }
  const trump = view.contract.trumpSuit;
  const lengths = suitLengths(retained);
  const buriedPoints = discarded.filter(isPointCard).length;
  const buriedTrumps = discarded.filter(
    (card) => isStandardCard(card) && card.suit === trump
  ).length;
  const retainedTrumps = lengths[trump];
  const countRank = (rank: Rank) =>
    discarded.filter((card) => isStandardCard(card) && card.rank === rank).length;
  const buriedJoker = discarded.some((card) => card.type === "joker");
  const buriedOruma = discarded.some((card) => card.id === view.specialCards.orumaCardId);
  const buriedSei = discarded.some((card) => card.id === view.specialCards.seiJackCardId);
  const buriedUra = discarded.some((card) => card.id === view.specialCards.uraJackCardId);
  const buriedYoro = discarded.some((card) => card.id === view.specialCards.yoromekiCardId);
  const buriedCalled = discarded.some((card) => card.id === view.adjutant?.calledCardId);
  const retainedPoints = retained.filter(isPointCard).length;
  const retainedHighs = retained.filter(isHighCard).length;
  const lengthValues = suits.map((suit) => lengths[suit]);
  const voids = lengthValues.filter((length) => length === 0).length;
  const singletons = lengthValues.filter((length) => length === 1).length;
  const doubletons = lengthValues.filter((length) => length === 2).length;
  const longest = Math.max(...lengthValues);
  const shortestNonzero = Math.min(...lengthValues.filter((length) => length > 0));
  const nonTrumpVoids = suits.filter((suit) => suit !== trump && lengths[suit] === 0).length;
  let sameSuitPairs = 0;
  for (let left = 0; left < discarded.length; left += 1) {
    for (let right = left + 1; right < discarded.length; right += 1) {
      const leftCard = discarded[left];
      const rightCard = discarded[right];
      if (isStandardCard(leftCard) && isStandardCard(rightCard) && leftCard.suit === rightCard.suit) {
        sameSuitPairs += 1;
      }
    }
  }
  const allSameSuit = discarded.every(
    (card) => isStandardCard(card) && isStandardCard(discarded[0]) && card.suit === discarded[0].suit
  );
  const kittyBuried = discarded.filter((card) => kittyCardIds.has(card.id)).length;
  const target = (view.contract.targetPointCards - 13) / 6;
  const regularDiscards = discarded.filter(
    (card) =>
      card.type !== "joker" &&
      card.id !== view.specialCards.orumaCardId &&
      card.id !== view.specialCards.seiJackCardId &&
      card.id !== view.specialCards.uraJackCardId &&
      card.id !== view.specialCards.yoromekiCardId
  );
  const features = [
    1,
    buriedPoints / 3,
    buriedTrumps / 3,
    retainedTrumps / 10,
    discarded.reduce((sum, card) => sum + aiRank(card), 0) / 60,
    countRank("A") / 3,
    countRank("10") / 3,
    countRank("J") / 3,
    countRank("Q") / 3,
    countRank("K") / 3,
    flag(buriedJoker),
    flag(buriedOruma),
    flag(buriedSei),
    flag(buriedUra),
    flag(buriedYoro),
    flag(buriedCalled),
    retainedPoints / 10,
    retainedHighs / 10,
    retainedTrumps / 10,
    ...lengthValues.map((length) => length / 10),
    ...lengthValues.map((length) => flag(length === 0)),
    ...lengthValues.map((length) => flag(length === 1)),
    ...lengthValues.map((length) => flag(length === 2)),
    voids / 4,
    singletons / 4,
    doubletons / 4,
    longest / 10,
    shortestNonzero / 10,
    nonTrumpVoids / 3,
    sameSuitPairs / 3,
    flag(allSameSuit),
    kittyBuried / 3,
    (3 - kittyBuried) / 3,
    target,
    (buriedPoints / 3) * target,
    (buriedTrumps / 3) * target,
    (retainedTrumps / 10) * target,
    flag(buriedCalled) * target,
    (kittyBuried / 3) * (buriedPoints / 3),
    (kittyBuried / 3) * (buriedTrumps / 3),
    regularDiscards.reduce((sum, card) => sum + aiRank(card), 0) / 60,
    regularDiscards.filter((card) => isStandardCard(card) && card.suit === trump).length / 3,
    target * flag(buriedOruma),
    target * flag(buriedSei),
    target * flag(buriedUra),
    target * flag(buriedJoker),
    target * flag(buriedYoro),
    flag(buriedYoro && trump === "hearts")
  ];
  assertFeatureCount("exchange", features, PARAMETERIZED_EXCHANGE_FEATURE_COUNT);
  return features;
}

function getSelfHand(
  observation: Pick<PlayerObservation, "playerId" | "view">
): readonly Card[] {
  const hand = observation.view.players.find((player) => player.id === observation.playerId)?.hand;
  if (hand === undefined) {
    throw new Error(`Visible hand is missing for ${observation.playerId}.`);
  }
  return hand;
}

function suitLengths(cards: readonly Card[]): Record<Suit, number> {
  const result: Record<Suit, number> = { spades: 0, hearts: 0, diamonds: 0, clubs: 0 };
  for (const card of cards) {
    if (isStandardCard(card)) {
      result[card.suit] += 1;
    }
  }
  return result;
}

function aiRank(card: Card): number {
  return isStandardCard(card) ? getAiRankValue(card.rank) : 0;
}

function isHighCard(card: Card): boolean {
  return isStandardCard(card) && highRanks.has(card.rank);
}

function flag(value: boolean): number {
  return value ? 1 : 0;
}

function dot(features: readonly number[], weights: readonly number[]): number {
  return features.reduce((sum, value, index) => sum + value * weights[index], 0);
}

function compareCardIds(left: string, right: string): number {
  return (cardOrder.get(left) ?? Number.MAX_SAFE_INTEGER) -
    (cardOrder.get(right) ?? Number.MAX_SAFE_INTEGER);
}

function combinations<T>(values: readonly T[], count: number): readonly (readonly T[])[] {
  if (count === 0) {
    return [[]];
  }
  const result: (readonly T[])[] = [];
  for (let index = 0; index <= values.length - count; index += 1) {
    for (const tail of combinations(values.slice(index + 1), count - 1)) {
      result.push([values[index], ...tail]);
    }
  }
  return result;
}

function assertFiniteWeights(label: string, weights: readonly number[], expected: number): void {
  if (weights.length !== expected) {
    throw new Error(`Parameterized ${label} policy expected ${expected} weights, got ${weights.length}.`);
  }
  if (weights.some((weight) => !Number.isFinite(weight))) {
    throw new Error(`Parameterized ${label} policy weights must all be finite.`);
  }
}

function assertFeatureCount(label: string, features: readonly number[], expected: number): void {
  if (features.length !== expected) {
    throw new Error(`Parameterized ${label} feature count mismatch: expected ${expected}, got ${features.length}.`);
  }
}
