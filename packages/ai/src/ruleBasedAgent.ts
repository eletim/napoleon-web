import { createDeck, isStandardCard } from "@napoleon/game-core";
import type { Card, GameAction, PlayerId, PlayerView, Suit } from "@napoleon/game-core";
import {
  countStandardCardsOfSuit,
  createSpecialCardsForTrump,
  evaluateCardForTrump,
  getAiRankValue
} from "./cardEvaluation.js";
import { NoLegalActionsError } from "./errors.js";
import { selectPlayAction } from "./playStrategy.js";
import { selectRandom } from "./random.js";
import type { Agent, PlayerObservation } from "./types.js";

const suitEvaluationOrder: readonly Suit[] = ["spades", "hearts", "diamonds", "clubs"];
const maxAiBidTargetPointCards = 20;
const minAiBidTargetPointCards = 13;
const epsilon = 1e-9;

export class RuleBasedAgent implements Agent {
  private readonly rng: () => number;

  constructor(rng: () => number = Math.random) {
    this.rng = rng;
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    switch (observation.view.phase) {
      case "bidding":
        return selectBiddingAction(observation);
      case "choosing-adjutant":
        return selectAdjutantAction(observation, this.rng);
      case "exchanging":
        return selectDiscardAction(observation, this.rng);
      case "playing":
        return selectPlayAction(
          observation.playerId,
          observation.view,
          observation.legalActions,
          this.rng
        );
      case "finished":
        throw new NoLegalActionsError(observation.playerId);
    }
  }
}

export function getBidLimitForScore(score: number): number | null {
  return score < 200
    ? null
    : Math.min(maxAiBidTargetPointCards, minAiBidTargetPointCards + Math.floor((score - 200) / 30));
}

export function evaluateHandForTrump(hand: readonly Card[], trumpSuit: Suit): number {
  const specialCards = createSpecialCardsForTrump(trumpSuit);
  return hand.reduce(
    (total, card) => total + evaluateCardForTrump(card, trumpSuit, specialCards),
    0
  );
}

function selectBiddingAction(observation: PlayerObservation): GameAction {
  const selfHand = getSelfHand(observation.view, observation.playerId);
  const bestSuit = selectBestTrumpSuit(selfHand);
  const bidLimit = getBidLimitForScore(evaluateHandForTrump(selfHand, bestSuit));
  const passAction = observation.legalActions.find((action) => action.type === "pass");

  if (bidLimit === null) {
    return passOrThrow(passAction, observation.playerId);
  }

  const currentTarget = observation.view.bidding?.highestBid?.targetPointCards ?? null;
  const neededTarget = currentTarget === null ? minAiBidTargetPointCards : currentTarget + 1;

  if (neededTarget > bidLimit) {
    return passOrThrow(passAction, observation.playerId);
  }

  const bidAction = observation.legalActions.find(
    (action) =>
      action.type === "bid" &&
      action.suit === bestSuit &&
      action.targetPointCards === neededTarget
  );

  return bidAction ?? passOrThrow(passAction, observation.playerId);
}

function selectBestTrumpSuit(hand: readonly Card[]): Suit {
  return suitEvaluationOrder.reduce((bestSuit, candidateSuit) => {
    const bestScore = evaluateHandForTrump(hand, bestSuit);
    const candidateScore = evaluateHandForTrump(hand, candidateSuit);
    return candidateScore > bestScore ? candidateSuit : bestSuit;
  }, suitEvaluationOrder[0]);
}

function selectAdjutantAction(observation: PlayerObservation, rng: () => number): GameAction {
  if (observation.view.adjutantChoiceRequirement === null) {
    throw new NoLegalActionsError(observation.playerId);
  }

  const trumpSuit = observation.view.contract?.trumpSuit ?? observation.view.trumpSuit;

  if (trumpSuit === null) {
    throw new NoLegalActionsError(observation.playerId);
  }

  const selfHand = getSelfHand(observation.view, observation.playerId);
  const legalAdjutantActions = observation.legalActions.filter(isChooseAdjutantAction);

  if (legalAdjutantActions.length === 0) {
    throw new NoLegalActionsError(observation.playerId);
  }

  const legalAdjutantCardIds = new Set(legalAdjutantActions.map((action) => action.cardId));
  const cardId = selectAdjutantCardId(
    selfHand,
    trumpSuit,
    observation.view,
    rng,
    legalAdjutantCardIds
  );

  if (cardId === null) {
    return selectRandom(legalAdjutantActions, rng);
  }

  const selectedAction = legalAdjutantActions.find((action) => action.cardId === cardId);

  if (selectedAction === undefined) {
    throw new NoLegalActionsError(observation.playerId);
  }

  return selectedAction;
}

export function selectAdjutantCardId(
  selfHand: readonly Card[],
  trumpSuit: Suit,
  view: Pick<PlayerView, "specialCards">,
  _rng: () => number,
  legalAdjutantCardIds?: ReadonlySet<string>
): string | null {
  const selfCardIds = new Set(selfHand.map((card) => card.id));
  const trumpCount = countStandardCardsOfSuit(selfHand, trumpSuit);
  const candidates = createDeck()
    .filter(isStandardCard)
    .filter(
      (card) =>
        (legalAdjutantCardIds === undefined || legalAdjutantCardIds.has(card.id)) &&
        isStandardAdjutantCandidate(card, trumpSuit, view, selfCardIds)
    )
    .map((card) => ({
      card,
      value: evaluateAdjutantCandidate(card, view, trumpCount)
    }));

  if (candidates.length === 0) {
    return null;
  }

  const maxValue = Math.max(...candidates.map((candidate) => candidate.value));
  const best = candidates
    .filter((candidate) => Math.abs(candidate.value - maxValue) < epsilon)
    .sort((left, right) => compareAdjutantTie(left.card, right.card, view));

  return best[0].card.id;
}

function isChooseAdjutantAction(
  action: GameAction
): action is Extract<GameAction, { type: "choose-adjutant" }> {
  return action.type === "choose-adjutant";
}

function isStandardAdjutantCandidate(
  card: Card,
  trumpSuit: Suit,
  view: Pick<PlayerView, "specialCards">,
  selfCardIds: ReadonlySet<string>
): boolean {
  if (!isStandardCard(card) || selfCardIds.has(card.id)) {
    return false;
  }

  if (card.id === view.specialCards.yoromekiCardId) {
    return false;
  }

  return (
    card.id === view.specialCards.orumaCardId ||
    card.id === view.specialCards.seiJackCardId ||
    card.id === view.specialCards.uraJackCardId ||
    card.suit === trumpSuit
  );
}

function evaluateAdjutantCandidate(
  card: Card,
  view: Pick<PlayerView, "specialCards">,
  trumpCount: number
): number {
  if (card.id === view.specialCards.orumaCardId) {
    return 60;
  }

  if (card.id === view.specialCards.seiJackCardId) {
    return 55;
  }

  if (card.id === view.specialCards.uraJackCardId) {
    return 50.5 + trumpCount;
  }

  if (!isStandardCard(card)) {
    return 0;
  }

  return 30 + getAiRankValue(card.rank);
}

function compareAdjutantTie(
  left: Card,
  right: Card,
  view: Pick<PlayerView, "specialCards">
): number {
  const leftPriority = getAdjutantTiePriority(left, view);
  const rightPriority = getAdjutantTiePriority(right, view);

  if (leftPriority !== rightPriority) {
    return rightPriority - leftPriority;
  }

  if (isStandardCard(left) && isStandardCard(right) && left.rank !== right.rank) {
    return getAiRankValue(right.rank) - getAiRankValue(left.rank);
  }

  return left.id.localeCompare(right.id);
}

function getAdjutantTiePriority(
  card: Card,
  view: Pick<PlayerView, "specialCards">
): number {
  if (card.id === view.specialCards.orumaCardId) {
    return 4;
  }

  if (card.id === view.specialCards.seiJackCardId) {
    return 3;
  }

  if (card.id === view.specialCards.uraJackCardId) {
    return 2;
  }

  return 1;
}

function selectDiscardAction(observation: PlayerObservation, rng: () => number): GameAction {
  const discardCount = observation.view.exchangeRequirement?.discardCount;

  if (discardCount === undefined) {
    throw new NoLegalActionsError(observation.playerId);
  }

  const selfHand = getSelfHand(observation.view, observation.playerId);
  const trumpSuit = observation.view.contract?.trumpSuit ?? observation.view.trumpSuit;

  if (trumpSuit === null) {
    throw new NoLegalActionsError(observation.playerId);
  }

  return {
    type: "discard-cards",
    playerId: observation.playerId,
    cardIds: selectDiscardCardIds(selfHand, discardCount, trumpSuit, observation.view, rng)
  };
}

export function selectDiscardCardIds(
  hand: readonly Card[],
  discardCount: number,
  trumpSuit: Suit,
  view: Pick<PlayerView, "specialCards">,
  rng: () => number
): readonly string[] {
  const combinations = getCombinations(hand, discardCount);

  if (combinations.length === 0) {
    throw new Error("No discard combinations are available.");
  }

  const scored = combinations.map((discardCards) => {
    const discardIds = new Set(discardCards.map((card) => card.id));
    const keptScore = hand
      .filter((card) => !discardIds.has(card.id))
      .reduce((total, card) => total + evaluateCardForTrump(card, trumpSuit, view.specialCards), 0);

    return {
      cards: discardCards,
      keptScore
    };
  });
  const maxScore = Math.max(...scored.map((candidate) => candidate.keptScore));
  const best = scored.filter((candidate) => Math.abs(candidate.keptScore - maxScore) < epsilon);

  return selectRandom(best, rng).cards.map((card) => card.id);
}

function getCombinations(cards: readonly Card[], count: number): readonly (readonly Card[])[] {
  if (count === 0) {
    return [[]];
  }

  if (cards.length < count) {
    return [];
  }

  const combinations: (readonly Card[])[] = [];

  for (let index = 0; index <= cards.length - count; index += 1) {
    const head = cards[index];
    const tails = getCombinations(cards.slice(index + 1), count - 1);

    for (const tail of tails) {
      combinations.push([head, ...tail]);
    }
  }

  return combinations;
}

function passOrThrow(passAction: GameAction | undefined, playerId: PlayerId): GameAction {
  if (passAction === undefined) {
    throw new NoLegalActionsError(playerId);
  }

  return passAction;
}

function getSelfHand(view: PlayerView, playerId: PlayerId): readonly Card[] {
  const self = view.players.find((player) => player.id === playerId);

  if (self?.hand === undefined) {
    throw new NoLegalActionsError(playerId);
  }

  return self.hand;
}
