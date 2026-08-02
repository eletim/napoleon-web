import {
  createDeck,
  determineCurrentWinningPlayer,
  isPointCard
} from "@napoleon/game-core";
import type { Card, GameAction, PlayedCard, PlayerId, PlayerView, Suit } from "@napoleon/game-core";
import { evaluateCardForTrump } from "./cardEvaluation.js";
import { NoLegalActionsError } from "./errors.js";
import { selectRandom } from "./random.js";

const epsilon = 1e-9;

export function estimateLeadWinProbability(cardScore: number): number {
  if (cardScore < 10) {
    return 0.05;
  }

  if (cardScore < 20) {
    return 0.1;
  }

  if (cardScore < 30) {
    return 0.2;
  }

  if (cardScore < 40) {
    return 0.35;
  }

  if (cardScore < 50) {
    return 0.55;
  }

  if (cardScore < 55) {
    return 0.7;
  }

  if (cardScore < 60) {
    return 0.8;
  }

  return 0.9;
}

export function adjustTeamWinProbability(
  baseProbability: number,
  remainingPlayers: number,
  isTeamCurrentlyWinning: boolean
): number {
  const adjusted = isTeamCurrentlyWinning
    ? 1 - ((1 - baseProbability) * remainingPlayers) / 4
    : (baseProbability * remainingPlayers) / 4;

  return Math.min(1, Math.max(0, adjusted));
}

export function calculateUsedCardValue(cardScore: number): number {
  return (cardScore - 20) / 30;
}

export function selectPlayAction(
  playerId: PlayerId,
  view: PlayerView,
  legalActions: readonly GameAction[],
  rng: () => number
): GameAction {
  const playActions = legalActions.filter((action) => action.type === "play-card");

  if (playActions.length === 0) {
    throw new NoLegalActionsError(playerId);
  }

  if (playActions.length === 1) {
    return playActions[0];
  }

  const selfHand = getSelfHand(view, playerId);
  const trumpSuit = getTrumpSuit(view);
  const scored = playActions.map((action) => {
    const card = selfHand.find((candidate) => candidate.id === action.cardId);

    if (card === undefined) {
      throw new NoLegalActionsError(playerId);
    }

    return {
      action,
      value: evaluatePlayAction(playerId, view, card, trumpSuit)
    };
  });
  const maxValue = Math.max(...scored.map((candidate) => candidate.value));
  const bestActions = scored
    .filter((candidate) => Math.abs(candidate.value - maxValue) < epsilon)
    .map((candidate) => candidate.action);

  return selectRandom(bestActions, rng);
}

function evaluatePlayAction(
  playerId: PlayerId,
  view: PlayerView,
  card: Card,
  trumpSuit: Suit
): number {
  const cardScore = evaluateCardForTrump(card, trumpSuit, view.specialCards);
  const baseProbability = estimateLeadWinProbability(cardScore);
  const playedCard: PlayedCard = { playerId, card };
  const provisionalTrick = [...view.currentTrick, playedCard];
  const provisionalWinnerId = determineCurrentWinningPlayer(
    provisionalTrick,
    { trumpSuit },
    { trickNumber: view.trickNumber }
  );
  const remainingPlayers = Math.max(0, view.players.length - provisionalTrick.length);
  const teamWinProbability = adjustTeamWinProbability(
    baseProbability,
    remainingPlayers,
    isSelfTeamPlayer(provisionalWinnerId, playerId, view)
  );
  const expectedPointCardsInTrick = calculateExpectedPointCardsInTrick(view, card);
  const usedCardValue = calculateUsedCardValue(cardScore);

  return (2 * teamWinProbability - 1) * expectedPointCardsInTrick - usedCardValue;
}

function calculateExpectedPointCardsInTrick(view: PlayerView, candidateCard: Card): number {
  const knownCardIds = new Set<string>();
  const self = view.players.find((player) => player.id === view.selfId);

  for (const card of self?.hand ?? []) {
    knownCardIds.add(card.id);
  }

  for (const playedCard of view.currentTrick) {
    knownCardIds.add(playedCard.card.id);
  }

  for (const player of view.players) {
    for (const card of player.capturedPointCards) {
      knownCardIds.add(card.id);
    }
  }

  knownCardIds.add(candidateCard.id);

  const unknownCards = createDeck().filter((card) => !knownCardIds.has(card.id));
  const unknownPointCardCount = unknownCards.filter(isPointCard).length;
  const remainingPlayers = Math.max(0, view.players.length - (view.currentTrick.length + 1));
  const currentPointCardCount =
    view.currentTrick.filter((playedCard) => isPointCard(playedCard.card)).length +
    (isPointCard(candidateCard) ? 1 : 0);

  if (unknownCards.length === 0) {
    return currentPointCardCount;
  }

  return currentPointCardCount + (remainingPlayers * unknownPointCardCount) / unknownCards.length;
}

function getSelfHand(view: PlayerView, playerId: PlayerId): readonly Card[] {
  const self = view.players.find((player) => player.id === playerId);

  if (self?.hand === undefined) {
    throw new NoLegalActionsError(playerId);
  }

  return self.hand;
}

function getTrumpSuit(view: PlayerView): Suit {
  const trumpSuit = view.contract?.trumpSuit ?? view.trumpSuit;

  if (trumpSuit === null) {
    throw new NoLegalActionsError(view.selfId);
  }

  return trumpSuit;
}

function isSelfTeamPlayer(candidatePlayerId: PlayerId, selfId: PlayerId, view: PlayerView): boolean {
  if (candidatePlayerId === selfId) {
    return true;
  }

  const napoleonPlayerId = view.contract?.napoleonPlayerId;

  if (napoleonPlayerId === undefined) {
    return false;
  }

  const revealedAdjutantPlayerId = view.adjutant?.revealedPlayerId ?? null;
  const self = view.players.find((player) => player.id === selfId);
  const calledCardId = view.adjutant?.calledCardId;
  const selfHasCalledCard =
    calledCardId !== undefined && (self?.hand ?? []).some((card) => card.id === calledCardId);

  if (selfId === napoleonPlayerId) {
    return candidatePlayerId === revealedAdjutantPlayerId;
  }

  if (selfHasCalledCard) {
    return candidatePlayerId === napoleonPlayerId;
  }

  if (revealedAdjutantPlayerId !== null) {
    return candidatePlayerId !== napoleonPlayerId && candidatePlayerId !== revealedAdjutantPlayerId;
  }

  return false;
}
