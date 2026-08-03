import type { ActualCardState, DecisionRecord, AutomatedGameRecord } from "@napoleon/ai";
import type { CompletedTrick, GameAction, PlayerView } from "@napoleon/game-core";
import type {
  PublicGameAction,
  PublicGamePhase,
  PublicSimulationActualState,
  PublicSimulationCompletedTrick,
  PublicSimulationDecision,
  PublicSimulationObservedPlayer,
  PublicSimulationObservation,
  PublicSimulationSummary,
  RunAutomatedSimulationResponse
} from "@napoleon/protocol";
import {
  toPublicBiddingState,
  toPublicCard,
  toPublicGameEvent,
  toPublicGameResult,
  toPublicPlayedCard
} from "./publicState.js";

const gamePhases: readonly PublicGamePhase[] = [
  "bidding",
  "exchanging",
  "choosing-adjutant",
  "playing",
  "finished"
];

export function toPublicSimulationResponse(
  record: AutomatedGameRecord
): RunAutomatedSimulationResponse {
  const decisions = record.decisions.map(toPublicSimulationDecision);

  return {
    schemaVersion: record.schemaVersion,
    seed: record.seed,
    playerIds: record.playerIds,
    initialHands: record.initialHands,
    initialActualState: toPublicSimulationActualState(record.initialActualState),
    decisions,
    summary: createSummary(record.playerIds, decisions),
    result: toPublicGameResult(record.result)
  };
}

function toPublicSimulationDecision(decision: DecisionRecord): PublicSimulationDecision {
  return {
    step: decision.step,
    playerId: decision.playerId,
    phase: decision.phase,
    trickNumber: decision.trickNumber,
    observation: toPublicSimulationObservation(decision.observation.view),
    legalActions: decision.legalActions.map(toPublicGameAction),
    action: toPublicGameAction(decision.action),
    legalActionCount: decision.legalActions.length,
    handCounts: decision.handCounts,
    actualHands: decision.actualHands,
    actualState: toPublicSimulationActualState(decision.actualState)
  };
}

function toPublicSimulationObservation(view: PlayerView): PublicSimulationObservation {
  return {
    selfId: view.selfId,
    players: view.players.map(
      (player): PublicSimulationObservedPlayer => ({
        id: player.id,
        handCount: player.handCount,
        capturedPointCards: player.capturedPointCards.map((card) => ({
          type: "standard",
          id: card.id,
          suit: card.suit,
          rank: card.rank
        })),
        ...(player.hand === undefined ? {} : { hand: player.hand.map(toPublicCard) })
      })
    ),
    phase: view.phase,
    trumpSuit: view.trumpSuit,
    contract: view.contract,
    specialCards: {
      orumaCardId: view.specialCards.orumaCardId,
      yoromekiCardId: view.specialCards.yoromekiCardId,
      seiJackCardId: view.specialCards.seiJackCardId,
      uraJackCardId: view.specialCards.uraJackCardId
    },
    adjutant:
      view.adjutant === null
        ? null
        : {
            calledCardId: view.adjutant.calledCardId,
            revealedPlayerId: view.adjutant.revealedPlayerId
          },
    latestEvent: view.latestEvent === null ? null : toPublicGameEvent(view.latestEvent),
    bidding: view.bidding === null ? null : toPublicBiddingState(view.bidding),
    exchangeRequirement:
      view.exchangeRequirement === null
        ? null
        : {
            discardCount: view.exchangeRequirement.discardCount
          },
    adjutantChoiceRequirement:
      view.adjutantChoiceRequirement === null
        ? null
        : {
            jokerAllowed: view.adjutantChoiceRequirement.jokerAllowed
          },
    currentPlayerId: view.currentPlayerId,
    currentTrick: view.currentTrick.map(toPublicPlayedCard),
    completedTricks: view.completedTricks.map(toPublicSimulationCompletedTrick),
    completedTrickCount: view.completedTrickCount,
    trickNumber: view.trickNumber,
    isTrickComplete: view.isTrickComplete,
    isGameOver: view.isGameOver
  };
}

function toPublicSimulationCompletedTrick(
  trick: CompletedTrick
): PublicSimulationCompletedTrick {
  return {
    trickNumber: trick.trickNumber,
    winnerId: trick.winnerId,
    cards: trick.cards.map(toPublicPlayedCard)
  };
}

function toPublicSimulationActualState(
  actualState: ActualCardState
): PublicSimulationActualState {
  return {
    hands: copyCardIdRecord(actualState.hands),
    unusedCardIds: [...actualState.unusedCardIds],
    excludedCardIds: [...actualState.excludedCardIds],
    awardedPointCardIds: copyCardIdRecord(actualState.awardedPointCardIds),
    currentTrickCardIds: [...actualState.currentTrickCardIds],
    completedTrickCardIds: [...actualState.completedTrickCardIds]
  };
}

function copyCardIdRecord(
  record: Readonly<Record<string, readonly string[]>>
): Readonly<Record<string, readonly string[]>> {
  return Object.fromEntries(
    Object.entries(record).map(([playerId, cardIds]) => [playerId, [...cardIds]])
  );
}

function toPublicGameAction(action: GameAction): PublicGameAction {
  switch (action.type) {
    case "play-card":
      return {
        type: "play-card",
        cardId: action.cardId
      };
    case "bid":
      return {
        type: "bid",
        suit: action.suit,
        targetPointCards: action.targetPointCards
      };
    case "pass":
      return {
        type: "pass"
      };
    case "discard-cards":
      return {
        type: "discard-cards",
        cardIds: action.cardIds
      };
    case "choose-adjutant":
      return {
        type: "choose-adjutant",
        cardId: action.cardId
      };
  }
}

function createSummary(
  playerIds: readonly string[],
  decisions: readonly PublicSimulationDecision[]
): PublicSimulationSummary {
  const decisionCountByPlayer = Object.fromEntries(playerIds.map((playerId) => [playerId, 0]));
  const decisionCountByPhase = Object.fromEntries(gamePhases.map((phase) => [phase, 0])) as Record<
    PublicGamePhase,
    number
  >;
  const actionCountByType: Record<string, number> = {};

  for (const decision of decisions) {
    decisionCountByPlayer[decision.playerId] = (decisionCountByPlayer[decision.playerId] ?? 0) + 1;
    decisionCountByPhase[decision.phase] += 1;
    actionCountByType[decision.action.type] = (actionCountByType[decision.action.type] ?? 0) + 1;
  }

  return {
    totalDecisionCount: decisions.length,
    decisionCountByPlayer,
    decisionCountByPhase,
    actionCountByType
  };
}
