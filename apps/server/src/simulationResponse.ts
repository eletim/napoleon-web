import type { DecisionRecord, AutomatedGameRecord } from "@napoleon/ai";
import type { GameAction, PlayerView } from "@napoleon/game-core";
import type {
  PublicGameAction,
  PublicGamePhase,
  PublicSimulationDecision,
  PublicSimulationObservedPlayer,
  PublicSimulationObservation,
  PublicSimulationSummary,
  RunAutomatedSimulationResponse
} from "@napoleon/protocol";
import { toPublicCard, toPublicGameResult, toPublicPlayedCard } from "./publicState.js";

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
    action: toPublicGameAction(decision.action),
    legalActionCount: decision.legalActions.length,
    handCounts: decision.handCounts,
    actualHands: decision.actualHands
  };
}

function toPublicSimulationObservation(view: PlayerView): PublicSimulationObservation {
  return {
    selfId: view.selfId,
    players: view.players.map(
      (player): PublicSimulationObservedPlayer => ({
        id: player.id,
        handCount: player.handCount,
        ...(player.hand === undefined ? {} : { hand: player.hand.map(toPublicCard) })
      })
    ),
    phase: view.phase,
    trumpSuit: view.trumpSuit,
    contract: view.contract,
    adjutant:
      view.adjutant === null
        ? null
        : {
            calledCardId: view.adjutant.calledCardId,
            revealedPlayerId: view.adjutant.revealedPlayerId
          },
    currentPlayerId: view.currentPlayerId,
    currentTrick: view.currentTrick.map(toPublicPlayedCard),
    completedTrickCount: view.completedTrickCount,
    trickNumber: view.trickNumber,
    isTrickComplete: view.isTrickComplete,
    isGameOver: view.isGameOver
  };
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
