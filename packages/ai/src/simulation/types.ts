import type {
  GameAction,
  GamePhase,
  GameResult,
  PlayerId
} from "@napoleon/game-core";
import type { Agent, PlayerObservation } from "../types.js";

export interface AutomatedAgentContext {
  playerId: PlayerId;
  playerIndex: number;
  rng: () => number;
}

export interface RunAutomatedGameOptions {
  seed: number;
  createAgent: (context: AutomatedAgentContext) => Agent;
  playerIds?: readonly PlayerId[];
  maxDecisionSteps?: number;
}

export type ActualHands = Readonly<Record<PlayerId, readonly string[]>>;

export interface ActualCardState {
  hands: ActualHands;
  unusedCardIds: readonly string[];
  excludedCardIds: readonly string[];
  awardedPointCardIds: Readonly<Record<PlayerId, readonly string[]>>;
  currentTrickCardIds: readonly string[];
  completedTrickCardIds: readonly string[];
}

export interface DecisionRecord {
  step: number;
  playerId: PlayerId;
  phase: GamePhase;
  trickNumber: number;
  observation: PlayerObservation;
  legalActions: readonly GameAction[];
  action: GameAction;
  actualHands: ActualHands;
  actualState: ActualCardState;
  handCounts: Readonly<Record<PlayerId, number>>;
}

export interface AutomatedGameRecord {
  schemaVersion: 1;
  seed: number;
  playerIds: readonly PlayerId[];
  initialHands: ActualHands;
  initialActualState: ActualCardState;
  decisions: readonly DecisionRecord[];
  result: GameResult;
}
