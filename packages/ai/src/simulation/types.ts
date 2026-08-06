import type {
  GameAction,
  GamePhase,
  GameResult,
  PlayerId,
  WinningTeam
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

export interface EvaluationAgentContext extends AutomatedAgentContext {
  agentName: string;
  sourceAgentIndex: number;
  seatIndex: number;
  rotationOffset: number;
  evaluationSeed: number;
  evaluationGameIndex: number;
}

export interface EvaluationAgentDefinition {
  name: string;
  createAgent: (context: EvaluationAgentContext) => Agent;
}

export interface RunEvaluationOptions {
  startSeed: number;
  gameCount: number;
  agents: readonly EvaluationAgentDefinition[];
  playerIds?: readonly PlayerId[];
  rotationOffsets?: readonly number[];
  maxDecisionSteps?: number;
}

export type EvaluationSeatRole = "napoleon" | "adjutant" | "alliance" | "unknown";

export interface EvaluationSeatAssignment {
  playerId: PlayerId;
  seatIndex: number;
  agentName: string;
  sourceAgentIndex: number;
  role: EvaluationSeatRole;
}

export interface EvaluationContractSummary {
  napoleonPlayerId: PlayerId;
  targetPointCards: number;
  adjutantPlayerId: PlayerId | null;
}

export interface EvaluationPointCardSummary {
  napoleonTeam: number;
  alliance: number;
}

export interface CompletedEvaluationGameRecord {
  schemaVersion: 1;
  status: "completed";
  gameIndex: number;
  seed: number;
  rotationOffset: number;
  playerIds: readonly PlayerId[];
  seats: readonly EvaluationSeatAssignment[];
  contract: EvaluationContractSummary;
  pointCards: EvaluationPointCardSummary;
  winner: WinningTeam;
  contractSucceeded: boolean;
  result: GameResult;
}

export interface FailedEvaluationGameRecord {
  schemaVersion: 1;
  status: "failed";
  gameIndex: number;
  seed: number;
  rotationOffset: number;
  playerIds: readonly PlayerId[];
  seats: readonly EvaluationSeatAssignment[];
  failureReason: string;
}

export type EvaluationGameRecord =
  | CompletedEvaluationGameRecord
  | FailedEvaluationGameRecord;

export interface EvaluationRunRecord {
  schemaVersion: 1;
  startSeed: number;
  endSeed: number;
  gameCount: number;
  rotationOffsets: readonly number[];
  playerIds: readonly PlayerId[];
  games: readonly EvaluationGameRecord[];
  completedCount: number;
  failedCount: number;
}
