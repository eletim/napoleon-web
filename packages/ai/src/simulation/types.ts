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
  agentOrders?: readonly (readonly number[])[];
  maxDecisionSteps?: number;
}

export type EvaluationSeatRole = "napoleon" | "adjutant" | "alliance" | "starter" | "all-pass-other" | "unknown";

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
  resultType: GameResult["resultType"];
  gameIndex: number;
  seed: number;
  rotationOffset: number;
  playerIds: readonly PlayerId[];
  seats: readonly EvaluationSeatAssignment[];
  contract: EvaluationContractSummary | null;
  pointCards: EvaluationPointCardSummary | null;
  winner: WinningTeam | null;
  contractSucceeded: boolean | null;
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

export interface EvaluationRateSummary {
  numerator: number;
  denominator: number;
  rate: number | null;
  confidenceInterval: EvaluationConfidenceInterval;
}

export type EvaluationConfidenceIntervalMethod =
  | "wilson"
  | "newcombe-wilson"
  | "normal";

export interface EvaluationConfidenceInterval {
  level: 0.95;
  method: EvaluationConfidenceIntervalMethod;
  lower: number | null;
  upper: number | null;
}

export interface EvaluationGameCountSummary {
  total: number;
  completed: number;
  failed: number;
}

export interface EvaluationFailureSummary {
  total: number;
  byReason: Readonly<Record<string, number>>;
}

export interface EvaluationPerformanceSummary {
  games: EvaluationGameCountSummary;
  sampleCount: number;
  wins: number;
  losses: number;
  winRate: EvaluationRateSummary;
  contractSuccesses: number;
  contractSuccessRate: EvaluationRateSummary;
  averagePointCards: number | null;
  failures: EvaluationFailureSummary;
}

export interface EvaluationRolePerformanceSummary extends EvaluationPerformanceSummary {
  role: Exclude<EvaluationSeatRole, "unknown">;
}

export interface EvaluationSeatPerformanceSummary extends EvaluationPerformanceSummary {
  seatIndex: number;
}

export interface EvaluationAgentPerformanceSummary extends EvaluationPerformanceSummary {
  sourceAgentIndex: number;
  agentName: string;
  roleResults: readonly EvaluationRolePerformanceSummary[];
  seatResults: readonly EvaluationSeatPerformanceSummary[];
  comparison: EvaluationComparisonSummary;
}

export interface EvaluationComparisonSummary {
  winRateDelta: number | null;
  winRateDeltaConfidenceInterval: EvaluationConfidenceInterval;
  contractSuccessRateDelta: number | null;
  contractSuccessRateDeltaConfidenceInterval: EvaluationConfidenceInterval;
  averagePointCardsDelta: number | null;
  averagePointCardsDeltaConfidenceInterval: EvaluationConfidenceInterval;
}

export interface EvaluationReportSummary extends EvaluationPerformanceSummary {
  expectedGameCount: number;
}

export interface EvaluationReport {
  schemaVersion: 1;
  sourceSchemaVersion: 1;
  summary: EvaluationReportSummary;
  agents: readonly EvaluationAgentPerformanceSummary[];
  seats: readonly EvaluationSeatPerformanceSummary[];
  roles: readonly EvaluationRolePerformanceSummary[];
}
