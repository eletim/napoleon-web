import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import {
  ConservativeBiddingAgent,
  RuleBasedAgent,
  runAutomatedGame
} from "@napoleon/ai";
import type {
  ActualCardState,
  Agent,
  AutomatedGameRecord,
  PlayerObservation
} from "@napoleon/ai";
import type { GameAction, GameResult, PlayerId, WinningTeam } from "@napoleon/game-core";
import {
  BIDDING_ACTION_COUNT,
  BIDDING_ENCODER_SCHEMA_VERSION,
  BIDDING_MODEL_INPUT_FEATURE_COUNT,
  BIDDING_MODEL_INPUT_SCHEMA_VERSION,
  ADJUTANT_ENCODER_SCHEMA_VERSION,
  ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
  ADJUTANT_MODEL_INPUT_SCHEMA_VERSION,
  CARD_COUNT,
  CARD_IDS,
  EXCHANGE_ENCODER_SCHEMA_VERSION,
  EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
  EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
  MODEL_INPUT_FEATURE_COUNT,
  MODEL_INPUT_SCHEMA_VERSION,
  PLAYER_COUNT,
  createAdjutantModelInput,
  createBiddingModelInput,
  createExchangeModelInput,
  createPlayingModelInput,
  createRelativePlayerOrder,
  decodeAdjutantAction,
  decodeBiddingAction,
  encodeAdjutantObservation,
  encodeBiddingObservation,
  encodeBiddingHistoryFromPublicActions,
  encodeExchangeObservation,
  encodePlayingObservation,
  getCardId
} from "@napoleon/ai-observation";
import type { EncodedExchangeObservation } from "@napoleon/ai-observation";
import {
  DATASET_FORMAT,
  MAX_SHARD_COUNT,
  RULE_BASED_AGENT_VERSION,
  UINT32_MAX
} from "./schema.js";
import type { DatasetGenerationProgress, DatasetShardManifest } from "./types.js";
import { createJsonlShardWriter, type JsonlShardWriter } from "./shardWriter.js";
import { calculateCardIdsSha256, serializeManifest } from "./serialization.js";

export const NON_PLAYING_BIDDING_RL_DATASET_SAMPLE_TYPE = "non-playing-bidding-rl-sample" as const;
export const NON_PLAYING_ADJUTANT_RL_DATASET_SAMPLE_TYPE = "non-playing-adjutant-rl-sample" as const;
export const NON_PLAYING_EXCHANGE_RL_DATASET_SAMPLE_TYPE = "non-playing-exchange-rl-sample" as const;
export const NON_PLAYING_RL_DATASET_SAMPLE_TYPE = NON_PLAYING_BIDDING_RL_DATASET_SAMPLE_TYPE;
export const NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION = 4 as const;
export const NON_PLAYING_RL_DATASET_SCHEMA_VERSION = 4 as const;
export const NON_PLAYING_RL_DATASET_GENERATOR_VERSION = 5 as const;
export const NON_PLAYING_RL_ROLLOUT_POLICY_TOPOLOGY = "candidate-x1-frozen-x4-v1" as const;
export const NON_PLAYING_RL_GAME_COUNT_UNIT = "logical-seeds" as const;
export const NON_PLAYING_RL_ROTATION_OFFSETS = [0, 1, 2, 3, 4] as const;
export const CONSERVATIVE_BIDDING_BASELINE_ID = "conservative-bidding-v1" as const;
export const RULE_BASED_BIDDING_BASELINE_ID = "rule-based-bidding-v1" as const;
export const FROZEN_BIDDING_OPPONENT_MIX_RULE_VERSION =
  "per-seat-seeded-rule-based-conservative-50-50-v1" as const;
export const NON_PLAYING_RL_REWARD_TYPE = "non-playing-terminal-role-reward" as const;
export const NON_PLAYING_RL_REWARD_VERSION = 3 as const;
export const NON_PLAYING_RL_REWARD_ID = "non-playing-terminal-role-reward-v3" as const;
export const NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_TYPE =
  "raw-reward-minus-game-player-mean" as const;
export const NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_VERSION = 1 as const;
export const NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_ID =
  "non-playing-terminal-role-reward-v3-minus-game-player-mean-v1" as const;
export const NON_PLAYING_RL_ALL_PASS_RULE_ID = "all-pass-immediate-zero-raw-terminal-reward-v1" as const;
export const NON_PLAYING_RL_SAMPLING_ALGORITHM = "masked-categorical" as const;
export const DEFAULT_NON_PLAYING_RL_TEMPERATURE = 1.0 as const;
export const NON_PLAYING_BIDDING_RL_PHASE_SCOPE = "bidding-only" as const;
export const NON_PLAYING_ADJUTANT_RL_PHASE_SCOPE = "adjutant-only" as const;
export const NON_PLAYING_EXCHANGE_RL_PHASE_SCOPE = "exchange-only" as const;
export const NON_PLAYING_RL_PHASE_SCOPE = NON_PLAYING_BIDDING_RL_PHASE_SCOPE;
export const NON_PLAYING_EXCHANGE_RL_DECISION_MODE = "sequential-card-v1" as const;

export type NonPlayingBiddingRlRole =
  | "napoleon"
  | "adjutant"
  | "citizen"
  | "napoleon-adjutant"
  | "all-pass-starter"
  | "all-pass-other";

export type FrozenBiddingOpponentPolicyType = "rule-based-bidding" | "conservative-bidding";

export interface FrozenBiddingOpponentPolicyMetadata {
  type: FrozenBiddingOpponentPolicyType;
  id: typeof RULE_BASED_BIDDING_BASELINE_ID | typeof CONSERVATIVE_BIDDING_BASELINE_ID;
}

export interface FrozenBiddingOpponentSeatAssignment {
  seed: number;
  rotationOffset: number;
  candidateSeatIndex: number;
  playerIndex: number;
  playerId: PlayerId;
  policy: FrozenBiddingOpponentPolicyMetadata;
}

export interface FrozenBiddingOpponentMixMetadata {
  type: "mixed-frozen-bidding";
  mixingRuleVersion: typeof FROZEN_BIDDING_OPPONENT_MIX_RULE_VERSION;
  selectionUnit: "game-seat";
  ruleBasedWeight: 0.5;
  conservativeWeight: 0.5;
  policies: {
    ruleBased: {
      type: "rule-based-bidding";
      id: typeof RULE_BASED_BIDDING_BASELINE_ID;
      version: typeof RULE_BASED_AGENT_VERSION;
    };
    conservative: {
      type: "conservative-bidding";
      id: typeof CONSERVATIVE_BIDDING_BASELINE_ID;
    };
  };
}

export interface NonPlayingBiddingRlPolicy {
  metadata: unknown;
  runtime?: PolicyRuntimeInfo;
  predictLogits: (modelInput: Float32Array | readonly number[]) => Promise<Float32Array>;
}

export interface NonPlayingAdjutantRlPolicy {
  metadata: unknown;
  runtime?: PolicyRuntimeInfo;
  predictLogits: (modelInput: Float32Array | readonly number[]) => Promise<Float32Array>;
}

export interface NonPlayingExchangeRlPolicy {
  metadata: unknown;
  runtime?: PolicyRuntimeInfo;
  predictLogits: (modelInput: Float32Array | readonly number[]) => Promise<Float32Array>;
}

export interface FixedPlayingPolicy {
  metadata: unknown;
  runtime?: PolicyRuntimeInfo;
  predictLogits: (modelInput: Float32Array | readonly number[]) => Promise<Float32Array>;
}

export interface PolicyRuntimeInfo {
  requestedInferenceDevice?: "cpu" | "auto" | "cuda";
  resolvedInferenceDevice?: "cpu" | "cuda";
  executionProvider?: "cpu" | "cuda";
}

export interface NonPlayingRlPolicyArtifactOptions {
  onnxPath: string;
  metadataPath: string;
  artifactId?: string;
}

export interface NonPlayingStandardBiddingRlOutcome {
  outcomeType: "standard";
  winner: WinningTeam;
  targetPointCards: number;
  napoleonPlayerId: PlayerId;
  actingPlayerRole: Exclude<NonPlayingBiddingRlRole, "all-pass-starter" | "all-pass-other">;
}

export interface NonPlayingAllPassBiddingRlOutcome {
  outcomeType: "all-pass";
  starterPlayerId: PlayerId;
  actingPlayerRole: Extract<NonPlayingBiddingRlRole, "all-pass-starter" | "all-pass-other">;
  actingPlayerPayoff: 0;
}

export type NonPlayingBiddingRlOutcome =
  | NonPlayingStandardBiddingRlOutcome
  | NonPlayingAllPassBiddingRlOutcome;

export interface NonPlayingBiddingRlSample {
  sampleType: typeof NON_PLAYING_BIDDING_RL_DATASET_SAMPLE_TYPE;
  schemaVersion: typeof NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION;
  seed: number;
  step: number;
  phase: "bidding";
  actingPlayerId: PlayerId;
  actingPlayerIndex: number;
  candidateSeatIndex: number;
  rotationOffset: number;
  relativePlayerIds: readonly PlayerId[];
  modelInput: readonly number[];
  legalBidMask: readonly number[];
  selectedActionIndex: number;
  behaviorLogProbability: number;
  rawTerminalReward: number;
  gameMeanRawTerminalReward: number;
  terminalReward: number;
  outcome: NonPlayingBiddingRlOutcome;
}

export interface NonPlayingAdjutantRlSample {
  sampleType: typeof NON_PLAYING_ADJUTANT_RL_DATASET_SAMPLE_TYPE;
  schemaVersion: typeof NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION;
  seed: number;
  step: number;
  phase: "choosing-adjutant";
  actingPlayerId: PlayerId;
  actingPlayerIndex: number;
  candidateSeatIndex: number;
  rotationOffset: number;
  relativePlayerIds: readonly PlayerId[];
  modelInput: readonly number[];
  legalAdjutantMask: readonly number[];
  selectedActionIndex: number;
  behaviorLogProbability: number;
  rawTerminalReward: number;
  gameMeanRawTerminalReward: number;
  terminalReward: number;
  outcome: NonPlayingBiddingRlOutcome;
}

export interface NonPlayingExchangeRlSample {
  sampleType: typeof NON_PLAYING_EXCHANGE_RL_DATASET_SAMPLE_TYPE;
  schemaVersion: typeof NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION;
  seed: number;
  step: number;
  phase: "exchanging";
  actingPlayerId: PlayerId;
  actingPlayerIndex: number;
  candidateSeatIndex: number;
  rotationOffset: number;
  relativePlayerIds: readonly PlayerId[];
  exchangeStepIndex: 0 | 1 | 2;
  remainingDiscardCount: 1 | 2 | 3;
  modelInput: readonly number[];
  legalDiscardCardMask: readonly number[];
  selectedActionIndex: number;
  behaviorLogProbability: number;
  rawTerminalReward: number;
  gameMeanRawTerminalReward: number;
  terminalReward: number;
  outcome: NonPlayingBiddingRlOutcome;
}

export interface NonPlayingRlDatasetManifest {
  datasetSchemaVersion: typeof NON_PLAYING_RL_DATASET_SCHEMA_VERSION;
  generatorVersion: typeof NON_PLAYING_RL_DATASET_GENERATOR_VERSION;
  format: typeof DATASET_FORMAT;
  sampleType:
    | typeof NON_PLAYING_BIDDING_RL_DATASET_SAMPLE_TYPE
    | typeof NON_PLAYING_ADJUTANT_RL_DATASET_SAMPLE_TYPE
    | typeof NON_PLAYING_EXCHANGE_RL_DATASET_SAMPLE_TYPE;
  sampleSchemaVersion: typeof NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION;
  phaseScope:
    | typeof NON_PLAYING_BIDDING_RL_PHASE_SCOPE
    | typeof NON_PLAYING_ADJUTANT_RL_PHASE_SCOPE
    | typeof NON_PLAYING_EXCHANGE_RL_PHASE_SCOPE;
  learnedPhases: readonly ["bidding"] | readonly ["choosing-adjutant"] | readonly ["exchanging"];
  ruleBasedPhases:
    | readonly ["choosing-adjutant", "exchanging"]
    | readonly ["bidding", "exchanging"]
    | readonly ["bidding", "choosing-adjutant"]
    | readonly [];
  fixedPhases:
    | readonly ["playing"]
    | readonly ["choosing-adjutant", "exchanging", "playing"];
  rolloutPolicyTopology: typeof NON_PLAYING_RL_ROLLOUT_POLICY_TOPOLOGY;
  gameCountUnit: typeof NON_PLAYING_RL_GAME_COUNT_UNIT;
  logicalSeedCount: number;
  actualGameCount: number;
  rotationOffsets: readonly number[];
  startSeed: number;
  endSeed: number;
  gameCount: number;
  sampleCount: number;
  gamesPerShard: number;
  shardCount: number;
  playerCount: 5;
  cardCount: 53;
  cardIds: readonly string[];
  cardIdsSha256: string;
  biddingEncoderSchemaVersion: typeof BIDDING_ENCODER_SCHEMA_VERSION;
  biddingModelInputSchemaVersion: typeof BIDDING_MODEL_INPUT_SCHEMA_VERSION;
  biddingModelInputFeatureCount: typeof BIDDING_MODEL_INPUT_FEATURE_COUNT;
  adjutantEncoderSchemaVersion?: typeof ADJUTANT_ENCODER_SCHEMA_VERSION;
  adjutantModelInputSchemaVersion?: typeof ADJUTANT_MODEL_INPUT_SCHEMA_VERSION;
  adjutantModelInputFeatureCount?: typeof ADJUTANT_MODEL_INPUT_FEATURE_COUNT;
  exchangeEncoderSchemaVersion?: typeof EXCHANGE_ENCODER_SCHEMA_VERSION;
  exchangeModelInputSchemaVersion?: typeof EXCHANGE_MODEL_INPUT_SCHEMA_VERSION;
  exchangeModelInputFeatureCount?: typeof EXCHANGE_MODEL_INPUT_FEATURE_COUNT;
  decisionMode?: typeof NON_PLAYING_EXCHANGE_RL_DECISION_MODE;
  playingModelInputSchemaVersion: typeof MODEL_INPUT_SCHEMA_VERSION;
  playingModelInputFeatureCount: typeof MODEL_INPUT_FEATURE_COUNT;
  actionCount: typeof BIDDING_ACTION_COUNT | typeof CARD_COUNT;
  behaviorPolicy: NonPlayingRlPolicyArtifactManifest;
  fixedPlayingPolicy: NonPlayingRlPolicyArtifactManifest;
  samplingAlgorithm: typeof NON_PLAYING_RL_SAMPLING_ALGORITHM;
  temperature: number;
  reward: {
    type: typeof NON_PLAYING_RL_REWARD_TYPE;
    version: typeof NON_PLAYING_RL_REWARD_VERSION;
    id: typeof NON_PLAYING_RL_REWARD_ID;
  };
  terminalRewardTransform: {
    type: typeof NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_TYPE;
    version: typeof NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_VERSION;
    id: typeof NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_ID;
    sourceRewardId: typeof NON_PLAYING_RL_REWARD_ID;
    baseline: "meanRawRewardAllPlayers";
    formula: "relative_reward_i = raw_reward_i - mean(raw_reward_all_players)";
  };
  allPassRule: {
    id: typeof NON_PLAYING_RL_ALL_PASS_RULE_ID;
    starterPayoff: 0;
    otherPayoff: 0;
  };
  nonLearningAgents: {
    bidding?: {
      type: "rule-based";
      version?: typeof RULE_BASED_AGENT_VERSION;
    } | {
      type: "conservative-bidding";
      id?: typeof CONSERVATIVE_BIDDING_BASELINE_ID;
    } | FrozenBiddingOpponentMixMetadata;
    choosingAdjutant?: {
      type: "rule-based";
      version: typeof RULE_BASED_AGENT_VERSION;
    } | NonPlayingRlPolicyArtifactManifest;
    exchanging?: {
      type: "rule-based";
      version: typeof RULE_BASED_AGENT_VERSION;
    } | NonPlayingRlPolicyArtifactManifest;
  };
  diagnostics?: NonPlayingRlDatasetDiagnostics;
  shards: readonly DatasetShardManifest[];
}

export interface NonPlayingRlDatasetDiagnostics {
  candidateSeatCount: 1;
  frozenSeatCount: 4;
  candidateRotationSeatCount: 5;
  actualGameCount: number;
  logicalSeedCount: number;
  rotationOffsets: readonly number[];
  frozenBiddingOpponentMix?: FrozenBiddingOpponentMixDiagnostics;
  bidding?: NonPlayingBiddingDiagnostics;
}

export interface FrozenBiddingOpponentMixDiagnostics extends FrozenBiddingOpponentMixMetadata {
  ruleBasedSeatCount: number;
  conservativeSeatCount: number;
  seatAssignments: readonly FrozenBiddingOpponentSeatAssignment[];
}

export interface NonPlayingBiddingDiagnostics {
  candidateBiddingDecisionCount: number;
  passCount: number;
  passRate: number | null;
  bidCount: number;
  targetPointCardsDistribution: Record<string, number>;
  suitDistribution: Record<string, number>;
  candidateNapoleonFormationCount: number;
  candidateNapoleonFormationRate: number | null;
  declarationSuccessCount: number;
  declarationSuccessRate: number | null;
  allPassImmediateEndCount: number;
  candidateRoleDistribution: Record<NonPlayingBiddingRlRole, number>;
  meanRawTerminalReward: number | null;
  meanLearningTerminalReward: number | null;
  terminalRewardTransformId: typeof NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_ID;
}

export interface NonPlayingRlPolicyArtifactManifest {
  type: "bidding-onnx" | "adjutant-onnx" | "exchange-onnx" | "playing-onnx";
  artifactId: string;
  onnxFileName: string;
  metadataFileName: string;
  onnxSha256: string;
  metadataSha256: string;
  requestedInferenceDevice?: "cpu" | "auto" | "cuda";
  resolvedInferenceDevice?: "cpu" | "cuda";
  executionProvider?: "cpu" | "cuda";
  metadata: unknown;
}

export interface GenerateNonPlayingBiddingRlDatasetOptions {
  outputDirectory: string;
  biddingPolicy: NonPlayingBiddingRlPolicy;
  biddingPolicyArtifact: NonPlayingRlPolicyArtifactOptions;
  fixedAdjutantPolicy?: NonPlayingAdjutantRlPolicy;
  fixedAdjutantPolicyArtifact?: NonPlayingRlPolicyArtifactOptions;
  fixedExchangePolicy?: NonPlayingExchangeRlPolicy;
  fixedExchangePolicyArtifact?: NonPlayingRlPolicyArtifactOptions;
  playingPolicy: FixedPlayingPolicy;
  playingPolicyArtifact: NonPlayingRlPolicyArtifactOptions;
  startSeed: number;
  gameCount: number;
  gamesPerShard: number;
  temperature?: number;
  maxDecisionSteps?: number;
  onProgress?: (progress: DatasetGenerationProgress) => void;
}

export interface GenerateNonPlayingBiddingRlDatasetResult {
  outputDirectory: string;
  manifest: NonPlayingRlDatasetManifest;
}

export interface GenerateNonPlayingAdjutantRlDatasetOptions {
  outputDirectory: string;
  adjutantPolicy: NonPlayingAdjutantRlPolicy;
  adjutantPolicyArtifact: NonPlayingRlPolicyArtifactOptions;
  playingPolicy: FixedPlayingPolicy;
  playingPolicyArtifact: NonPlayingRlPolicyArtifactOptions;
  startSeed: number;
  gameCount: number;
  gamesPerShard: number;
  temperature?: number;
  maxDecisionSteps?: number;
  onProgress?: (progress: DatasetGenerationProgress) => void;
}

export interface GenerateNonPlayingAdjutantRlDatasetResult {
  outputDirectory: string;
  manifest: NonPlayingRlDatasetManifest;
}

export interface GenerateNonPlayingExchangeRlDatasetOptions {
  outputDirectory: string;
  exchangePolicy: NonPlayingExchangeRlPolicy;
  exchangePolicyArtifact: NonPlayingRlPolicyArtifactOptions;
  playingPolicy: FixedPlayingPolicy;
  playingPolicyArtifact: NonPlayingRlPolicyArtifactOptions;
  startSeed: number;
  gameCount: number;
  gamesPerShard: number;
  temperature?: number;
  maxDecisionSteps?: number;
  onProgress?: (progress: DatasetGenerationProgress) => void;
}

export interface GenerateNonPlayingExchangeRlDatasetResult {
  outputDirectory: string;
  manifest: NonPlayingRlDatasetManifest;
}

export interface BiddingRlSampleDraft {
  step: number;
  playerId: PlayerId;
  relativePlayerIds: readonly PlayerId[];
  modelInput: Float32Array;
  legalBidMask: readonly number[];
  selectedActionIndex: number;
  behaviorLogProbability: number;
}

export interface BiddingRlGameRunResult {
  record: AutomatedGameRecord;
  drafts: readonly BiddingRlSampleDraft[];
  candidateSeatIndex: number;
  rotationOffset: number;
  frozenBiddingOpponentAssignments: readonly FrozenBiddingOpponentSeatAssignment[];
}

export interface AdjutantRlSampleDraft {
  step: number;
  playerId: PlayerId;
  relativePlayerIds: readonly PlayerId[];
  modelInput: Float32Array;
  legalAdjutantMask: readonly number[];
  selectedActionIndex: number;
  behaviorLogProbability: number;
}

export interface AdjutantRlGameRunResult {
  record: AutomatedGameRecord;
  drafts: readonly AdjutantRlSampleDraft[];
  candidateSeatIndex: number;
  rotationOffset: number;
}

export interface ExchangeRlSampleDraft {
  step: number;
  playerId: PlayerId;
  relativePlayerIds: readonly PlayerId[];
  exchangeStepIndex: 0 | 1 | 2;
  remainingDiscardCount: 1 | 2 | 3;
  modelInput: Float32Array;
  legalDiscardCardMask: readonly number[];
  selectedActionIndex: number;
  behaviorLogProbability: number;
}

export interface ExchangeRlGameRunResult {
  record: AutomatedGameRecord;
  drafts: readonly ExchangeRlSampleDraft[];
  candidateSeatIndex: number;
  rotationOffset: number;
}

interface NonPlayingRlDiagnosticsAccumulator {
  actualGameCount: number;
  logicalSeedCount: number;
  rotationOffsets: readonly number[];
  bidding?: {
    candidateBiddingDecisionCount: number;
    passCount: number;
    bidCount: number;
    targetPointCardsDistribution: Record<string, number>;
    suitDistribution: Record<string, number>;
    candidateNapoleonFormationCount: number;
    declarationSuccessCount: number;
    allPassImmediateEndCount: number;
    candidateRoleDistribution: Record<NonPlayingBiddingRlRole, number>;
    rawRewardSum: number;
    learningRewardSum: number;
    rewardCount: number;
  };
  frozenBiddingOpponentMix?: {
    ruleBasedSeatCount: number;
    conservativeSeatCount: number;
    seatAssignments: FrozenBiddingOpponentSeatAssignment[];
  };
}

function createDiagnosticsAccumulator(
  logicalSeedCount: number,
  options: { includeBidding: boolean }
): NonPlayingRlDiagnosticsAccumulator {
  const accumulator: NonPlayingRlDiagnosticsAccumulator = {
    actualGameCount: 0,
    logicalSeedCount,
    rotationOffsets: NON_PLAYING_RL_ROTATION_OFFSETS
  };

  if (options.includeBidding) {
    accumulator.bidding = {
      candidateBiddingDecisionCount: 0,
      passCount: 0,
      bidCount: 0,
      targetPointCardsDistribution: emptyTargetDistribution(),
      suitDistribution: {
        spades: 0,
        hearts: 0,
        diamonds: 0,
        clubs: 0
      },
      candidateNapoleonFormationCount: 0,
      declarationSuccessCount: 0,
      allPassImmediateEndCount: 0,
      candidateRoleDistribution: emptyRoleDistribution(),
      rawRewardSum: 0,
      learningRewardSum: 0,
      rewardCount: 0
    };
    accumulator.frozenBiddingOpponentMix = {
      ruleBasedSeatCount: 0,
      conservativeSeatCount: 0,
      seatAssignments: []
    };
  }

  return accumulator;
}

function recordCandidateGame(
  accumulator: NonPlayingRlDiagnosticsAccumulator,
  result: BiddingRlGameRunResult | AdjutantRlGameRunResult | ExchangeRlGameRunResult
): void {
  accumulator.actualGameCount += 1;

  if (accumulator.bidding === undefined) {
    return;
  }

  if (
    accumulator.frozenBiddingOpponentMix !== undefined &&
    "frozenBiddingOpponentAssignments" in result
  ) {
    for (const assignment of result.frozenBiddingOpponentAssignments) {
      accumulator.frozenBiddingOpponentMix.seatAssignments.push(assignment);
      if (assignment.policy.type === "rule-based-bidding") {
        accumulator.frozenBiddingOpponentMix.ruleBasedSeatCount += 1;
      } else if (assignment.policy.type === "conservative-bidding") {
        accumulator.frozenBiddingOpponentMix.conservativeSeatCount += 1;
      }
    }
  }

  const candidatePlayerId = result.record.playerIds[result.candidateSeatIndex];
  const role = getNonPlayingRole(result.record.result, candidatePlayerId);
  accumulator.bidding.candidateRoleDistribution[role] += 1;

  if (
    result.record.result.resultType === "standard" &&
    (role === "napoleon" || role === "napoleon-adjutant")
  ) {
    accumulator.bidding.candidateNapoleonFormationCount += 1;
    if (result.record.result.winner === "napoleon-team") {
      accumulator.bidding.declarationSuccessCount += 1;
    }
  }

  if (isAllPassImmediateEndGame(result.record)) {
    accumulator.bidding.allPassImmediateEndCount += 1;
  }
}

function recordBiddingDiagnosticSample(
  accumulator: NonPlayingRlDiagnosticsAccumulator,
  sample: NonPlayingBiddingRlSample
): void {
  if (accumulator.bidding === undefined) {
    return;
  }

  accumulator.bidding.candidateBiddingDecisionCount += 1;
  accumulator.bidding.rawRewardSum += sample.rawTerminalReward;
  accumulator.bidding.learningRewardSum += sample.terminalReward;
  accumulator.bidding.rewardCount += 1;

  const action = decodeBiddingAction(sample.selectedActionIndex, sample.actingPlayerId);
  if (action.type === "pass") {
    accumulator.bidding.passCount += 1;
    return;
  }

  accumulator.bidding.bidCount += 1;
  accumulator.bidding.targetPointCardsDistribution[String(action.targetPointCards)] += 1;
  accumulator.bidding.suitDistribution[action.suit] += 1;
}

function finalizeDiagnostics(
  accumulator: NonPlayingRlDiagnosticsAccumulator
): NonPlayingRlDatasetDiagnostics {
  const bidding = accumulator.bidding;
  const opponentMix = accumulator.frozenBiddingOpponentMix;
  return {
    candidateSeatCount: 1,
    frozenSeatCount: 4,
    candidateRotationSeatCount: NON_PLAYING_RL_ROTATION_OFFSETS.length,
    actualGameCount: accumulator.actualGameCount,
    logicalSeedCount: accumulator.logicalSeedCount,
    rotationOffsets: [...accumulator.rotationOffsets],
    ...(opponentMix === undefined
      ? {}
      : {
          frozenBiddingOpponentMix: {
            ...createFrozenBiddingOpponentMixMetadata(),
            ruleBasedSeatCount: opponentMix.ruleBasedSeatCount,
            conservativeSeatCount: opponentMix.conservativeSeatCount,
            seatAssignments: opponentMix.seatAssignments
          }
        }),
    ...(bidding === undefined
      ? {}
      : {
          bidding: {
            candidateBiddingDecisionCount: bidding.candidateBiddingDecisionCount,
            passCount: bidding.passCount,
            passRate: safeRate(bidding.passCount, bidding.candidateBiddingDecisionCount),
            bidCount: bidding.bidCount,
            targetPointCardsDistribution: bidding.targetPointCardsDistribution,
            suitDistribution: bidding.suitDistribution,
            candidateNapoleonFormationCount: bidding.candidateNapoleonFormationCount,
            candidateNapoleonFormationRate: safeRate(
              bidding.candidateNapoleonFormationCount,
              accumulator.actualGameCount
            ),
            declarationSuccessCount: bidding.declarationSuccessCount,
            declarationSuccessRate: safeRate(
              bidding.declarationSuccessCount,
              bidding.candidateNapoleonFormationCount
            ),
            allPassImmediateEndCount: bidding.allPassImmediateEndCount,
            candidateRoleDistribution: bidding.candidateRoleDistribution,
            meanRawTerminalReward: safeMean(bidding.rawRewardSum, bidding.rewardCount),
            meanLearningTerminalReward: safeMean(bidding.learningRewardSum, bidding.rewardCount),
            terminalRewardTransformId: NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_ID
          }
        })
  };
}

function projectedCompletedShardCount(input: {
  completedShardCount: number;
  shardGameCount: number;
  gameOffset: number;
  gameCount: number;
  gamesPerShard: number;
  rotationOffset: number;
}): number {
  const lastRotationOffset = NON_PLAYING_RL_ROTATION_OFFSETS[
    NON_PLAYING_RL_ROTATION_OFFSETS.length - 1
  ];
  if (input.rotationOffset !== lastRotationOffset) {
    return input.completedShardCount;
  }

  const projectedShardGameCount = input.shardGameCount + 1;
  const shardWillClose =
    projectedShardGameCount === input.gamesPerShard ||
    input.gameOffset === input.gameCount - 1;

  return input.completedShardCount + (shardWillClose ? 1 : 0);
}

function emptyTargetDistribution(): Record<string, number> {
  return {
    "13": 0,
    "14": 0,
    "15": 0,
    "16": 0,
    "17": 0,
    "18": 0,
    "19": 0
  };
}

function emptyRoleDistribution(): Record<NonPlayingBiddingRlRole, number> {
  return {
    napoleon: 0,
    adjutant: 0,
    citizen: 0,
    "napoleon-adjutant": 0,
    "all-pass-starter": 0,
    "all-pass-other": 0
  };
}

function safeRate(numerator: number, denominator: number): number | null {
  return denominator === 0 ? null : numerator / denominator;
}

function safeMean(sumValue: number, count: number): number | null {
  return count === 0 ? null : sumValue / count;
}

function isAllPassImmediateEndGame(record: AutomatedGameRecord): boolean {
  if (record.result.resultType !== "all-pass") {
    return false;
  }
  const biddingDecisions = record.decisions.filter((decision) => decision.phase === "bidding");
  return biddingDecisions.length === PLAYER_COUNT &&
    biddingDecisions.every((decision) => decision.action.type === "pass");
}

export function createFrozenBiddingOpponentMixMetadata(): FrozenBiddingOpponentMixMetadata {
  return {
    type: "mixed-frozen-bidding",
    mixingRuleVersion: FROZEN_BIDDING_OPPONENT_MIX_RULE_VERSION,
    selectionUnit: "game-seat",
    ruleBasedWeight: 0.5,
    conservativeWeight: 0.5,
    policies: {
      ruleBased: {
        type: "rule-based-bidding",
        id: RULE_BASED_BIDDING_BASELINE_ID,
        version: RULE_BASED_AGENT_VERSION
      },
      conservative: {
        type: "conservative-bidding",
        id: CONSERVATIVE_BIDDING_BASELINE_ID
      }
    }
  };
}

function createAllPassRuleMetadata(): NonPlayingRlDatasetManifest["allPassRule"] {
  return {
    id: NON_PLAYING_RL_ALL_PASS_RULE_ID,
    starterPayoff: 0,
    otherPayoff: 0
  };
}

function createTerminalRewardTransformMetadata(): NonPlayingRlDatasetManifest["terminalRewardTransform"] {
  return {
    type: NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_TYPE,
    version: NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_VERSION,
    id: NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_ID,
    sourceRewardId: NON_PLAYING_RL_REWARD_ID,
    baseline: "meanRawRewardAllPlayers",
    formula: "relative_reward_i = raw_reward_i - mean(raw_reward_all_players)"
  };
}

export function selectFrozenBiddingOpponentPolicy(input: {
  seed: number;
  candidateSeatIndex: number;
  playerIndex: number;
}): FrozenBiddingOpponentPolicyMetadata {
  if (input.playerIndex === input.candidateSeatIndex) {
    throw new Error("candidate seat cannot be assigned a frozen bidding opponent policy.");
  }

  const digest = createHash("sha256")
    .update(
      `${FROZEN_BIDDING_OPPONENT_MIX_RULE_VERSION}:${input.seed}:${input.candidateSeatIndex}:${input.playerIndex}`
    )
    .digest();
  const bucket = digest.readUInt32BE(0) % 2;

  return bucket === 0
    ? {
        type: "rule-based-bidding",
        id: RULE_BASED_BIDDING_BASELINE_ID
      }
    : {
        type: "conservative-bidding",
        id: CONSERVATIVE_BIDDING_BASELINE_ID
      };
}

export async function generateNonPlayingBiddingRlDataset(
  options: GenerateNonPlayingBiddingRlDatasetOptions
): Promise<GenerateNonPlayingBiddingRlDatasetResult> {
  const temperature = options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE;
  validateNonPlayingBiddingRlGenerationOptions({ ...options, temperature });

  const outputDirectory = resolve(options.outputDirectory);
  await ensureOutputDoesNotExist(outputDirectory);
  await mkdir(dirname(outputDirectory), { recursive: true });

  const behaviorPolicy = await createPolicyArtifactManifest({
    type: "bidding-onnx",
    artifact: options.biddingPolicyArtifact,
    policy: options.biddingPolicy
  });
  const fixedPlayingPolicy = await createPolicyArtifactManifest({
    type: "playing-onnx",
    artifact: options.playingPolicyArtifact,
    policy: options.playingPolicy
  });
  const fixedAdjutantPolicy =
    options.fixedAdjutantPolicy !== undefined && options.fixedAdjutantPolicyArtifact !== undefined
      ? await createPolicyArtifactManifest({
          type: "adjutant-onnx",
          artifact: options.fixedAdjutantPolicyArtifact,
          policy: options.fixedAdjutantPolicy
        })
      : undefined;
  const fixedExchangePolicy =
    options.fixedExchangePolicy !== undefined && options.fixedExchangePolicyArtifact !== undefined
      ? await createPolicyArtifactManifest({
          type: "exchange-onnx",
          artifact: options.fixedExchangePolicyArtifact,
          policy: options.fixedExchangePolicy
        })
      : undefined;
  const tempDirectory = await mkdtemp(
    join(dirname(outputDirectory), `.${basenameForTemp(outputDirectory)}.tmp-`)
  );
  let activeShard: JsonlShardWriter<NonPlayingBiddingRlSample> | null = null;

  try {
    const shards: DatasetShardManifest[] = [];
    let totalSampleCount = 0;
    let shardGameCount = 0;
    let completedActualGames = 0;
    const diagnostics = createDiagnosticsAccumulator(options.gameCount, {
      includeBidding: true
    });

    for (let gameOffset = 0; gameOffset < options.gameCount; gameOffset += 1) {
      const seed = options.startSeed + gameOffset;

      if (activeShard === null) {
        activeShard = createJsonlShardWriter(
          tempDirectory,
          shards.length,
          seed,
          serializeNonPlayingBiddingRlSample
        );
        shardGameCount = 0;
      }

      for (const rotationOffset of NON_PLAYING_RL_ROTATION_OFFSETS) {
        const result = await runNonPlayingBiddingRlGame({
          seed,
          candidateSeatIndex: rotationOffset,
          biddingPolicy: options.biddingPolicy,
          fixedAdjutantPolicy: options.fixedAdjutantPolicy,
          fixedExchangePolicy: options.fixedExchangePolicy,
          playingPolicy: options.playingPolicy,
          temperature,
          maxDecisionSteps: options.maxDecisionSteps
        });
        recordCandidateGame(diagnostics, result);
        const samples = completeNonPlayingBiddingRlSamples(result.record, result.drafts, {
          candidateSeatIndex: result.candidateSeatIndex,
          rotationOffset: result.rotationOffset
        });

        for (const sample of samples) {
          validateNonPlayingBiddingRlSample(sample, seed);
          recordBiddingDiagnosticSample(diagnostics, sample);
          await activeShard.writeSample(sample);
        }

        totalSampleCount += samples.length;
        completedActualGames += 1;

        options.onProgress?.({
          completedGames: completedActualGames,
          totalGames: options.gameCount * NON_PLAYING_RL_ROTATION_OFFSETS.length,
          sampleCount: totalSampleCount,
          completedShards: projectedCompletedShardCount({
            completedShardCount: shards.length,
            shardGameCount,
            gameOffset,
            gameCount: options.gameCount,
            gamesPerShard: options.gamesPerShard,
            rotationOffset
          }),
          currentSeed: seed
        });
      }

      shardGameCount += 1;

      const shardIsComplete =
        shardGameCount === options.gamesPerShard || gameOffset === options.gameCount - 1;

      if (shardIsComplete) {
        const completedShard = await activeShard.close(seed, shardGameCount);
        shards.push(completedShard);
        activeShard = null;
      }

    }

    const manifest = createNonPlayingRlDatasetManifest({
      options,
      temperature,
      sampleCount: totalSampleCount,
      shards,
      behaviorPolicy,
      fixedAdjutantPolicy,
      fixedExchangePolicy,
      fixedPlayingPolicy,
      diagnostics: finalizeDiagnostics(diagnostics)
    });

    validateNonPlayingRlDatasetManifest(manifest);
    await writeFile(join(tempDirectory, "manifest.json"), serializeManifest(manifest), "utf8");
    await rename(tempDirectory, outputDirectory);

    return {
      outputDirectory: options.outputDirectory,
      manifest
    };
  } catch (error) {
    if (activeShard !== null) {
      await activeShard.abort().catch(() => undefined);
    }

    await rm(tempDirectory, { recursive: true, force: true }).catch((cleanupError: unknown) => {
      const message = cleanupError instanceof Error ? cleanupError.message : String(cleanupError);

      if (error instanceof Error) {
        error.message = `${error.message} Cleanup failed: ${message}`;
      }
    });

    throw error;
  }
}

export async function generateNonPlayingAdjutantRlDataset(
  options: GenerateNonPlayingAdjutantRlDatasetOptions
): Promise<GenerateNonPlayingAdjutantRlDatasetResult> {
  const temperature = options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE;
  validateNonPlayingAdjutantRlGenerationOptions({ ...options, temperature });

  const outputDirectory = resolve(options.outputDirectory);
  await ensureOutputDoesNotExist(outputDirectory);
  await mkdir(dirname(outputDirectory), { recursive: true });

  const behaviorPolicy = await createPolicyArtifactManifest({
    type: "adjutant-onnx",
    artifact: options.adjutantPolicyArtifact,
    policy: options.adjutantPolicy
  });
  const fixedPlayingPolicy = await createPolicyArtifactManifest({
    type: "playing-onnx",
    artifact: options.playingPolicyArtifact,
    policy: options.playingPolicy
  });
  const tempDirectory = await mkdtemp(
    join(dirname(outputDirectory), `.${basenameForTemp(outputDirectory)}.tmp-`)
  );
  let activeShard: JsonlShardWriter<NonPlayingAdjutantRlSample> | null = null;

  try {
    const shards: DatasetShardManifest[] = [];
    let totalSampleCount = 0;
    let shardGameCount = 0;
    let completedActualGames = 0;
    const diagnostics = createDiagnosticsAccumulator(options.gameCount, {
      includeBidding: false
    });

    for (let gameOffset = 0; gameOffset < options.gameCount; gameOffset += 1) {
      const seed = options.startSeed + gameOffset;

      if (activeShard === null) {
        activeShard = createJsonlShardWriter(
          tempDirectory,
          shards.length,
          seed,
          serializeNonPlayingAdjutantRlSample
        );
        shardGameCount = 0;
      }

      for (const rotationOffset of NON_PLAYING_RL_ROTATION_OFFSETS) {
        const result = await runNonPlayingAdjutantRlGame({
          seed,
          candidateSeatIndex: rotationOffset,
          adjutantPolicy: options.adjutantPolicy,
          playingPolicy: options.playingPolicy,
          temperature,
          maxDecisionSteps: options.maxDecisionSteps
        });
        recordCandidateGame(diagnostics, result);
        const samples = completeNonPlayingAdjutantRlSamples(result.record, result.drafts, {
          candidateSeatIndex: result.candidateSeatIndex,
          rotationOffset: result.rotationOffset
        });

        for (const sample of samples) {
          validateNonPlayingAdjutantRlSample(sample, seed);
          await activeShard.writeSample(sample);
        }

        totalSampleCount += samples.length;
        completedActualGames += 1;

        options.onProgress?.({
          completedGames: completedActualGames,
          totalGames: options.gameCount * NON_PLAYING_RL_ROTATION_OFFSETS.length,
          sampleCount: totalSampleCount,
          completedShards: projectedCompletedShardCount({
            completedShardCount: shards.length,
            shardGameCount,
            gameOffset,
            gameCount: options.gameCount,
            gamesPerShard: options.gamesPerShard,
            rotationOffset
          }),
          currentSeed: seed
        });
      }

      shardGameCount += 1;

      const shardIsComplete =
        shardGameCount === options.gamesPerShard || gameOffset === options.gameCount - 1;

      if (shardIsComplete) {
        const completedShard = await activeShard.close(seed, shardGameCount);
        shards.push(completedShard);
        activeShard = null;
      }

    }

    const manifest = createNonPlayingAdjutantRlDatasetManifest({
      options,
      temperature,
      sampleCount: totalSampleCount,
      shards,
      behaviorPolicy,
      fixedPlayingPolicy,
      diagnostics: finalizeDiagnostics(diagnostics)
    });

    validateNonPlayingAdjutantRlDatasetManifest(manifest);
    await writeFile(join(tempDirectory, "manifest.json"), serializeManifest(manifest), "utf8");
    await rename(tempDirectory, outputDirectory);

    return {
      outputDirectory: options.outputDirectory,
      manifest
    };
  } catch (error) {
    if (activeShard !== null) {
      await activeShard.abort().catch(() => undefined);
    }

    await rm(tempDirectory, { recursive: true, force: true }).catch((cleanupError: unknown) => {
      const message = cleanupError instanceof Error ? cleanupError.message : String(cleanupError);

      if (error instanceof Error) {
        error.message = `${error.message} Cleanup failed: ${message}`;
      }
    });

    throw error;
  }
}

export async function generateNonPlayingExchangeRlDataset(
  options: GenerateNonPlayingExchangeRlDatasetOptions
): Promise<GenerateNonPlayingExchangeRlDatasetResult> {
  const temperature = options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE;
  validateNonPlayingExchangeRlGenerationOptions({ ...options, temperature });

  const outputDirectory = resolve(options.outputDirectory);
  await ensureOutputDoesNotExist(outputDirectory);
  await mkdir(dirname(outputDirectory), { recursive: true });

  const behaviorPolicy = await createPolicyArtifactManifest({
    type: "exchange-onnx",
    artifact: options.exchangePolicyArtifact,
    policy: options.exchangePolicy
  });
  const fixedPlayingPolicy = await createPolicyArtifactManifest({
    type: "playing-onnx",
    artifact: options.playingPolicyArtifact,
    policy: options.playingPolicy
  });
  const tempDirectory = await mkdtemp(
    join(dirname(outputDirectory), `.${basenameForTemp(outputDirectory)}.tmp-`)
  );
  let activeShard: JsonlShardWriter<NonPlayingExchangeRlSample> | null = null;

  try {
    const shards: DatasetShardManifest[] = [];
    let totalSampleCount = 0;
    let shardGameCount = 0;
    let completedActualGames = 0;
    const diagnostics = createDiagnosticsAccumulator(options.gameCount, {
      includeBidding: false
    });

    for (let gameOffset = 0; gameOffset < options.gameCount; gameOffset += 1) {
      const seed = options.startSeed + gameOffset;

      if (activeShard === null) {
        activeShard = createJsonlShardWriter(
          tempDirectory,
          shards.length,
          seed,
          serializeNonPlayingExchangeRlSample
        );
        shardGameCount = 0;
      }

      for (const rotationOffset of NON_PLAYING_RL_ROTATION_OFFSETS) {
        const result = await runNonPlayingExchangeRlGame({
          seed,
          candidateSeatIndex: rotationOffset,
          exchangePolicy: options.exchangePolicy,
          playingPolicy: options.playingPolicy,
          temperature,
          maxDecisionSteps: options.maxDecisionSteps
        });
        recordCandidateGame(diagnostics, result);
        const samples = completeNonPlayingExchangeRlSamples(result.record, result.drafts, {
          candidateSeatIndex: result.candidateSeatIndex,
          rotationOffset: result.rotationOffset
        });

        for (const sample of samples) {
          validateNonPlayingExchangeRlSample(sample, seed);
          await activeShard.writeSample(sample);
        }

        totalSampleCount += samples.length;
        completedActualGames += 1;

        options.onProgress?.({
          completedGames: completedActualGames,
          totalGames: options.gameCount * NON_PLAYING_RL_ROTATION_OFFSETS.length,
          sampleCount: totalSampleCount,
          completedShards: projectedCompletedShardCount({
            completedShardCount: shards.length,
            shardGameCount,
            gameOffset,
            gameCount: options.gameCount,
            gamesPerShard: options.gamesPerShard,
            rotationOffset
          }),
          currentSeed: seed
        });
      }

      shardGameCount += 1;

      const shardIsComplete =
        shardGameCount === options.gamesPerShard || gameOffset === options.gameCount - 1;

      if (shardIsComplete) {
        const completedShard = await activeShard.close(seed, shardGameCount);
        shards.push(completedShard);
        activeShard = null;
      }

    }

    const manifest = createNonPlayingExchangeRlDatasetManifest({
      options,
      temperature,
      sampleCount: totalSampleCount,
      shards,
      behaviorPolicy,
      fixedPlayingPolicy,
      diagnostics: finalizeDiagnostics(diagnostics)
    });

    validateNonPlayingExchangeRlDatasetManifest(manifest);
    await writeFile(join(tempDirectory, "manifest.json"), serializeManifest(manifest), "utf8");
    await rename(tempDirectory, outputDirectory);

    return {
      outputDirectory: options.outputDirectory,
      manifest
    };
  } catch (error) {
    if (activeShard !== null) {
      await activeShard.abort().catch(() => undefined);
    }

    await rm(tempDirectory, { recursive: true, force: true }).catch((cleanupError: unknown) => {
      const message = cleanupError instanceof Error ? cleanupError.message : String(cleanupError);

      if (error instanceof Error) {
        error.message = `${error.message} Cleanup failed: ${message}`;
      }
    });

    throw error;
  }
}

export async function runNonPlayingBiddingRlGame(options: {
  seed: number;
  candidateSeatIndex: number;
  biddingPolicy: NonPlayingBiddingRlPolicy;
  fixedAdjutantPolicy?: NonPlayingAdjutantRlPolicy;
  fixedExchangePolicy?: NonPlayingExchangeRlPolicy;
  playingPolicy: FixedPlayingPolicy;
  temperature?: number;
  maxDecisionSteps?: number;
}): Promise<BiddingRlGameRunResult> {
  const drafts: BiddingRlSampleDraft[] = [];
  const temperature = options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE;
  validateTemperature(temperature);
  const frozenBiddingOpponentAssignments = new Map<number, FrozenBiddingOpponentPolicyMetadata>();

  const record = await runAutomatedGame({
    seed: options.seed,
    maxDecisionSteps: options.maxDecisionSteps,
    createAgent: ({ rng, playerIndex }) => {
      if (playerIndex === options.candidateSeatIndex) {
        return new NonPlayingBiddingRlAgent({
          biddingPolicy: options.biddingPolicy,
          fixedAdjutantPolicy: options.fixedAdjutantPolicy,
          fixedExchangePolicy: options.fixedExchangePolicy,
          playingPolicy: options.playingPolicy,
          rng,
          temperature,
          recordSample: (sample) => {
            drafts.push(sample);
          }
        });
      }

      const policy = selectFrozenBiddingOpponentPolicy({
        seed: options.seed,
        candidateSeatIndex: options.candidateSeatIndex,
        playerIndex
      });
      frozenBiddingOpponentAssignments.set(playerIndex, policy);
      return new FrozenNonPlayingRlAgent({
        playingPolicy: options.playingPolicy,
        fixedAdjutantPolicy: options.fixedAdjutantPolicy,
        fixedExchangePolicy: options.fixedExchangePolicy,
        rng,
        biddingPolicyType: policy.type
      });
    }
  });

  return {
    record,
    drafts,
    candidateSeatIndex: options.candidateSeatIndex,
    rotationOffset: options.candidateSeatIndex,
    frozenBiddingOpponentAssignments: [...frozenBiddingOpponentAssignments.entries()]
      .sort(([left], [right]) => left - right)
      .map(([playerIndex, policy]) => ({
        seed: options.seed,
        rotationOffset: options.candidateSeatIndex,
        candidateSeatIndex: options.candidateSeatIndex,
        playerIndex,
        playerId: record.playerIds[playerIndex],
        policy
      }))
  };
}

export async function runNonPlayingAdjutantRlGame(options: {
  seed: number;
  candidateSeatIndex: number;
  adjutantPolicy: NonPlayingAdjutantRlPolicy;
  playingPolicy: FixedPlayingPolicy;
  temperature?: number;
  maxDecisionSteps?: number;
}): Promise<AdjutantRlGameRunResult> {
  const drafts: AdjutantRlSampleDraft[] = [];
  const temperature = options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE;
  validateTemperature(temperature);

  const record = await runAutomatedGame({
    seed: options.seed,
    maxDecisionSteps: options.maxDecisionSteps,
    createAgent: ({ rng, playerIndex }) =>
      playerIndex === options.candidateSeatIndex
        ? new NonPlayingAdjutantRlAgent({
            adjutantPolicy: options.adjutantPolicy,
            playingPolicy: options.playingPolicy,
            rng,
            temperature,
            recordSample: (sample) => {
              drafts.push(sample);
            }
          })
        : new FrozenNonPlayingRlAgent({
            playingPolicy: options.playingPolicy,
            rng
          })
  });

  return {
    record,
    drafts,
    candidateSeatIndex: options.candidateSeatIndex,
    rotationOffset: options.candidateSeatIndex
  };
}

export async function runNonPlayingExchangeRlGame(options: {
  seed: number;
  candidateSeatIndex: number;
  exchangePolicy: NonPlayingExchangeRlPolicy;
  playingPolicy: FixedPlayingPolicy;
  temperature?: number;
  maxDecisionSteps?: number;
}): Promise<ExchangeRlGameRunResult> {
  const drafts: ExchangeRlSampleDraft[] = [];
  const temperature = options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE;
  validateTemperature(temperature);

  const record = await runAutomatedGame({
    seed: options.seed,
    maxDecisionSteps: options.maxDecisionSteps,
    createAgent: ({ rng, playerIndex }) =>
      playerIndex === options.candidateSeatIndex
        ? new NonPlayingExchangeRlAgent({
            exchangePolicy: options.exchangePolicy,
            playingPolicy: options.playingPolicy,
            rng,
            temperature,
            recordSample: (sample) => {
              drafts.push(sample);
            }
          })
        : new FrozenNonPlayingRlAgent({
            playingPolicy: options.playingPolicy,
            rng
          })
  });

  return {
    record,
    drafts,
    candidateSeatIndex: options.candidateSeatIndex,
    rotationOffset: options.candidateSeatIndex
  };
}

export function completeNonPlayingBiddingRlSamples(
  record: AutomatedGameRecord,
  drafts: readonly BiddingRlSampleDraft[],
  context: { candidateSeatIndex: number; rotationOffset: number }
): readonly NonPlayingBiddingRlSample[] {
  const samples: NonPlayingBiddingRlSample[] = [];
  let draftIndex = 0;
  const candidatePlayerId = record.playerIds[context.candidateSeatIndex];

  for (const decision of record.decisions) {
    if (decision.phase !== "bidding") {
      continue;
    }
    if (decision.playerId !== candidatePlayerId) {
      continue;
    }

    const draft = drafts[draftIndex];
    draftIndex += 1;

    if (draft === undefined) {
      throw new Error("Missing sampled bidding action for non-playing RL sample.");
    }
    if (draft.step !== decision.step || draft.playerId !== decision.playerId) {
      throw new Error("Sampled bidding action does not match automated decision order.");
    }
    if (decision.action.type !== "bid" && decision.action.type !== "pass") {
      throw new Error(`Bidding RL decision action must be bid/pass, got ${decision.action.type}.`);
    }

    const outcome = createBiddingRlOutcome(record.result, decision.playerId);
    const reward = calculateNonPlayingLearningTerminalReward(
      record.result,
      decision.playerId,
      record.playerIds
    );
    const sample: NonPlayingBiddingRlSample = {
      sampleType: NON_PLAYING_RL_DATASET_SAMPLE_TYPE,
      schemaVersion: NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION,
      seed: record.seed,
      step: decision.step,
      phase: "bidding",
      actingPlayerId: decision.playerId,
      actingPlayerIndex: record.playerIds.indexOf(decision.playerId),
      candidateSeatIndex: context.candidateSeatIndex,
      rotationOffset: context.rotationOffset,
      relativePlayerIds: draft.relativePlayerIds,
      modelInput: Array.from(draft.modelInput),
      legalBidMask: [...draft.legalBidMask],
      selectedActionIndex: draft.selectedActionIndex,
      behaviorLogProbability: draft.behaviorLogProbability,
      rawTerminalReward: reward.rawTerminalReward,
      gameMeanRawTerminalReward: reward.gameMeanRawTerminalReward,
      terminalReward: reward.terminalReward,
      outcome
    };

    validateNonPlayingBiddingRlSample(sample, record.seed);
    samples.push(sample);
  }

  if (draftIndex !== drafts.length) {
    throw new Error("Unused sampled bidding actions remain after completing RL samples.");
  }

  return samples;
}

export function completeNonPlayingAdjutantRlSamples(
  record: AutomatedGameRecord,
  drafts: readonly AdjutantRlSampleDraft[],
  context: { candidateSeatIndex: number; rotationOffset: number }
): readonly NonPlayingAdjutantRlSample[] {
  const samples: NonPlayingAdjutantRlSample[] = [];
  let draftIndex = 0;
  const candidatePlayerId = record.playerIds[context.candidateSeatIndex];

  for (const decision of record.decisions) {
    if (decision.phase !== "choosing-adjutant") {
      continue;
    }
    if (decision.playerId !== candidatePlayerId) {
      continue;
    }

    const draft = drafts[draftIndex];
    draftIndex += 1;

    if (draft === undefined) {
      throw new Error("Missing sampled adjutant action for non-playing RL sample.");
    }
    if (draft.step !== decision.step || draft.playerId !== decision.playerId) {
      throw new Error("Sampled adjutant action does not match automated decision order.");
    }
    if (decision.action.type !== "choose-adjutant") {
      throw new Error(
        `Adjutant RL decision action must be choose-adjutant, got ${decision.action.type}.`
      );
    }

    const outcome = createBiddingRlOutcome(record.result, decision.playerId);
    const reward = calculateNonPlayingLearningTerminalReward(
      record.result,
      decision.playerId,
      record.playerIds
    );
    const sample: NonPlayingAdjutantRlSample = {
      sampleType: NON_PLAYING_ADJUTANT_RL_DATASET_SAMPLE_TYPE,
      schemaVersion: NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION,
      seed: record.seed,
      step: decision.step,
      phase: "choosing-adjutant",
      actingPlayerId: decision.playerId,
      actingPlayerIndex: record.playerIds.indexOf(decision.playerId),
      candidateSeatIndex: context.candidateSeatIndex,
      rotationOffset: context.rotationOffset,
      relativePlayerIds: draft.relativePlayerIds,
      modelInput: Array.from(draft.modelInput),
      legalAdjutantMask: [...draft.legalAdjutantMask],
      selectedActionIndex: draft.selectedActionIndex,
      behaviorLogProbability: draft.behaviorLogProbability,
      rawTerminalReward: reward.rawTerminalReward,
      gameMeanRawTerminalReward: reward.gameMeanRawTerminalReward,
      terminalReward: reward.terminalReward,
      outcome
    };

    validateNonPlayingAdjutantRlSample(sample, record.seed);
    samples.push(sample);
  }

  if (draftIndex !== drafts.length) {
    throw new Error("Unused sampled adjutant actions remain after completing RL samples.");
  }

  return samples;
}

export function completeNonPlayingExchangeRlSamples(
  record: AutomatedGameRecord,
  drafts: readonly ExchangeRlSampleDraft[],
  context: { candidateSeatIndex: number; rotationOffset: number }
): readonly NonPlayingExchangeRlSample[] {
  const samples: NonPlayingExchangeRlSample[] = [];
  let draftIndex = 0;
  const candidatePlayerId = record.playerIds[context.candidateSeatIndex];

  for (const decision of record.decisions) {
    if (decision.phase !== "exchanging") {
      continue;
    }
    if (decision.playerId !== candidatePlayerId) {
      continue;
    }

    if (decision.action.type !== "discard-cards") {
      throw new Error(`Exchange RL decision action must be discard-cards, got ${decision.action.type}.`);
    }

    const selectedActionIndices = decision.action.cardIds.map((cardId) => CARD_IDS.indexOf(cardId));
    if (selectedActionIndices.includes(-1)) {
      throw new Error("Exchange RL decision contains unknown card ids.");
    }

    const outcome = createBiddingRlOutcome(record.result, decision.playerId);
    const reward = calculateNonPlayingLearningTerminalReward(
      record.result,
      decision.playerId,
      record.playerIds
    );
    for (let exchangeStepIndex = 0; exchangeStepIndex < 3; exchangeStepIndex += 1) {
      const draft = drafts[draftIndex];
      draftIndex += 1;

      if (draft === undefined) {
        throw new Error("Missing sampled exchange action for non-playing RL sample.");
      }
      if (
        draft.playerId !== decision.playerId ||
        draft.exchangeStepIndex !== exchangeStepIndex
      ) {
        throw new Error("Sampled exchange action does not match automated decision order.");
      }
      if (draft.selectedActionIndex !== selectedActionIndices[exchangeStepIndex]) {
        throw new Error("Sampled exchange action does not match discarded card order.");
      }

      const sample: NonPlayingExchangeRlSample = {
        sampleType: NON_PLAYING_EXCHANGE_RL_DATASET_SAMPLE_TYPE,
        schemaVersion: NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION,
        seed: record.seed,
        step: decision.step,
        phase: "exchanging",
        actingPlayerId: decision.playerId,
        actingPlayerIndex: record.playerIds.indexOf(decision.playerId),
        candidateSeatIndex: context.candidateSeatIndex,
        rotationOffset: context.rotationOffset,
        relativePlayerIds: draft.relativePlayerIds,
        exchangeStepIndex: draft.exchangeStepIndex,
        remainingDiscardCount: draft.remainingDiscardCount,
        modelInput: Array.from(draft.modelInput),
        legalDiscardCardMask: [...draft.legalDiscardCardMask],
        selectedActionIndex: draft.selectedActionIndex,
        behaviorLogProbability: draft.behaviorLogProbability,
        rawTerminalReward: reward.rawTerminalReward,
        gameMeanRawTerminalReward: reward.gameMeanRawTerminalReward,
        terminalReward: reward.terminalReward,
        outcome
      };

      validateNonPlayingExchangeRlSample(sample, record.seed);
      samples.push(sample);
    }
  }

  if (draftIndex !== drafts.length) {
    throw new Error("Unused sampled exchange actions remain after completing RL samples.");
  }

  return samples;
}

export function calculateNonPlayingTerminalRoleReward(
  outcome:
    | NonPlayingBiddingRlOutcome
    | Pick<NonPlayingStandardBiddingRlOutcome, "winner" | "targetPointCards" | "actingPlayerRole">
): number {
  if ("outcomeType" in outcome && outcome.outcomeType === "all-pass") {
    return outcome.actingPlayerPayoff;
  }

  const d = outcome.targetPointCards;
  const napoleonWon = outcome.winner === "napoleon-team";

  switch (outcome.actingPlayerRole) {
    case "napoleon":
      return napoleonWon ? 2 * d : -5;
    case "adjutant":
      return napoleonWon ? d : 0;
    case "citizen":
      return napoleonWon ? d : 0;
    case "napoleon-adjutant":
      return napoleonWon ? 3 * d : -5;
  }
}

export function calculateNonPlayingLearningTerminalReward(
  result: GameResult,
  actingPlayerId: PlayerId,
  playerIds: readonly PlayerId[]
): {
  rawTerminalReward: number;
  gameMeanRawTerminalReward: number;
  terminalReward: number;
} {
  expectLength("playerIds", playerIds, PLAYER_COUNT);
  if (!playerIds.includes(actingPlayerId)) {
    throw new Error(`actingPlayerId ${actingPlayerId} is not present in playerIds.`);
  }

  const rawRewards = playerIds.map((playerId) =>
    calculateNonPlayingTerminalRoleReward(createBiddingRlOutcome(result, playerId))
  );
  const rawTerminalReward = rawRewards[playerIds.indexOf(actingPlayerId)];
  const gameMeanRawTerminalReward = sum(rawRewards) / PLAYER_COUNT;
  return {
    rawTerminalReward,
    gameMeanRawTerminalReward,
    terminalReward: rawTerminalReward - gameMeanRawTerminalReward
  };
}

export function calculateNonPlayingBiddingLogProbability(options: {
  logits: Float32Array | readonly number[];
  legalBidMask: ArrayLike<number | boolean>;
  selectedActionIndex: number;
  temperature?: number;
}): number {
  const distribution = createMaskedCategoricalDistribution(
    options.logits,
    options.legalBidMask,
    BIDDING_ACTION_COUNT,
    options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE
  );
  const index = distribution.legalIndices.indexOf(options.selectedActionIndex);

  if (index === -1) {
    throw new Error(
      `selectedActionIndex ${options.selectedActionIndex} is not legal under legalBidMask.`
    );
  }

  return distribution.logProbabilities[index];
}

export function calculateNonPlayingAdjutantLogProbability(options: {
  logits: Float32Array | readonly number[];
  legalAdjutantMask: ArrayLike<number | boolean>;
  selectedActionIndex: number;
  temperature?: number;
}): number {
  const distribution = createMaskedCategoricalDistribution(
    options.logits,
    options.legalAdjutantMask,
    CARD_COUNT,
    options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE
  );
  const index = distribution.legalIndices.indexOf(options.selectedActionIndex);

  if (index === -1) {
    throw new Error(
      `selectedActionIndex ${options.selectedActionIndex} is not legal under legalAdjutantMask.`
    );
  }

  return distribution.logProbabilities[index];
}

export function calculateNonPlayingExchangeLogProbability(options: {
  logits: Float32Array | readonly number[];
  legalDiscardCardMask: ArrayLike<number | boolean>;
  selectedActionIndex: number;
  temperature?: number;
}): number {
  const distribution = createMaskedCategoricalDistribution(
    options.logits,
    options.legalDiscardCardMask,
    CARD_COUNT,
    options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE
  );
  const index = distribution.legalIndices.indexOf(options.selectedActionIndex);

  if (index === -1) {
    throw new Error(
      `selectedActionIndex ${options.selectedActionIndex} is not legal under legalDiscardCardMask.`
    );
  }

  return distribution.logProbabilities[index];
}

export function serializeNonPlayingBiddingRlSample(
  sample: NonPlayingBiddingRlSample
): string {
  return `${JSON.stringify(sample)}\n`;
}

export function serializeNonPlayingAdjutantRlSample(
  sample: NonPlayingAdjutantRlSample
): string {
  return `${JSON.stringify(sample)}\n`;
}

export function serializeNonPlayingExchangeRlSample(
  sample: NonPlayingExchangeRlSample
): string {
  return `${JSON.stringify(sample)}\n`;
}

export function validateNonPlayingBiddingRlGenerationOptions(
  options: GenerateNonPlayingBiddingRlDatasetOptions & { temperature?: number }
): void {
  validateUint32("startSeed", options.startSeed);
  validatePositiveInteger("gameCount", options.gameCount);
  validatePositiveInteger("gamesPerShard", options.gamesPerShard);
  validateTemperature(options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE);

  if (options.outputDirectory.length === 0) {
    throw new Error("outputDirectory must be a non-empty path.");
  }
  if (
    (options.fixedAdjutantPolicy === undefined) !==
    (options.fixedAdjutantPolicyArtifact === undefined)
  ) {
    throw new Error("fixedAdjutantPolicy and fixedAdjutantPolicyArtifact must be provided together.");
  }
  if (
    (options.fixedExchangePolicy === undefined) !==
    (options.fixedExchangePolicyArtifact === undefined)
  ) {
    throw new Error("fixedExchangePolicy and fixedExchangePolicyArtifact must be provided together.");
  }

  const endSeed = options.startSeed + options.gameCount - 1;

  if (!Number.isSafeInteger(endSeed) || endSeed > UINT32_MAX) {
    throw new Error(`Seed range exceeds uint32: ${options.startSeed}..${endSeed}`);
  }

  const expectedShardCount = calculateExpectedShardCount(
    options.gameCount,
    options.gamesPerShard
  );

  if (expectedShardCount > MAX_SHARD_COUNT) {
    throw new Error(
      `Dataset would require ${expectedShardCount} shards, exceeding the maximum ${MAX_SHARD_COUNT}.`
    );
  }
}

export function validateNonPlayingAdjutantRlGenerationOptions(
  options: GenerateNonPlayingAdjutantRlDatasetOptions & { temperature?: number }
): void {
  validateUint32("startSeed", options.startSeed);
  validatePositiveInteger("gameCount", options.gameCount);
  validatePositiveInteger("gamesPerShard", options.gamesPerShard);
  validateTemperature(options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE);

  if (options.outputDirectory.length === 0) {
    throw new Error("outputDirectory must be a non-empty path.");
  }

  const endSeed = options.startSeed + options.gameCount - 1;

  if (!Number.isSafeInteger(endSeed) || endSeed > UINT32_MAX) {
    throw new Error(`Seed range exceeds uint32: ${options.startSeed}..${endSeed}`);
  }

  const expectedShardCount = calculateExpectedShardCount(
    options.gameCount,
    options.gamesPerShard
  );

  if (expectedShardCount > MAX_SHARD_COUNT) {
    throw new Error(
      `Dataset would require ${expectedShardCount} shards, exceeding the maximum ${MAX_SHARD_COUNT}.`
    );
  }
}

export function validateNonPlayingExchangeRlGenerationOptions(
  options: GenerateNonPlayingExchangeRlDatasetOptions & { temperature?: number }
): void {
  validateUint32("startSeed", options.startSeed);
  validatePositiveInteger("gameCount", options.gameCount);
  validatePositiveInteger("gamesPerShard", options.gamesPerShard);
  validateTemperature(options.temperature ?? DEFAULT_NON_PLAYING_RL_TEMPERATURE);

  if (options.outputDirectory.length === 0) {
    throw new Error("outputDirectory must be a non-empty path.");
  }

  const endSeed = options.startSeed + options.gameCount - 1;

  if (!Number.isSafeInteger(endSeed) || endSeed > UINT32_MAX) {
    throw new Error(`Seed range exceeds uint32: ${options.startSeed}..${endSeed}`);
  }

  const expectedShardCount = calculateExpectedShardCount(
    options.gameCount,
    options.gamesPerShard
  );

  if (expectedShardCount > MAX_SHARD_COUNT) {
    throw new Error(
      `Dataset would require ${expectedShardCount} shards, exceeding the maximum ${MAX_SHARD_COUNT}.`
    );
  }
}

export function validateNonPlayingBiddingRlSample(
  sample: NonPlayingBiddingRlSample,
  expectedSeed: number
): void {
  validateJsonSafeValue("sample", sample);

  if (sample.sampleType !== NON_PLAYING_RL_DATASET_SAMPLE_TYPE) {
    throw new Error(`Sample sampleType must be ${NON_PLAYING_RL_DATASET_SAMPLE_TYPE}.`);
  }
  if (sample.schemaVersion !== NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION) {
    throw new Error(`Unexpected non-playing RL sample schemaVersion: ${sample.schemaVersion}`);
  }
  if (sample.seed !== expectedSeed) {
    throw new Error(`Sample seed must match current seed: ${sample.seed} !== ${expectedSeed}`);
  }
  validateUint32("Sample seed", sample.seed);
  validatePositiveInteger("Sample step", sample.step);
  validatePlayerIndex("actingPlayerIndex", sample.actingPlayerIndex);
  validatePlayerIndex("candidateSeatIndex", sample.candidateSeatIndex);
  validatePlayerIndex("rotationOffset", sample.rotationOffset);
  if (sample.actingPlayerIndex !== sample.candidateSeatIndex) {
    throw new Error("actingPlayerIndex must match candidateSeatIndex.");
  }
  if (sample.candidateSeatIndex !== sample.rotationOffset) {
    throw new Error("candidateSeatIndex must match rotationOffset.");
  }
  expectLength("relativePlayerIds", sample.relativePlayerIds, PLAYER_COUNT);
  if (sample.relativePlayerIds[0] !== sample.actingPlayerId) {
    throw new Error("actingPlayerId must match relativePlayerIds[0].");
  }
  expectLength("modelInput", sample.modelInput, BIDDING_MODEL_INPUT_FEATURE_COUNT);
  for (const [index, value] of sample.modelInput.entries()) {
    if (!Number.isFinite(value)) {
      throw new Error(`modelInput[${index}] must be finite.`);
    }
  }
  validateLegalBidMask(sample.legalBidMask);
  validateBiddingActionIndex(sample.selectedActionIndex);
  if (sample.legalBidMask[sample.selectedActionIndex] !== 1) {
    throw new Error("selectedActionIndex must be legal in legalBidMask.");
  }
  if (
    !Number.isFinite(sample.behaviorLogProbability) ||
    sample.behaviorLogProbability > 1e-12
  ) {
    throw new Error("behaviorLogProbability must be finite and <= 0.");
  }
  if (!Number.isFinite(sample.terminalReward)) {
    throw new Error("terminalReward must be finite.");
  }
  validateOutcome(sample.outcome);
  validateTerminalRewardTransformFields(sample);
  if (sample.rawTerminalReward !== calculateNonPlayingTerminalRoleReward(sample.outcome)) {
    throw new Error("rawTerminalReward must match reward version formula.");
  }
}

export function validateNonPlayingAdjutantRlSample(
  sample: NonPlayingAdjutantRlSample,
  expectedSeed: number
): void {
  validateJsonSafeValue("sample", sample);

  if (sample.sampleType !== NON_PLAYING_ADJUTANT_RL_DATASET_SAMPLE_TYPE) {
    throw new Error(`Sample sampleType must be ${NON_PLAYING_ADJUTANT_RL_DATASET_SAMPLE_TYPE}.`);
  }
  if (sample.schemaVersion !== NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION) {
    throw new Error(`Unexpected non-playing RL sample schemaVersion: ${sample.schemaVersion}`);
  }
  if (sample.seed !== expectedSeed) {
    throw new Error(`Sample seed must match current seed: ${sample.seed} !== ${expectedSeed}`);
  }
  validateUint32("Sample seed", sample.seed);
  validatePositiveInteger("Sample step", sample.step);
  validatePlayerIndex("actingPlayerIndex", sample.actingPlayerIndex);
  validatePlayerIndex("candidateSeatIndex", sample.candidateSeatIndex);
  validatePlayerIndex("rotationOffset", sample.rotationOffset);
  if (sample.actingPlayerIndex !== sample.candidateSeatIndex) {
    throw new Error("actingPlayerIndex must match candidateSeatIndex.");
  }
  if (sample.candidateSeatIndex !== sample.rotationOffset) {
    throw new Error("candidateSeatIndex must match rotationOffset.");
  }
  expectLength("relativePlayerIds", sample.relativePlayerIds, PLAYER_COUNT);
  if (sample.relativePlayerIds[0] !== sample.actingPlayerId) {
    throw new Error("actingPlayerId must match relativePlayerIds[0].");
  }
  expectLength("modelInput", sample.modelInput, ADJUTANT_MODEL_INPUT_FEATURE_COUNT);
  for (const [index, value] of sample.modelInput.entries()) {
    if (!Number.isFinite(value)) {
      throw new Error(`modelInput[${index}] must be finite.`);
    }
  }
  validateLegalAdjutantMask(sample.legalAdjutantMask);
  validateCardActionIndex(sample.selectedActionIndex);
  if (sample.legalAdjutantMask[sample.selectedActionIndex] !== 1) {
    throw new Error("selectedActionIndex must be legal in legalAdjutantMask.");
  }
  if (
    !Number.isFinite(sample.behaviorLogProbability) ||
    sample.behaviorLogProbability > 1e-12
  ) {
    throw new Error("behaviorLogProbability must be finite and <= 0.");
  }
  if (!Number.isFinite(sample.terminalReward)) {
    throw new Error("terminalReward must be finite.");
  }
  validateOutcome(sample.outcome);
  validateTerminalRewardTransformFields(sample);
  if (sample.rawTerminalReward !== calculateNonPlayingTerminalRoleReward(sample.outcome)) {
    throw new Error("rawTerminalReward must match reward version formula.");
  }
}

export function validateNonPlayingExchangeRlSample(
  sample: NonPlayingExchangeRlSample,
  expectedSeed: number
): void {
  validateJsonSafeValue("sample", sample);

  if (sample.sampleType !== NON_PLAYING_EXCHANGE_RL_DATASET_SAMPLE_TYPE) {
    throw new Error(`Sample sampleType must be ${NON_PLAYING_EXCHANGE_RL_DATASET_SAMPLE_TYPE}.`);
  }
  if (sample.schemaVersion !== NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION) {
    throw new Error(`Unexpected non-playing RL sample schemaVersion: ${sample.schemaVersion}`);
  }
  if (sample.seed !== expectedSeed) {
    throw new Error(`Sample seed must match current seed: ${sample.seed} !== ${expectedSeed}`);
  }
  validateUint32("Sample seed", sample.seed);
  validatePositiveInteger("Sample step", sample.step);
  validatePlayerIndex("actingPlayerIndex", sample.actingPlayerIndex);
  validatePlayerIndex("candidateSeatIndex", sample.candidateSeatIndex);
  validatePlayerIndex("rotationOffset", sample.rotationOffset);
  if (sample.actingPlayerIndex !== sample.candidateSeatIndex) {
    throw new Error("actingPlayerIndex must match candidateSeatIndex.");
  }
  if (sample.candidateSeatIndex !== sample.rotationOffset) {
    throw new Error("candidateSeatIndex must match rotationOffset.");
  }
  expectLength("relativePlayerIds", sample.relativePlayerIds, PLAYER_COUNT);
  if (sample.relativePlayerIds[0] !== sample.actingPlayerId) {
    throw new Error("actingPlayerId must match relativePlayerIds[0].");
  }
  if (!Number.isSafeInteger(sample.exchangeStepIndex) || sample.exchangeStepIndex < 0 || sample.exchangeStepIndex > 2) {
    throw new Error("exchangeStepIndex must be 0, 1, or 2.");
  }
  if (sample.remainingDiscardCount !== 3 - sample.exchangeStepIndex) {
    throw new Error("remainingDiscardCount must equal 3 - exchangeStepIndex.");
  }
  expectLength("modelInput", sample.modelInput, EXCHANGE_MODEL_INPUT_FEATURE_COUNT);
  for (const [index, value] of sample.modelInput.entries()) {
    if (!Number.isFinite(value)) {
      throw new Error(`modelInput[${index}] must be finite.`);
    }
  }
  validateLegalDiscardMask(sample.legalDiscardCardMask);
  validateCardActionIndex(sample.selectedActionIndex);
  if (sample.legalDiscardCardMask[sample.selectedActionIndex] !== 1) {
    throw new Error("selectedActionIndex must be legal in legalDiscardCardMask.");
  }
  if (
    !Number.isFinite(sample.behaviorLogProbability) ||
    sample.behaviorLogProbability > 1e-12
  ) {
    throw new Error("behaviorLogProbability must be finite and <= 0.");
  }
  if (!Number.isFinite(sample.terminalReward)) {
    throw new Error("terminalReward must be finite.");
  }
  validateOutcome(sample.outcome);
  validateTerminalRewardTransformFields(sample);
  if (sample.rawTerminalReward !== calculateNonPlayingTerminalRoleReward(sample.outcome)) {
    throw new Error("rawTerminalReward must match reward version formula.");
  }
}

export function validateNonPlayingRlDatasetManifest(
  manifest: NonPlayingRlDatasetManifest
): void {
  if (manifest.datasetSchemaVersion !== NON_PLAYING_RL_DATASET_SCHEMA_VERSION) {
    throw new Error("Non-playing RL manifest datasetSchemaVersion mismatch.");
  }
  if (manifest.generatorVersion !== NON_PLAYING_RL_DATASET_GENERATOR_VERSION) {
    throw new Error("Non-playing RL manifest generatorVersion mismatch.");
  }
  if (
    manifest.format !== DATASET_FORMAT ||
    manifest.sampleType !== NON_PLAYING_RL_DATASET_SAMPLE_TYPE ||
    manifest.sampleSchemaVersion !== NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION
  ) {
    throw new Error("Non-playing RL manifest format or sample type mismatch.");
  }
  if (
    manifest.phaseScope !== NON_PLAYING_RL_PHASE_SCOPE ||
    !sameStringArray(manifest.learnedPhases, ["bidding"]) ||
    !(
      sameStringArray(manifest.ruleBasedPhases, ["choosing-adjutant", "exchanging"]) ||
      sameStringArray(manifest.ruleBasedPhases, [])
    ) ||
    !(
      sameStringArray(manifest.fixedPhases, ["playing"]) ||
      sameStringArray(manifest.fixedPhases, ["choosing-adjutant", "exchanging", "playing"])
    )
  ) {
    throw new Error("Non-playing RL manifest phase scope mismatch.");
  }
  validateUint32("Manifest startSeed", manifest.startSeed);
  validateUint32("Manifest endSeed", manifest.endSeed);
  if (manifest.endSeed !== manifest.startSeed + manifest.gameCount - 1) {
    throw new Error("Non-playing RL manifest seed range mismatch.");
  }
  validatePositiveInteger("Manifest gameCount", manifest.gameCount);
  validateNonPlayingRolloutTopologyMetadata(manifest);
  validateNonNegativeInteger("Manifest sampleCount", manifest.sampleCount);
  validatePositiveInteger("Manifest gamesPerShard", manifest.gamesPerShard);
  validatePositiveInteger("Manifest shardCount", manifest.shardCount);
  if (manifest.shardCount !== calculateExpectedShardCount(manifest.gameCount, manifest.gamesPerShard)) {
    throw new Error("Non-playing RL manifest shardCount mismatch.");
  }
  if (manifest.shardCount !== manifest.shards.length) {
    throw new Error("Non-playing RL manifest shardCount must match shards length.");
  }
  if (manifest.sampleCount !== sum(manifest.shards.map((shard) => shard.sampleCount))) {
    throw new Error("Non-playing RL manifest sampleCount must equal shard sample counts.");
  }
  if (manifest.playerCount !== PLAYER_COUNT || manifest.cardCount !== CARD_COUNT) {
    throw new Error("Non-playing RL manifest fixed dimensions mismatch.");
  }
  if (!sameStringArray(manifest.cardIds, CARD_IDS)) {
    throw new Error("Non-playing RL manifest cardIds mismatch.");
  }
  if (manifest.cardIdsSha256 !== calculateCardIdsSha256()) {
    throw new Error("Non-playing RL manifest cardIdsSha256 mismatch.");
  }
  if (
    manifest.biddingEncoderSchemaVersion !== BIDDING_ENCODER_SCHEMA_VERSION ||
    manifest.biddingModelInputSchemaVersion !== BIDDING_MODEL_INPUT_SCHEMA_VERSION ||
    manifest.biddingModelInputFeatureCount !== BIDDING_MODEL_INPUT_FEATURE_COUNT ||
    manifest.playingModelInputSchemaVersion !== MODEL_INPUT_SCHEMA_VERSION ||
    manifest.playingModelInputFeatureCount !== MODEL_INPUT_FEATURE_COUNT ||
    manifest.actionCount !== BIDDING_ACTION_COUNT
  ) {
    throw new Error("Non-playing RL manifest model schema metadata mismatch.");
  }
  validatePolicyArtifactManifest(manifest.behaviorPolicy, "bidding-onnx");
  validatePolicyArtifactManifest(manifest.fixedPlayingPolicy, "playing-onnx");
  if (sameStringArray(manifest.fixedPhases, ["choosing-adjutant", "exchanging", "playing"])) {
    const fixedAdjutant = manifest.nonLearningAgents.choosingAdjutant;
    const fixedExchange = manifest.nonLearningAgents.exchanging;
    if (fixedAdjutant === undefined || fixedExchange === undefined) {
      throw new Error("Non-playing RL manifest fixed non-playing policy metadata missing.");
    }
    if (fixedAdjutant.type !== "adjutant-onnx" || fixedExchange.type !== "exchange-onnx") {
      throw new Error("Non-playing RL manifest fixed non-playing policy type mismatch.");
    }
    validatePolicyArtifactManifest(fixedAdjutant, "adjutant-onnx");
    validatePolicyArtifactManifest(fixedExchange, "exchange-onnx");
  }
  if (manifest.samplingAlgorithm !== NON_PLAYING_RL_SAMPLING_ALGORITHM) {
    throw new Error("Non-playing RL manifest samplingAlgorithm mismatch.");
  }
  validateTemperature(manifest.temperature);
  if (
    manifest.reward.type !== NON_PLAYING_RL_REWARD_TYPE ||
    manifest.reward.version !== NON_PLAYING_RL_REWARD_VERSION ||
    manifest.reward.id !== NON_PLAYING_RL_REWARD_ID
  ) {
    throw new Error("Non-playing RL manifest reward metadata mismatch.");
  }
  validateTerminalRewardTransformMetadata(manifest);
  validateAllPassRuleMetadata(manifest);
  if (sameStringArray(manifest.fixedPhases, ["playing"])) {
    if (
      manifest.nonLearningAgents.choosingAdjutant?.type !== "rule-based" ||
      manifest.nonLearningAgents.choosingAdjutant?.version !== RULE_BASED_AGENT_VERSION ||
      manifest.nonLearningAgents.exchanging?.type !== "rule-based" ||
      manifest.nonLearningAgents.exchanging?.version !== RULE_BASED_AGENT_VERSION
    ) {
      throw new Error("Non-playing RL manifest non-learning agent metadata mismatch.");
    }
  }
  validateFrozenBiddingOpponentMixMetadata(manifest.nonLearningAgents.bidding);
  validateFrozenBiddingOpponentMixDiagnostics(manifest.diagnostics?.frozenBiddingOpponentMix);
  validateShards(manifest);
}

export function validateNonPlayingAdjutantRlDatasetManifest(
  manifest: NonPlayingRlDatasetManifest
): void {
  validateCommonNonPlayingRlDatasetManifest(manifest);
  if (
    manifest.sampleType !== NON_PLAYING_ADJUTANT_RL_DATASET_SAMPLE_TYPE ||
    manifest.sampleSchemaVersion !== NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION
  ) {
    throw new Error("Non-playing adjutant RL manifest sample type mismatch.");
  }
  if (
    manifest.phaseScope !== NON_PLAYING_ADJUTANT_RL_PHASE_SCOPE ||
    !sameStringArray(manifest.learnedPhases, ["choosing-adjutant"]) ||
    !sameStringArray(manifest.ruleBasedPhases, ["bidding", "exchanging"]) ||
    !sameStringArray(manifest.fixedPhases, ["playing"])
  ) {
    throw new Error("Non-playing adjutant RL manifest phase scope mismatch.");
  }
  if (
    manifest.adjutantEncoderSchemaVersion !== ADJUTANT_ENCODER_SCHEMA_VERSION ||
    manifest.adjutantModelInputSchemaVersion !== ADJUTANT_MODEL_INPUT_SCHEMA_VERSION ||
    manifest.adjutantModelInputFeatureCount !== ADJUTANT_MODEL_INPUT_FEATURE_COUNT ||
    manifest.playingModelInputSchemaVersion !== MODEL_INPUT_SCHEMA_VERSION ||
    manifest.playingModelInputFeatureCount !== MODEL_INPUT_FEATURE_COUNT ||
    manifest.actionCount !== CARD_COUNT
  ) {
    throw new Error("Non-playing adjutant RL manifest model schema metadata mismatch.");
  }
  validatePolicyArtifactManifest(manifest.behaviorPolicy, "adjutant-onnx");
  validatePolicyArtifactManifest(manifest.fixedPlayingPolicy, "playing-onnx");
  validateCommonNonPlayingRlPolicyMetadata(manifest);
  if (
    manifest.nonLearningAgents.bidding?.type !== "conservative-bidding" ||
    manifest.nonLearningAgents.bidding?.id !== CONSERVATIVE_BIDDING_BASELINE_ID ||
    manifest.nonLearningAgents.exchanging?.type !== "rule-based" ||
    manifest.nonLearningAgents.exchanging?.version !== RULE_BASED_AGENT_VERSION
  ) {
    throw new Error("Non-playing adjutant RL manifest non-learning agent metadata mismatch.");
  }
  validateShards(manifest);
}

export function validateNonPlayingExchangeRlDatasetManifest(
  manifest: NonPlayingRlDatasetManifest
): void {
  validateCommonNonPlayingRlDatasetManifest(manifest);
  if (
    manifest.sampleType !== NON_PLAYING_EXCHANGE_RL_DATASET_SAMPLE_TYPE ||
    manifest.sampleSchemaVersion !== NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION
  ) {
    throw new Error("Non-playing exchange RL manifest sample type mismatch.");
  }
  if (
    manifest.phaseScope !== NON_PLAYING_EXCHANGE_RL_PHASE_SCOPE ||
    !sameStringArray(manifest.learnedPhases, ["exchanging"]) ||
    !sameStringArray(manifest.ruleBasedPhases, ["bidding", "choosing-adjutant"]) ||
    !sameStringArray(manifest.fixedPhases, ["playing"])
  ) {
    throw new Error("Non-playing exchange RL manifest phase scope mismatch.");
  }
  if (
    manifest.exchangeEncoderSchemaVersion !== EXCHANGE_ENCODER_SCHEMA_VERSION ||
    manifest.exchangeModelInputSchemaVersion !== EXCHANGE_MODEL_INPUT_SCHEMA_VERSION ||
    manifest.exchangeModelInputFeatureCount !== EXCHANGE_MODEL_INPUT_FEATURE_COUNT ||
    manifest.playingModelInputSchemaVersion !== MODEL_INPUT_SCHEMA_VERSION ||
    manifest.playingModelInputFeatureCount !== MODEL_INPUT_FEATURE_COUNT ||
    manifest.actionCount !== CARD_COUNT ||
    manifest.decisionMode !== NON_PLAYING_EXCHANGE_RL_DECISION_MODE
  ) {
    throw new Error("Non-playing exchange RL manifest model schema metadata mismatch.");
  }
  validatePolicyArtifactManifest(manifest.behaviorPolicy, "exchange-onnx");
  validatePolicyArtifactManifest(manifest.fixedPlayingPolicy, "playing-onnx");
  validateCommonNonPlayingRlPolicyMetadata(manifest);
  if (
    manifest.nonLearningAgents.bidding?.type !== "conservative-bidding" ||
    manifest.nonLearningAgents.bidding?.id !== CONSERVATIVE_BIDDING_BASELINE_ID ||
    manifest.nonLearningAgents.choosingAdjutant?.type !== "rule-based" ||
    manifest.nonLearningAgents.choosingAdjutant?.version !== RULE_BASED_AGENT_VERSION
  ) {
    throw new Error("Non-playing exchange RL manifest non-learning agent metadata mismatch.");
  }
  validateShards(manifest);
}

function validateCommonNonPlayingRlDatasetManifest(manifest: NonPlayingRlDatasetManifest): void {
  if (manifest.datasetSchemaVersion !== NON_PLAYING_RL_DATASET_SCHEMA_VERSION) {
    throw new Error("Non-playing RL manifest datasetSchemaVersion mismatch.");
  }
  if (manifest.generatorVersion !== NON_PLAYING_RL_DATASET_GENERATOR_VERSION) {
    throw new Error("Non-playing RL manifest generatorVersion mismatch.");
  }
  if (manifest.format !== DATASET_FORMAT) {
    throw new Error("Non-playing RL manifest format mismatch.");
  }
  validateUint32("Manifest startSeed", manifest.startSeed);
  validateUint32("Manifest endSeed", manifest.endSeed);
  if (manifest.endSeed !== manifest.startSeed + manifest.gameCount - 1) {
    throw new Error("Non-playing RL manifest seed range mismatch.");
  }
  validatePositiveInteger("Manifest gameCount", manifest.gameCount);
  validateNonPlayingRolloutTopologyMetadata(manifest);
  validateNonNegativeInteger("Manifest sampleCount", manifest.sampleCount);
  validatePositiveInteger("Manifest gamesPerShard", manifest.gamesPerShard);
  validatePositiveInteger("Manifest shardCount", manifest.shardCount);
  if (manifest.shardCount !== calculateExpectedShardCount(manifest.gameCount, manifest.gamesPerShard)) {
    throw new Error("Non-playing RL manifest shardCount mismatch.");
  }
  if (manifest.shardCount !== manifest.shards.length) {
    throw new Error("Non-playing RL manifest shardCount must match shards length.");
  }
  if (manifest.sampleCount !== sum(manifest.shards.map((shard) => shard.sampleCount))) {
    throw new Error("Non-playing RL manifest sampleCount must equal shard sample counts.");
  }
  if (manifest.playerCount !== PLAYER_COUNT || manifest.cardCount !== CARD_COUNT) {
    throw new Error("Non-playing RL manifest fixed dimensions mismatch.");
  }
  if (!sameStringArray(manifest.cardIds, CARD_IDS)) {
    throw new Error("Non-playing RL manifest cardIds mismatch.");
  }
  if (manifest.cardIdsSha256 !== calculateCardIdsSha256()) {
    throw new Error("Non-playing RL manifest cardIdsSha256 mismatch.");
  }
}

function validateNonPlayingRolloutTopologyMetadata(
  manifest: NonPlayingRlDatasetManifest
): void {
  if (
    manifest.rolloutPolicyTopology !== NON_PLAYING_RL_ROLLOUT_POLICY_TOPOLOGY ||
    manifest.gameCountUnit !== NON_PLAYING_RL_GAME_COUNT_UNIT
  ) {
    throw new Error("Non-playing RL manifest rollout topology mismatch.");
  }
  if (
    manifest.logicalSeedCount !== manifest.gameCount ||
    manifest.actualGameCount !== manifest.logicalSeedCount * NON_PLAYING_RL_ROTATION_OFFSETS.length ||
    !sameNumberArray(manifest.rotationOffsets, NON_PLAYING_RL_ROTATION_OFFSETS)
  ) {
    throw new Error("Non-playing RL manifest rotation metadata mismatch.");
  }
}

function validateCommonNonPlayingRlPolicyMetadata(manifest: NonPlayingRlDatasetManifest): void {
  if (manifest.samplingAlgorithm !== NON_PLAYING_RL_SAMPLING_ALGORITHM) {
    throw new Error("Non-playing RL manifest samplingAlgorithm mismatch.");
  }
  validateTemperature(manifest.temperature);
  if (
    manifest.reward.type !== NON_PLAYING_RL_REWARD_TYPE ||
    manifest.reward.version !== NON_PLAYING_RL_REWARD_VERSION ||
    manifest.reward.id !== NON_PLAYING_RL_REWARD_ID
  ) {
    throw new Error("Non-playing RL manifest reward metadata mismatch.");
  }
  validateTerminalRewardTransformMetadata(manifest);
  validateAllPassRuleMetadata(manifest);
}

function validateTerminalRewardTransformFields(sample: {
  rawTerminalReward: number;
  gameMeanRawTerminalReward: number;
  terminalReward: number;
}): void {
  if (!Number.isFinite(sample.rawTerminalReward)) {
    throw new Error("rawTerminalReward must be finite.");
  }
  if (!Number.isFinite(sample.gameMeanRawTerminalReward)) {
    throw new Error("gameMeanRawTerminalReward must be finite.");
  }
  if (
    Math.abs(
      sample.terminalReward - (sample.rawTerminalReward - sample.gameMeanRawTerminalReward)
    ) > 1e-9
  ) {
    throw new Error("terminalReward must equal rawTerminalReward - gameMeanRawTerminalReward.");
  }
}

function validateTerminalRewardTransformMetadata(manifest: NonPlayingRlDatasetManifest): void {
  const transform = manifest.terminalRewardTransform;
  if (
    transform.type !== NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_TYPE ||
    transform.version !== NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_VERSION ||
    transform.id !== NON_PLAYING_RL_TERMINAL_REWARD_TRANSFORM_ID ||
    transform.sourceRewardId !== NON_PLAYING_RL_REWARD_ID ||
    transform.baseline !== "meanRawRewardAllPlayers" ||
    transform.formula !== "relative_reward_i = raw_reward_i - mean(raw_reward_all_players)"
  ) {
    throw new Error("Non-playing RL manifest terminalRewardTransform metadata mismatch.");
  }
}

function validateAllPassRuleMetadata(manifest: NonPlayingRlDatasetManifest): void {
  if (
    manifest.allPassRule.id !== NON_PLAYING_RL_ALL_PASS_RULE_ID ||
    manifest.allPassRule.starterPayoff !== 0 ||
    manifest.allPassRule.otherPayoff !== 0
  ) {
    throw new Error("Non-playing RL manifest all-pass rule metadata mismatch.");
  }
}

function validateFrozenBiddingOpponentMixMetadata(
  metadata: NonPlayingRlDatasetManifest["nonLearningAgents"]["bidding"] | undefined
): void {
  if (metadata === undefined || metadata.type !== "mixed-frozen-bidding") {
    throw new Error("Non-playing RL manifest frozen bidding opponent mix metadata mismatch.");
  }
  if (
    metadata.mixingRuleVersion !== FROZEN_BIDDING_OPPONENT_MIX_RULE_VERSION ||
    metadata.selectionUnit !== "game-seat" ||
    metadata.ruleBasedWeight !== 0.5 ||
    metadata.conservativeWeight !== 0.5 ||
    metadata.policies?.ruleBased?.type !== "rule-based-bidding" ||
    metadata.policies?.ruleBased?.id !== RULE_BASED_BIDDING_BASELINE_ID ||
    metadata.policies?.ruleBased?.version !== RULE_BASED_AGENT_VERSION ||
    metadata.policies?.conservative?.type !== "conservative-bidding" ||
    metadata.policies?.conservative?.id !== CONSERVATIVE_BIDDING_BASELINE_ID
  ) {
    throw new Error("Non-playing RL manifest frozen bidding opponent mix metadata mismatch.");
  }
}

function validateFrozenBiddingOpponentMixDiagnostics(
  diagnostics: FrozenBiddingOpponentMixDiagnostics | undefined
): void {
  if (diagnostics === undefined) {
    throw new Error("Non-playing RL manifest frozen bidding opponent mix diagnostics missing.");
  }
  validateFrozenBiddingOpponentMixMetadata(diagnostics);
  if (
    diagnostics.ruleBasedSeatCount < 0 ||
    diagnostics.conservativeSeatCount < 0 ||
    !Number.isInteger(diagnostics.ruleBasedSeatCount) ||
    !Number.isInteger(diagnostics.conservativeSeatCount)
  ) {
    throw new Error("Non-playing RL manifest frozen bidding opponent mix counts mismatch.");
  }
  if (
    diagnostics.ruleBasedSeatCount + diagnostics.conservativeSeatCount !==
    diagnostics.seatAssignments.length
  ) {
    throw new Error("Non-playing RL manifest frozen bidding opponent mix counts mismatch.");
  }
  for (const assignment of diagnostics.seatAssignments) {
    validateUint32("Frozen bidding opponent assignment seed", assignment.seed);
    validatePlayerIndex("Frozen bidding opponent assignment candidateSeatIndex", assignment.candidateSeatIndex);
    validatePlayerIndex("Frozen bidding opponent assignment playerIndex", assignment.playerIndex);
    if (assignment.rotationOffset !== assignment.candidateSeatIndex) {
      throw new Error("Frozen bidding opponent assignment rotation mismatch.");
    }
    if (assignment.playerIndex === assignment.candidateSeatIndex) {
      throw new Error("Frozen bidding opponent assignment cannot target candidate seat.");
    }
    const expected = selectFrozenBiddingOpponentPolicy({
      seed: assignment.seed,
      candidateSeatIndex: assignment.candidateSeatIndex,
      playerIndex: assignment.playerIndex
    });
    if (assignment.policy.type !== expected.type || assignment.policy.id !== expected.id) {
      throw new Error("Frozen bidding opponent assignment policy mismatch.");
    }
  }
}

class NonPlayingBiddingRlAgent implements Agent {
  private readonly ruleBasedAgent: RuleBasedAgent;

  constructor(
    private readonly options: {
      biddingPolicy: NonPlayingBiddingRlPolicy;
      fixedAdjutantPolicy?: NonPlayingAdjutantRlPolicy;
      fixedExchangePolicy?: NonPlayingExchangeRlPolicy;
      playingPolicy: FixedPlayingPolicy;
      rng: () => number;
      temperature: number;
      recordSample: (sample: BiddingRlSampleDraft) => void;
    }
  ) {
    this.ruleBasedAgent = new RuleBasedAgent(options.rng);
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    return this.selectActionWithContext(observation);
  }

  async selectActionWithContext(
    observation: PlayerObservation,
    context?: { actualState: ActualCardState; playerIds: readonly PlayerId[] }
  ): Promise<GameAction> {
    switch (observation.view.phase) {
      case "bidding":
        return this.selectBiddingAction(observation);
      case "playing":
        return this.selectPlayingAction(observation, context);
      case "choosing-adjutant":
        if (this.options.fixedAdjutantPolicy !== undefined) {
          return selectFixedAdjutantAction({
            observation,
            adjutantPolicy: this.options.fixedAdjutantPolicy
          });
        }
        return this.ruleBasedAgent.selectAction(observation);
      case "exchanging":
        if (this.options.fixedExchangePolicy !== undefined) {
          return selectFixedExchangeAction({
            observation,
            exchangePolicy: this.options.fixedExchangePolicy
          });
        }
        return this.ruleBasedAgent.selectAction(observation);
      case "finished":
        return this.ruleBasedAgent.selectAction(observation);
    }
  }

  private async selectBiddingAction(observation: PlayerObservation): Promise<GameAction> {
    const absolutePlayerIds = observation.view.players.map((player) => player.id);
    const encoded = encodeBiddingObservation(observation, absolutePlayerIds);
    const { modelInput, legalBidMask } = createBiddingModelInput(encoded);
    const logits = await this.options.biddingPolicy.predictLogits(modelInput);
    const selection = sampleMaskedCategoricalAction({
      logits,
      legalMask: legalBidMask,
      actionCount: BIDDING_ACTION_COUNT,
      rng: this.options.rng,
      temperature: this.options.temperature
    });
    const selectedAction = decodeBiddingAction(selection.selectedIndex, observation.playerId);
    const legalAction = observation.legalActions.find((action) =>
      biddingActionsEqual(action, selectedAction)
    );

    if (legalAction === undefined) {
      throw new Error(
        `Bidding policy selected action index ${selection.selectedIndex} outside legal actions.`
      );
    }

    this.options.recordSample({
      step: nextDecisionStep(observation),
      playerId: observation.playerId,
      relativePlayerIds: encoded.relativePlayerIds,
      modelInput: Float32Array.from(modelInput),
      legalBidMask: [...legalBidMask],
      selectedActionIndex: selection.selectedIndex,
      behaviorLogProbability: selection.logProbability
    });

    return legalAction;
  }

  private async selectPlayingAction(
    observation: PlayerObservation,
    context: { actualState: ActualCardState; playerIds: readonly PlayerId[] } | undefined
  ): Promise<GameAction> {
    if (observation.publicActionHistory === undefined) {
      throw new Error("Fixed playing policy input requires publicActionHistory.");
    }
    const absolutePlayerIds = context?.playerIds ?? observation.view.players.map((player) => player.id);
    const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
    const biddingHistory = encodeBiddingHistoryFromPublicActions(
      observation.publicActionHistory,
      relativePlayerIds
    );
    const encoded = encodePlayingObservation(observation, absolutePlayerIds, biddingHistory);
    const { modelInput, legalPlayMask } = createPlayingModelInput(encoded);
    const logits = await this.options.playingPolicy.predictLogits(modelInput);
    const selectedCardIndex = selectHighestLegalIndex(logits, legalPlayMask, CARD_COUNT);
    const selectedCardId = getCardId(selectedCardIndex);
    const selectedAction = observation.legalActions.find(
      (action) => action.type === "play-card" && action.cardId === selectedCardId
    );

    if (selectedAction === undefined) {
      throw new Error(
        `Fixed playing policy selected card index ${selectedCardIndex} (${selectedCardId}) outside legal actions.`
      );
    }

    return selectedAction;
  }
}

class FrozenNonPlayingRlAgent implements Agent {
  private readonly conservativeBiddingAgent: ConservativeBiddingAgent;
  private readonly ruleBasedAgent: RuleBasedAgent;

  constructor(
    private readonly options: {
      playingPolicy: FixedPlayingPolicy;
      fixedAdjutantPolicy?: NonPlayingAdjutantRlPolicy;
      fixedExchangePolicy?: NonPlayingExchangeRlPolicy;
      rng: () => number;
      biddingPolicyType?: FrozenBiddingOpponentPolicyType;
    }
  ) {
    this.conservativeBiddingAgent = new ConservativeBiddingAgent(options.rng);
    this.ruleBasedAgent = new RuleBasedAgent(options.rng);
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    return this.selectActionWithContext(observation);
  }

  async selectActionWithContext(
    observation: PlayerObservation,
    context?: { actualState: ActualCardState; playerIds: readonly PlayerId[] }
  ): Promise<GameAction> {
    switch (observation.view.phase) {
      case "bidding":
        return this.options.biddingPolicyType === "rule-based-bidding"
          ? this.ruleBasedAgent.selectAction(observation)
          : this.conservativeBiddingAgent.selectAction(observation);
      case "playing":
        return selectFixedPlayingAction({
          observation,
          context,
          playingPolicy: this.options.playingPolicy
        });
      case "choosing-adjutant":
        if (this.options.fixedAdjutantPolicy !== undefined) {
          return selectFixedAdjutantAction({
            observation,
            adjutantPolicy: this.options.fixedAdjutantPolicy
          });
        }
        return this.ruleBasedAgent.selectAction(observation);
      case "exchanging":
        if (this.options.fixedExchangePolicy !== undefined) {
          return selectFixedExchangeAction({
            observation,
            exchangePolicy: this.options.fixedExchangePolicy
          });
        }
        return this.ruleBasedAgent.selectAction(observation);
      case "finished":
        return this.ruleBasedAgent.selectAction(observation);
    }
  }
}

class NonPlayingAdjutantRlAgent implements Agent {
  private readonly conservativeBiddingAgent: ConservativeBiddingAgent;
  private readonly ruleBasedAgent: RuleBasedAgent;

  constructor(
    private readonly options: {
      adjutantPolicy: NonPlayingAdjutantRlPolicy;
      playingPolicy: FixedPlayingPolicy;
      rng: () => number;
      temperature: number;
      recordSample: (sample: AdjutantRlSampleDraft) => void;
    }
  ) {
    this.conservativeBiddingAgent = new ConservativeBiddingAgent(options.rng);
    this.ruleBasedAgent = new RuleBasedAgent(options.rng);
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    return this.selectActionWithContext(observation);
  }

  async selectActionWithContext(
    observation: PlayerObservation,
    context?: { actualState: ActualCardState; playerIds: readonly PlayerId[] }
  ): Promise<GameAction> {
    switch (observation.view.phase) {
      case "choosing-adjutant":
        return this.selectAdjutantAction(observation);
      case "playing":
        return this.selectPlayingAction(observation, context);
      case "bidding":
        return this.conservativeBiddingAgent.selectAction(observation);
      case "exchanging":
      case "finished":
        return this.ruleBasedAgent.selectAction(observation);
    }
  }

  private async selectAdjutantAction(observation: PlayerObservation): Promise<GameAction> {
    if (observation.publicActionHistory === undefined) {
      throw new Error("Adjutant policy input requires publicActionHistory.");
    }
    const absolutePlayerIds = observation.view.players.map((player) => player.id);
    const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
    const biddingHistory = encodeBiddingHistoryFromPublicActions(
      observation.publicActionHistory,
      relativePlayerIds
    );
    const encoded = encodeAdjutantObservation(observation, absolutePlayerIds, biddingHistory);
    const { modelInput, legalAdjutantMask } = createAdjutantModelInput(encoded);
    const logits = await this.options.adjutantPolicy.predictLogits(modelInput);
    const selection = sampleMaskedCategoricalAction({
      logits,
      legalMask: legalAdjutantMask,
      actionCount: CARD_COUNT,
      rng: this.options.rng,
      temperature: this.options.temperature
    });
    const selectedAction = decodeAdjutantAction(selection.selectedIndex, observation.playerId);
    const legalAction = observation.legalActions.find((action) =>
      adjutantActionsEqual(action, selectedAction)
    );

    if (legalAction === undefined) {
      throw new Error(
        `Adjutant policy selected card index ${selection.selectedIndex} outside legal actions.`
      );
    }

    this.options.recordSample({
      step: nextDecisionStep(observation),
      playerId: observation.playerId,
      relativePlayerIds: encoded.relativePlayerIds,
      modelInput: Float32Array.from(modelInput),
      legalAdjutantMask: [...legalAdjutantMask],
      selectedActionIndex: selection.selectedIndex,
      behaviorLogProbability: selection.logProbability
    });

    return legalAction;
  }

  private async selectPlayingAction(
    observation: PlayerObservation,
    context: { actualState: ActualCardState; playerIds: readonly PlayerId[] } | undefined
  ): Promise<GameAction> {
    return selectFixedPlayingAction({
      observation,
      context,
      playingPolicy: this.options.playingPolicy
    });
  }
}

class NonPlayingExchangeRlAgent implements Agent {
  private readonly conservativeBiddingAgent: ConservativeBiddingAgent;
  private readonly ruleBasedAgent: RuleBasedAgent;

  constructor(
    private readonly options: {
      exchangePolicy: NonPlayingExchangeRlPolicy;
      playingPolicy: FixedPlayingPolicy;
      rng: () => number;
      temperature: number;
      recordSample: (sample: ExchangeRlSampleDraft) => void;
    }
  ) {
    this.conservativeBiddingAgent = new ConservativeBiddingAgent(options.rng);
    this.ruleBasedAgent = new RuleBasedAgent(options.rng);
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    return this.selectActionWithContext(observation);
  }

  async selectActionWithContext(
    observation: PlayerObservation,
    context?: { actualState: ActualCardState; playerIds: readonly PlayerId[] }
  ): Promise<GameAction> {
    switch (observation.view.phase) {
      case "exchanging":
        return this.selectExchangeAction(observation);
      case "playing":
        return selectFixedPlayingAction({
          observation,
          context,
          playingPolicy: this.options.playingPolicy
        });
      case "bidding":
        return this.conservativeBiddingAgent.selectAction(observation);
      case "choosing-adjutant":
      case "finished":
        return this.ruleBasedAgent.selectAction(observation);
    }
  }

  private async selectExchangeAction(observation: PlayerObservation): Promise<GameAction> {
    if (observation.publicActionHistory === undefined) {
      throw new Error("Exchange policy input requires publicActionHistory.");
    }
    const absolutePlayerIds = observation.view.players.map((player) => player.id);
    const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
    const biddingHistory = encodeBiddingHistoryFromPublicActions(
      observation.publicActionHistory,
      relativePlayerIds
    );
    const baseObservation = encodeExchangeObservation(observation, absolutePlayerIds, biddingHistory);
    const selectedActionIndices: number[] = [];

    for (let exchangeStepIndex = 0; exchangeStepIndex < 3; exchangeStepIndex += 1) {
      const steppedObservation = createExchangeStepObservation(baseObservation, selectedActionIndices);
      const { modelInput, legalDiscardCardMask } = createExchangeModelInput(steppedObservation);
      const logits = await this.options.exchangePolicy.predictLogits(modelInput);
      const selection = sampleMaskedCategoricalAction({
        logits,
        legalMask: legalDiscardCardMask,
        actionCount: CARD_COUNT,
        rng: this.options.rng,
        temperature: this.options.temperature
      });

      this.options.recordSample({
        step: nextDecisionStep(observation),
        playerId: observation.playerId,
        relativePlayerIds: steppedObservation.relativePlayerIds,
        exchangeStepIndex: exchangeStepIndex as 0 | 1 | 2,
        remainingDiscardCount: (3 - exchangeStepIndex) as 1 | 2 | 3,
        modelInput: Float32Array.from(modelInput),
        legalDiscardCardMask: [...legalDiscardCardMask],
        selectedActionIndex: selection.selectedIndex,
        behaviorLogProbability: selection.logProbability
      });

      selectedActionIndices.push(selection.selectedIndex);
    }

    const selectedCardIds = selectedActionIndices.map((index) => getCardId(index));
    const legalAction = observation.legalActions.find(
      (action) =>
        action.type === "discard-cards" &&
        action.playerId === observation.playerId &&
        sameStringSet(action.cardIds, selectedCardIds)
    );

    if (legalAction === undefined) {
      throw new Error("Exchange policy selected cards outside legal discard action.");
    }

    return {
      type: "discard-cards",
      playerId: observation.playerId,
      cardIds: selectedCardIds
    };
  }
}

function createExchangeStepObservation(
  observation: EncodedExchangeObservation,
  selectedCardIndices: readonly number[]
): EncodedExchangeObservation {
  const partialDiscardMask = Array(CARD_COUNT).fill(0);
  const legalDiscardCardMask = observation.selfHandMask.map((value) => value);
  const seen = new Set<number>();

  for (const index of selectedCardIndices) {
    if (!Number.isSafeInteger(index) || index < 0 || index >= CARD_COUNT) {
      throw new Error(`Exchange selectedActionIndex is invalid: ${index}.`);
    }
    if (seen.has(index)) {
      throw new Error(`Exchange selectedActionIndex is duplicated: ${index}.`);
    }
    if (observation.selfHandMask[index] !== 1) {
      throw new Error(`Exchange selectedActionIndex is outside self hand: ${index}.`);
    }
    seen.add(index);
    partialDiscardMask[index] = 1;
    legalDiscardCardMask[index] = 0;
  }

  return {
    ...observation,
    partialDiscardMask,
    legalDiscardCardMask,
    exchangeStepIndex: selectedCardIndices.length,
    remainingDiscardCount: 3 - selectedCardIndices.length
  };
}

async function selectFixedPlayingAction(options: {
  observation: PlayerObservation;
  context: { actualState: ActualCardState; playerIds: readonly PlayerId[] } | undefined;
  playingPolicy: FixedPlayingPolicy;
}): Promise<GameAction> {
  const { observation, context } = options;
  if (observation.publicActionHistory === undefined) {
    throw new Error("Fixed playing policy input requires publicActionHistory.");
  }
  const absolutePlayerIds = context?.playerIds ?? observation.view.players.map((player) => player.id);
  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
  const biddingHistory = encodeBiddingHistoryFromPublicActions(
    observation.publicActionHistory,
    relativePlayerIds
  );
  const encoded = encodePlayingObservation(observation, absolutePlayerIds, biddingHistory);
  const { modelInput, legalPlayMask } = createPlayingModelInput(encoded);
  const logits = await options.playingPolicy.predictLogits(modelInput);
  const selectedCardIndex = selectHighestLegalIndex(logits, legalPlayMask, CARD_COUNT);
  const selectedCardId = getCardId(selectedCardIndex);
  const selectedAction = observation.legalActions.find(
    (action) => action.type === "play-card" && action.cardId === selectedCardId
  );

  if (selectedAction === undefined) {
    throw new Error(
      `Fixed playing policy selected card index ${selectedCardIndex} (${selectedCardId}) outside legal actions.`
    );
  }

  return selectedAction;
}

async function selectFixedAdjutantAction(options: {
  observation: PlayerObservation;
  adjutantPolicy: NonPlayingAdjutantRlPolicy;
}): Promise<GameAction> {
  const { observation } = options;
  if (observation.publicActionHistory === undefined) {
    throw new Error("Fixed adjutant policy input requires publicActionHistory.");
  }
  const absolutePlayerIds = observation.view.players.map((player) => player.id);
  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
  const biddingHistory = encodeBiddingHistoryFromPublicActions(
    observation.publicActionHistory,
    relativePlayerIds
  );
  const encoded = encodeAdjutantObservation(observation, absolutePlayerIds, biddingHistory);
  const { modelInput, legalAdjutantMask } = createAdjutantModelInput(encoded);
  const logits = await options.adjutantPolicy.predictLogits(modelInput);
  const selectedIndex = selectHighestLegalIndex(logits, legalAdjutantMask, CARD_COUNT);
  const selectedAction = decodeAdjutantAction(selectedIndex, observation.playerId);
  const legalAction = observation.legalActions.find((action) =>
    adjutantActionsEqual(action, selectedAction)
  );

  if (legalAction === undefined) {
    throw new Error(`Fixed adjutant policy selected card index ${selectedIndex} outside legal actions.`);
  }

  return legalAction;
}

async function selectFixedExchangeAction(options: {
  observation: PlayerObservation;
  exchangePolicy: NonPlayingExchangeRlPolicy;
}): Promise<GameAction> {
  const { observation } = options;
  if (observation.publicActionHistory === undefined) {
    throw new Error("Fixed exchange policy input requires publicActionHistory.");
  }
  const absolutePlayerIds = observation.view.players.map((player) => player.id);
  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
  const biddingHistory = encodeBiddingHistoryFromPublicActions(
    observation.publicActionHistory,
    relativePlayerIds
  );
  const baseObservation = encodeExchangeObservation(observation, absolutePlayerIds, biddingHistory);
  const selectedActionIndices: number[] = [];

  for (let exchangeStepIndex = 0; exchangeStepIndex < 3; exchangeStepIndex += 1) {
    const steppedObservation = createExchangeStepObservation(baseObservation, selectedActionIndices);
    const { modelInput, legalDiscardCardMask } = createExchangeModelInput(steppedObservation);
    const logits = await options.exchangePolicy.predictLogits(modelInput);
    selectedActionIndices.push(selectHighestLegalIndex(logits, legalDiscardCardMask, CARD_COUNT));
  }

  const selectedCardIds = selectedActionIndices.map((index) => getCardId(index));
  const legalAction = observation.legalActions.find(
    (action) =>
      action.type === "discard-cards" &&
      action.playerId === observation.playerId &&
      sameStringSet(action.cardIds, selectedCardIds)
  );

  if (legalAction === undefined) {
    throw new Error("Fixed exchange policy selected cards outside legal discard action.");
  }

  return {
    type: "discard-cards",
    playerId: observation.playerId,
    cardIds: selectedCardIds
  };
}

function createBiddingRlOutcome(
  result: GameResult,
  actingPlayerId: PlayerId
): NonPlayingBiddingRlOutcome {
  if (result.resultType === "all-pass") {
    const isStarter = actingPlayerId === result.starterPlayerId;
    return {
      outcomeType: "all-pass",
      starterPlayerId: result.starterPlayerId,
      actingPlayerRole: isStarter ? "all-pass-starter" : "all-pass-other",
      actingPlayerPayoff: 0
    };
  }

  const actingPlayerRole = getNonPlayingRole(result, actingPlayerId);
  if (actingPlayerRole === "all-pass-starter" || actingPlayerRole === "all-pass-other") {
    throw new Error("Standard bidding outcome cannot use an all-pass role.");
  }

  return {
    outcomeType: "standard",
    winner: result.winner,
    targetPointCards: result.targetPointCards,
    napoleonPlayerId: result.napoleonPlayerId,
    actingPlayerRole
  };
}

function getNonPlayingRole(result: GameResult, playerId: PlayerId): NonPlayingBiddingRlRole {
  if (result.resultType === "all-pass") {
    return playerId === result.starterPlayerId ? "all-pass-starter" : "all-pass-other";
  }

  if (playerId === result.napoleonPlayerId) {
    return result.adjutantPlayerId === null || result.adjutantPlayerId === playerId
      ? "napoleon-adjutant"
      : "napoleon";
  }
  if (playerId === result.adjutantPlayerId) {
    return "adjutant";
  }
  return "citizen";
}

function nextDecisionStep(observation: PlayerObservation): number {
  const history = observation.publicActionHistory ?? [];
  if (history.length === 0) {
    return 1;
  }
  return Math.max(...history.map((record) => record.step)) + 1;
}

function sampleMaskedCategoricalAction(options: {
  logits: Float32Array | readonly number[];
  legalMask: ArrayLike<number | boolean>;
  actionCount: number;
  rng: () => number;
  temperature: number;
}): { selectedIndex: number; logProbability: number } {
  const distribution = createMaskedCategoricalDistribution(
    options.logits,
    options.legalMask,
    options.actionCount,
    options.temperature
  );

  if (distribution.legalIndices.length === 1) {
    return {
      selectedIndex: distribution.legalIndices[0],
      logProbability: 0
    };
  }

  const randomValue = options.rng();
  if (!Number.isFinite(randomValue) || randomValue < 0 || randomValue >= 1) {
    throw new Error("rng must return a finite value in [0, 1).");
  }

  let cumulativeProbability = 0;
  for (let index = 0; index < distribution.legalIndices.length; index += 1) {
    cumulativeProbability += distribution.probabilities[index];

    if (randomValue < cumulativeProbability) {
      return {
        selectedIndex: distribution.legalIndices[index],
        logProbability: distribution.logProbabilities[index]
      };
    }
  }

  const lastIndex = distribution.legalIndices.length - 1;
  return {
    selectedIndex: distribution.legalIndices[lastIndex],
    logProbability: distribution.logProbabilities[lastIndex]
  };
}

function createMaskedCategoricalDistribution(
  logits: Float32Array | readonly number[],
  legalMask: ArrayLike<number | boolean>,
  actionCount: number,
  temperature: number
): {
  legalIndices: readonly number[];
  probabilities: readonly number[];
  logProbabilities: readonly number[];
} {
  if (logits.length !== actionCount) {
    throw new Error(`logits must contain ${actionCount} values, got ${logits.length}.`);
  }
  validateTemperature(temperature);

  const legalIndices: number[] = [];
  const scaledLogits: number[] = [];

  for (let index = 0; index < actionCount; index += 1) {
    const logit = Number(logits[index]);

    if (!Number.isFinite(logit)) {
      throw new Error(`logits[${index}] must be finite.`);
    }

    if (isLegalMaskValue(legalMask[index])) {
      legalIndices.push(index);
      scaledLogits.push(logit / temperature);
    }
  }

  if (legalIndices.length === 0) {
    throw new Error("legal mask must contain at least one legal action.");
  }
  if (legalIndices.length === 1) {
    return {
      legalIndices,
      probabilities: [1],
      logProbabilities: [0]
    };
  }

  const maxScaledLogit = Math.max(...scaledLogits);
  const expValues = scaledLogits.map((logit) => Math.exp(logit - maxScaledLogit));
  const expSum = expValues.reduce((sumValue, value) => sumValue + value, 0);

  if (!Number.isFinite(expSum) || expSum <= 0) {
    throw new Error("masked categorical softmax normalization failed.");
  }

  const logDenominator = maxScaledLogit + Math.log(expSum);
  const probabilities = expValues.map((value) => value / expSum);
  const logProbabilities = scaledLogits.map((logit) => logit - logDenominator);

  return { legalIndices, probabilities, logProbabilities };
}

function selectHighestLegalIndex(
  logits: Float32Array | readonly number[],
  legalMask: ArrayLike<number | boolean>,
  actionCount: number
): number {
  if (logits.length !== actionCount) {
    throw new Error(`logits must contain ${actionCount} values, got ${logits.length}.`);
  }

  let selectedIndex = -1;
  let selectedLogit = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < actionCount; index += 1) {
    const logit = Number(logits[index]);
    if (!Number.isFinite(logit)) {
      throw new Error(`logits[${index}] must be finite.`);
    }
    if (!isLegalMaskValue(legalMask[index])) {
      continue;
    }
    if (selectedIndex === -1 || logit > selectedLogit) {
      selectedIndex = index;
      selectedLogit = logit;
    }
  }

  if (selectedIndex === -1) {
    throw new Error("legal mask must contain at least one legal action.");
  }

  return selectedIndex;
}

function createNonPlayingRlDatasetManifest(input: {
  options: GenerateNonPlayingBiddingRlDatasetOptions;
  temperature: number;
  sampleCount: number;
  shards: readonly DatasetShardManifest[];
  behaviorPolicy: NonPlayingRlPolicyArtifactManifest;
  fixedAdjutantPolicy?: NonPlayingRlPolicyArtifactManifest;
  fixedExchangePolicy?: NonPlayingRlPolicyArtifactManifest;
  fixedPlayingPolicy: NonPlayingRlPolicyArtifactManifest;
  diagnostics: NonPlayingRlDatasetDiagnostics;
}): NonPlayingRlDatasetManifest {
  const usesFixedNonPlayingPolicies =
    input.fixedAdjutantPolicy !== undefined && input.fixedExchangePolicy !== undefined;
  return {
    datasetSchemaVersion: NON_PLAYING_RL_DATASET_SCHEMA_VERSION,
    generatorVersion: NON_PLAYING_RL_DATASET_GENERATOR_VERSION,
    format: DATASET_FORMAT,
    sampleType: NON_PLAYING_RL_DATASET_SAMPLE_TYPE,
    sampleSchemaVersion: NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION,
    phaseScope: NON_PLAYING_RL_PHASE_SCOPE,
    learnedPhases: ["bidding"],
    ruleBasedPhases: usesFixedNonPlayingPolicies ? [] : ["choosing-adjutant", "exchanging"],
    fixedPhases: usesFixedNonPlayingPolicies
      ? ["choosing-adjutant", "exchanging", "playing"]
      : ["playing"],
    rolloutPolicyTopology: NON_PLAYING_RL_ROLLOUT_POLICY_TOPOLOGY,
    gameCountUnit: NON_PLAYING_RL_GAME_COUNT_UNIT,
    logicalSeedCount: input.options.gameCount,
    actualGameCount: input.diagnostics.actualGameCount,
    rotationOffsets: [...NON_PLAYING_RL_ROTATION_OFFSETS],
    startSeed: input.options.startSeed,
    endSeed: input.options.startSeed + input.options.gameCount - 1,
    gameCount: input.options.gameCount,
    sampleCount: input.sampleCount,
    gamesPerShard: input.options.gamesPerShard,
    shardCount: input.shards.length,
    playerCount: PLAYER_COUNT,
    cardCount: CARD_COUNT,
    cardIds: CARD_IDS,
    cardIdsSha256: calculateCardIdsSha256(),
    biddingEncoderSchemaVersion: BIDDING_ENCODER_SCHEMA_VERSION,
    biddingModelInputSchemaVersion: BIDDING_MODEL_INPUT_SCHEMA_VERSION,
    biddingModelInputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT,
    playingModelInputSchemaVersion: MODEL_INPUT_SCHEMA_VERSION,
    playingModelInputFeatureCount: MODEL_INPUT_FEATURE_COUNT,
    actionCount: BIDDING_ACTION_COUNT,
    behaviorPolicy: input.behaviorPolicy,
    fixedPlayingPolicy: input.fixedPlayingPolicy,
    samplingAlgorithm: NON_PLAYING_RL_SAMPLING_ALGORITHM,
    temperature: input.temperature,
    reward: {
      type: NON_PLAYING_RL_REWARD_TYPE,
      version: NON_PLAYING_RL_REWARD_VERSION,
      id: NON_PLAYING_RL_REWARD_ID
    },
    terminalRewardTransform: createTerminalRewardTransformMetadata(),
    allPassRule: createAllPassRuleMetadata(),
    nonLearningAgents: {
      bidding: createFrozenBiddingOpponentMixMetadata(),
      choosingAdjutant:
        input.fixedAdjutantPolicy ??
        {
          type: "rule-based",
          version: RULE_BASED_AGENT_VERSION
        },
      exchanging:
        input.fixedExchangePolicy ??
        {
          type: "rule-based",
          version: RULE_BASED_AGENT_VERSION
        }
    },
    diagnostics: input.diagnostics,
    shards: input.shards
  };
}

function createNonPlayingAdjutantRlDatasetManifest(input: {
  options: GenerateNonPlayingAdjutantRlDatasetOptions;
  temperature: number;
  sampleCount: number;
  shards: readonly DatasetShardManifest[];
  behaviorPolicy: NonPlayingRlPolicyArtifactManifest;
  fixedPlayingPolicy: NonPlayingRlPolicyArtifactManifest;
  diagnostics: NonPlayingRlDatasetDiagnostics;
}): NonPlayingRlDatasetManifest {
  return {
    datasetSchemaVersion: NON_PLAYING_RL_DATASET_SCHEMA_VERSION,
    generatorVersion: NON_PLAYING_RL_DATASET_GENERATOR_VERSION,
    format: DATASET_FORMAT,
    sampleType: NON_PLAYING_ADJUTANT_RL_DATASET_SAMPLE_TYPE,
    sampleSchemaVersion: NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION,
    phaseScope: NON_PLAYING_ADJUTANT_RL_PHASE_SCOPE,
    learnedPhases: ["choosing-adjutant"],
    ruleBasedPhases: ["bidding", "exchanging"],
    fixedPhases: ["playing"],
    rolloutPolicyTopology: NON_PLAYING_RL_ROLLOUT_POLICY_TOPOLOGY,
    gameCountUnit: NON_PLAYING_RL_GAME_COUNT_UNIT,
    logicalSeedCount: input.options.gameCount,
    actualGameCount: input.diagnostics.actualGameCount,
    rotationOffsets: [...NON_PLAYING_RL_ROTATION_OFFSETS],
    startSeed: input.options.startSeed,
    endSeed: input.options.startSeed + input.options.gameCount - 1,
    gameCount: input.options.gameCount,
    sampleCount: input.sampleCount,
    gamesPerShard: input.options.gamesPerShard,
    shardCount: input.shards.length,
    playerCount: PLAYER_COUNT,
    cardCount: CARD_COUNT,
    cardIds: CARD_IDS,
    cardIdsSha256: calculateCardIdsSha256(),
    biddingEncoderSchemaVersion: BIDDING_ENCODER_SCHEMA_VERSION,
    biddingModelInputSchemaVersion: BIDDING_MODEL_INPUT_SCHEMA_VERSION,
    biddingModelInputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT,
    adjutantEncoderSchemaVersion: ADJUTANT_ENCODER_SCHEMA_VERSION,
    adjutantModelInputSchemaVersion: ADJUTANT_MODEL_INPUT_SCHEMA_VERSION,
    adjutantModelInputFeatureCount: ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
    playingModelInputSchemaVersion: MODEL_INPUT_SCHEMA_VERSION,
    playingModelInputFeatureCount: MODEL_INPUT_FEATURE_COUNT,
    actionCount: CARD_COUNT,
    behaviorPolicy: input.behaviorPolicy,
    fixedPlayingPolicy: input.fixedPlayingPolicy,
    samplingAlgorithm: NON_PLAYING_RL_SAMPLING_ALGORITHM,
    temperature: input.temperature,
    reward: {
      type: NON_PLAYING_RL_REWARD_TYPE,
      version: NON_PLAYING_RL_REWARD_VERSION,
      id: NON_PLAYING_RL_REWARD_ID
    },
    terminalRewardTransform: createTerminalRewardTransformMetadata(),
    allPassRule: createAllPassRuleMetadata(),
    nonLearningAgents: {
      bidding: {
        type: "conservative-bidding",
        id: CONSERVATIVE_BIDDING_BASELINE_ID
      },
      choosingAdjutant: {
        type: "rule-based",
        version: RULE_BASED_AGENT_VERSION
      },
      exchanging: {
        type: "rule-based",
        version: RULE_BASED_AGENT_VERSION
      }
    },
    diagnostics: input.diagnostics,
    shards: input.shards
  };
}

function createNonPlayingExchangeRlDatasetManifest(input: {
  options: GenerateNonPlayingExchangeRlDatasetOptions;
  temperature: number;
  sampleCount: number;
  shards: readonly DatasetShardManifest[];
  behaviorPolicy: NonPlayingRlPolicyArtifactManifest;
  fixedPlayingPolicy: NonPlayingRlPolicyArtifactManifest;
  diagnostics: NonPlayingRlDatasetDiagnostics;
}): NonPlayingRlDatasetManifest {
  return {
    datasetSchemaVersion: NON_PLAYING_RL_DATASET_SCHEMA_VERSION,
    generatorVersion: NON_PLAYING_RL_DATASET_GENERATOR_VERSION,
    format: DATASET_FORMAT,
    sampleType: NON_PLAYING_EXCHANGE_RL_DATASET_SAMPLE_TYPE,
    sampleSchemaVersion: NON_PLAYING_RL_SAMPLE_SCHEMA_VERSION,
    phaseScope: NON_PLAYING_EXCHANGE_RL_PHASE_SCOPE,
    learnedPhases: ["exchanging"],
    ruleBasedPhases: ["bidding", "choosing-adjutant"],
    fixedPhases: ["playing"],
    rolloutPolicyTopology: NON_PLAYING_RL_ROLLOUT_POLICY_TOPOLOGY,
    gameCountUnit: NON_PLAYING_RL_GAME_COUNT_UNIT,
    logicalSeedCount: input.options.gameCount,
    actualGameCount: input.diagnostics.actualGameCount,
    rotationOffsets: [...NON_PLAYING_RL_ROTATION_OFFSETS],
    startSeed: input.options.startSeed,
    endSeed: input.options.startSeed + input.options.gameCount - 1,
    gameCount: input.options.gameCount,
    sampleCount: input.sampleCount,
    gamesPerShard: input.options.gamesPerShard,
    shardCount: input.shards.length,
    playerCount: PLAYER_COUNT,
    cardCount: CARD_COUNT,
    cardIds: CARD_IDS,
    cardIdsSha256: calculateCardIdsSha256(),
    biddingEncoderSchemaVersion: BIDDING_ENCODER_SCHEMA_VERSION,
    biddingModelInputSchemaVersion: BIDDING_MODEL_INPUT_SCHEMA_VERSION,
    biddingModelInputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT,
    exchangeEncoderSchemaVersion: EXCHANGE_ENCODER_SCHEMA_VERSION,
    exchangeModelInputSchemaVersion: EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
    exchangeModelInputFeatureCount: EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
    decisionMode: NON_PLAYING_EXCHANGE_RL_DECISION_MODE,
    playingModelInputSchemaVersion: MODEL_INPUT_SCHEMA_VERSION,
    playingModelInputFeatureCount: MODEL_INPUT_FEATURE_COUNT,
    actionCount: CARD_COUNT,
    behaviorPolicy: input.behaviorPolicy,
    fixedPlayingPolicy: input.fixedPlayingPolicy,
    samplingAlgorithm: NON_PLAYING_RL_SAMPLING_ALGORITHM,
    temperature: input.temperature,
    reward: {
      type: NON_PLAYING_RL_REWARD_TYPE,
      version: NON_PLAYING_RL_REWARD_VERSION,
      id: NON_PLAYING_RL_REWARD_ID
    },
    terminalRewardTransform: createTerminalRewardTransformMetadata(),
    allPassRule: createAllPassRuleMetadata(),
    nonLearningAgents: {
      bidding: {
        type: "conservative-bidding",
        id: CONSERVATIVE_BIDDING_BASELINE_ID
      },
      choosingAdjutant: {
        type: "rule-based",
        version: RULE_BASED_AGENT_VERSION
      }
    },
    diagnostics: input.diagnostics,
    shards: input.shards
  };
}

async function createPolicyArtifactManifest(input: {
  type: NonPlayingRlPolicyArtifactManifest["type"];
  artifact: NonPlayingRlPolicyArtifactOptions;
  policy: { metadata: unknown; runtime?: PolicyRuntimeInfo };
}): Promise<NonPlayingRlPolicyArtifactManifest> {
  return {
    type: input.type,
    artifactId: input.artifact.artifactId ?? basename(input.artifact.metadataPath),
    onnxFileName: basename(input.artifact.onnxPath),
    metadataFileName: basename(input.artifact.metadataPath),
    onnxSha256: await sha256File(input.artifact.onnxPath),
    metadataSha256: await sha256File(input.artifact.metadataPath),
    ...runtimeInfoForPolicy(input.policy),
    metadata: input.policy.metadata
  };
}

function runtimeInfoForPolicy(policy: {
  runtime?: PolicyRuntimeInfo;
}): Pick<
  NonPlayingRlPolicyArtifactManifest,
  "requestedInferenceDevice" | "resolvedInferenceDevice" | "executionProvider"
> {
  return {
    requestedInferenceDevice: policy.runtime?.requestedInferenceDevice,
    resolvedInferenceDevice: policy.runtime?.resolvedInferenceDevice,
    executionProvider: policy.runtime?.executionProvider
  };
}

function validatePolicyArtifactManifest(
  artifact: NonPlayingRlPolicyArtifactManifest,
  expectedType: NonPlayingRlPolicyArtifactManifest["type"]
): void {
  if (
    artifact.type !== expectedType ||
    artifact.artifactId.length === 0 ||
    artifact.onnxFileName.length === 0 ||
    artifact.metadataFileName.length === 0 ||
    !sha256Pattern.test(artifact.onnxSha256) ||
    !sha256Pattern.test(artifact.metadataSha256)
  ) {
    throw new Error(`Non-playing RL ${expectedType} artifact metadata mismatch.`);
  }
}

function validateShards(manifest: NonPlayingRlDatasetManifest): void {
  let expectedStartSeed = manifest.startSeed;
  const seenFiles = new Set<string>();

  manifest.shards.forEach((shard, index) => {
    validateShard(shard);

    const expectedFile = `shard-${index.toString().padStart(5, "0")}.jsonl`;

    if (shard.file !== expectedFile) {
      throw new Error(`Shard file must be ${expectedFile}, got ${shard.file}.`);
    }
    if (seenFiles.has(shard.file)) {
      throw new Error(`Duplicate shard file: ${shard.file}`);
    }
    seenFiles.add(shard.file);
    if (shard.startSeed !== expectedStartSeed) {
      throw new Error(`Shard ${shard.file} has a seed gap or overlap.`);
    }
    if (shard.gameCount !== shard.endSeed - shard.startSeed + 1) {
      throw new Error(`Shard ${shard.file} gameCount must match seed range.`);
    }
    if (index !== manifest.shards.length - 1 && shard.gameCount !== manifest.gamesPerShard) {
      throw new Error(`Shard ${shard.file} gameCount must match gamesPerShard.`);
    }
    if (index === manifest.shards.length - 1 && shard.gameCount > manifest.gamesPerShard) {
      throw new Error(`Final shard ${shard.file} gameCount must not exceed gamesPerShard.`);
    }
    expectedStartSeed = shard.endSeed + 1;
  });

  if (manifest.shards[0]?.startSeed !== manifest.startSeed) {
    throw new Error("First shard must start at dataset startSeed.");
  }
  if (manifest.shards.at(-1)?.endSeed !== manifest.endSeed) {
    throw new Error("Last shard must end at dataset endSeed.");
  }
}

function validateShard(shard: DatasetShardManifest): void {
  validateUint32(`Shard ${shard.file} startSeed`, shard.startSeed);
  validateUint32(`Shard ${shard.file} endSeed`, shard.endSeed);
  validatePositiveInteger(`Shard ${shard.file} gameCount`, shard.gameCount);
  validateNonNegativeInteger(`Shard ${shard.file} sampleCount`, shard.sampleCount);
  validateNonNegativeInteger(`Shard ${shard.file} byteLength`, shard.byteLength);
  if (!sha256Pattern.test(shard.sha256)) {
    throw new Error(`Shard ${shard.file} sha256 must be lowercase hex.`);
  }
}

function validateOutcome(outcome: NonPlayingBiddingRlOutcome): void {
  if (outcome.outcomeType === "all-pass") {
    if (outcome.starterPlayerId.length === 0) {
      throw new Error("outcome.starterPlayerId must be non-empty.");
    }
    if (
      outcome.actingPlayerRole !== "all-pass-starter" &&
      outcome.actingPlayerRole !== "all-pass-other"
    ) {
      throw new Error("outcome.actingPlayerRole is invalid.");
    }
    if (outcome.actingPlayerPayoff !== 0) {
      throw new Error("outcome.actingPlayerPayoff must be 0 for all-pass.");
    }
    return;
  }

  if (outcome.winner !== "napoleon-team" && outcome.winner !== "alliance") {
    throw new Error("outcome.winner is invalid.");
  }
  validatePositiveInteger("outcome.targetPointCards", outcome.targetPointCards);
  if (!isNonPlayingBiddingRlRole(outcome.actingPlayerRole)) {
    throw new Error("outcome.actingPlayerRole is invalid.");
  }
  if (outcome.napoleonPlayerId.length === 0) {
    throw new Error("outcome.napoleonPlayerId must be non-empty.");
  }
}

function validateLegalBidMask(mask: readonly number[]): void {
  expectLength("legalBidMask", mask, BIDDING_ACTION_COUNT);
  let legalCount = 0;

  for (const value of mask) {
    if (!Number.isFinite(value) || !Number.isInteger(value) || (value !== 0 && value !== 1)) {
      throw new Error("legalBidMask must contain only 0/1 values.");
    }
    legalCount += value;
  }

  if (legalCount === 0) {
    throw new Error("legalBidMask must contain at least one legal action.");
  }
}

function validateLegalAdjutantMask(mask: readonly number[]): void {
  expectLength("legalAdjutantMask", mask, CARD_COUNT);
  let legalCount = 0;

  for (const value of mask) {
    if (!Number.isFinite(value) || !Number.isInteger(value) || (value !== 0 && value !== 1)) {
      throw new Error("legalAdjutantMask must contain only 0/1 values.");
    }
    legalCount += value;
  }

  if (legalCount === 0) {
    throw new Error("legalAdjutantMask must contain at least one legal action.");
  }
}

function validateLegalDiscardMask(mask: readonly number[]): void {
  expectLength("legalDiscardCardMask", mask, CARD_COUNT);
  let legalCount = 0;

  for (const value of mask) {
    if (!Number.isFinite(value) || !Number.isInteger(value) || (value !== 0 && value !== 1)) {
      throw new Error("legalDiscardCardMask must contain only 0/1 values.");
    }
    legalCount += value;
  }

  if (legalCount === 0) {
    throw new Error("legalDiscardCardMask must contain at least one legal action.");
  }
}

function validateBiddingActionIndex(value: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= BIDDING_ACTION_COUNT) {
    throw new Error(`selectedActionIndex must be between 0 and ${BIDDING_ACTION_COUNT - 1}.`);
  }
}

function validateCardActionIndex(value: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= CARD_COUNT) {
    throw new Error(`selectedActionIndex must be between 0 and ${CARD_COUNT - 1}.`);
  }
}

function validatePlayerIndex(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= PLAYER_COUNT) {
    throw new Error(`${name} must be between 0 and ${PLAYER_COUNT - 1}.`);
  }
}

function validateUint32(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > UINT32_MAX) {
    throw new Error(`${name} must be an integer between 0 and ${UINT32_MAX}.`);
  }
}

function validatePositiveInteger(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function validateNonNegativeInteger(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${name} must be a non-negative integer.`);
  }
}

function validateTemperature(value: number): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`temperature must be finite and positive, got ${value}.`);
  }
}

function expectLength(name: string, value: readonly unknown[], expectedLength: number): void {
  if (value.length !== expectedLength) {
    throw new Error(`${name} must have length ${expectedLength}, got ${value.length}.`);
  }
}

function isLegalMaskValue(value: unknown): boolean {
  return value === 1 || value === true;
}

function isNonPlayingBiddingRlRole(value: string): value is NonPlayingBiddingRlRole {
  return (
    value === "napoleon" ||
    value === "adjutant" ||
    value === "citizen" ||
    value === "napoleon-adjutant" ||
    value === "all-pass-starter" ||
    value === "all-pass-other"
  );
}

function biddingActionsEqual(left: GameAction, right: GameAction): boolean {
  if (left.type !== right.type || left.playerId !== right.playerId) {
    return false;
  }
  if (left.type === "pass") {
    return right.type === "pass";
  }
  if (left.type !== "bid" || right.type !== "bid") {
    return false;
  }
  return right.type === "bid" &&
    left.suit === right.suit &&
    left.targetPointCards === right.targetPointCards;
}

function adjutantActionsEqual(left: GameAction, right: GameAction): boolean {
  return left.type === "choose-adjutant" &&
    right.type === "choose-adjutant" &&
    left.playerId === right.playerId &&
    left.cardId === right.cardId;
}

function sameStringSet(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) {
    return false;
  }
  const rightSet = new Set(right);
  return left.every((value) => rightSet.has(value));
}

function calculateExpectedShardCount(gameCount: number, gamesPerShard: number): number {
  const expectedShardCount = Math.ceil(gameCount / gamesPerShard);

  if (!Number.isSafeInteger(expectedShardCount) || expectedShardCount < 1) {
    throw new Error("Expected shard count must be a positive safe integer.");
  }

  return expectedShardCount;
}

async function ensureOutputDoesNotExist(outputDirectory: string): Promise<void> {
  try {
    await stat(outputDirectory);
  } catch (error) {
    if (isNotFoundError(error)) {
      return;
    }
    throw error;
  }

  throw new Error(`Output directory already exists: ${outputDirectory}`);
}

function isNotFoundError(error: unknown): boolean {
  return typeof error === "object" &&
    error !== null &&
    "code" in error &&
    error.code === "ENOENT";
}

function basenameForTemp(outputDirectory: string): string {
  const parts = outputDirectory.split(/[\\/]/);

  return parts.at(-1) ?? "dataset";
}

async function sha256File(path: string): Promise<string> {
  return createHash("sha256").update(await readFile(path)).digest("hex");
}

const sha256Pattern = /^[0-9a-f]{64}$/;

function validateJsonSafeValue(name: string, value: unknown, seen = new WeakSet<object>()): void {
  if (value === undefined) {
    throw new Error(`${name} must not contain undefined.`);
  }
  if (typeof value === "number" && !Number.isFinite(value)) {
    throw new Error(`${name} must not contain NaN or Infinity.`);
  }
  if (typeof value === "function" || typeof value === "symbol") {
    throw new Error(`${name} must not contain unserializable values.`);
  }
  if (value === null || typeof value !== "object") {
    return;
  }
  if (seen.has(value)) {
    throw new Error(`${name} must not contain circular references.`);
  }

  seen.add(value);
  if (Array.isArray(value)) {
    value.forEach((entry, index) => validateJsonSafeValue(`${name}[${index}]`, entry, seen));
    seen.delete(value);
    return;
  }
  Object.entries(value).forEach(([key, entry]) =>
    validateJsonSafeValue(`${name}.${key}`, entry, seen)
  );
  seen.delete(value);
}

function sameStringArray(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) {
    return false;
  }

  return left.every((value, index) => value === right[index]);
}

function sameNumberArray(left: readonly number[], right: readonly number[]): boolean {
  if (left.length !== right.length) {
    return false;
  }

  return left.every((value, index) => value === right[index]);
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
