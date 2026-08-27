import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import {
  ConservativeBiddingAgent,
  RuleBasedAgent,
  createSeededRandom,
  evaluateHandForTrump,
  runAutomatedGame
} from "@napoleon/ai";
import type {
  ActualCardState,
  Agent,
  PlayerObservation
} from "@napoleon/ai";
import {
  createDeck,
  minBidTargetPointCards,
  maxBidTargetPointCards
} from "@napoleon/game-core";
import type {
  GameAction,
  GameResult,
  PlayerId,
  Suit
} from "@napoleon/game-core";
import {
  BIDDING_ACTION_COUNT,
  BIDDING_ENCODER_SCHEMA_VERSION,
  BIDDING_HISTORY_SUIT_ORDER,
  BIDDING_MODEL_INPUT_FEATURE_COUNT,
  BIDDING_MODEL_INPUT_SCHEMA_VERSION,
  CARD_COUNT,
  CARD_IDS,
  createAdjutantModelInput,
  createBiddingModelInput,
  createExchangeModelInput,
  createPlayingModelInput,
  createRelativePlayerOrder,
  decodeAdjutantAction,
  decodeBiddingAction,
  encodeAdjutantObservation,
  encodeBiddingHistoryFromPublicActions,
  encodeBiddingObservation,
  encodeExchangeObservation,
  encodePlayingObservation,
  getCardId
} from "@napoleon/ai-observation";
import {
  NON_PLAYING_RL_ROTATION_OFFSETS,
  type FixedPlayingPolicy,
  type NonPlayingAdjutantRlPolicy,
  type NonPlayingExchangeRlPolicy,
  type NonPlayingRlPolicyArtifactManifest,
  type NonPlayingRlPolicyArtifactOptions,
  type PolicyRuntimeInfo
} from "./generateNonPlayingRlDataset.js";
import { DATASET_FORMAT, RULE_BASED_AGENT_VERSION } from "./schema.js";
import type { DatasetGenerationProgress, DatasetShardManifest } from "./types.js";
import { calculateCardIdsSha256, serializeManifest } from "./serialization.js";

export const BIDDING_Q_COUNTERFACTUAL_DATASET_SAMPLE_TYPE =
  "bidding-q-monte-carlo-counterfactual-sample" as const;
export const BIDDING_Q_COUNTERFACTUAL_SAMPLE_SCHEMA_VERSION = 2 as const;
export const BIDDING_Q_COUNTERFACTUAL_DATASET_SCHEMA_VERSION = 2 as const;
export const BIDDING_Q_COUNTERFACTUAL_DATASET_GENERATOR_VERSION = 2 as const;
export const BIDDING_Q_COUNTERFACTUAL_ACTION_MAPPING_ID =
  "bidding-action-index-v1-pass-then-13-19-spades-hearts-diamonds-clubs" as const;
export const BIDDING_Q_COUNTERFACTUAL_ACTION_PLAN_ID =
  "pass-lowest-suits-strongest-policy-top-random-target-coverage-v1" as const;
export const BIDDING_Q_COUNTERFACTUAL_ROLE_VALUE_ACTION_PLAN_ID =
  "all-legal-actions-role-value-v1" as const;
export const BIDDING_Q_COUNTERFACTUAL_ACTION_PLAN_VERSION = 1 as const;
export const BIDDING_Q_COUNTERFACTUAL_REWARD_ID =
  "bidding-q-contract-result-loss-minus-one-v1" as const;
export const BIDDING_Q_COUNTERFACTUAL_REWARD_TYPE =
  "bidding-q-contract-result-terminal-reward" as const;
export const BIDDING_Q_COUNTERFACTUAL_REWARD_VERSION = 1 as const;
export const BIDDING_Q_COUNTERFACTUAL_TERMINAL_REWARD_TRANSFORM_ID =
  "bidding-q-contract-result-loss-minus-one-identity-v1" as const;
export const BIDDING_Q_COUNTERFACTUAL_SIMULATION_BACKEND = "typescript" as const;
export const BIDDING_Q_COUNTERFACTUAL_TEAM_POINT_CARDS_TARGET_ID =
  "bidding-q-final-role-team-point-cards-v1" as const;
export const BIDDING_Q_COUNTERFACTUAL_OPPONENT_MIX_RULE_VERSION =
  "per-seat-seeded-rule-based-conservative-50-50-v1" as const;
export const RULE_BASED_BIDDING_BASELINE_ID = "rule-based-bidding-v1" as const;

const PLAYER_COUNT = 5;
const DEFAULT_GAMES_PER_SHARD = 100;
const DEFAULT_RANDOM_LEGAL_BID_COUNT = 2;
const DEFAULT_MAX_SOURCE_STATES = 100;
const ACTION_INDEXES = Array.from({ length: BIDDING_ACTION_COUNT }, (_, index) => index);
const TARGETS = Array.from(
  { length: maxBidTargetPointCards - minBidTargetPointCards + 1 },
  (_, index) => index + minBidTargetPointCards
);
const SUITS = BIDDING_HISTORY_SUIT_ORDER;
const CARD_BY_ID = new Map(createDeck().map((card) => [card.id, card]));
type BiddingQCounterfactualActionPlanId =
  | typeof BIDDING_Q_COUNTERFACTUAL_ACTION_PLAN_ID
  | typeof BIDDING_Q_COUNTERFACTUAL_ROLE_VALUE_ACTION_PLAN_ID;

export interface BiddingQCounterfactualPolicy {
  metadata: unknown;
  runtime?: PolicyRuntimeInfo;
  predictLogits: (modelInput: Float32Array | readonly number[]) => Promise<Float32Array>;
}

export interface GenerateBiddingQCounterfactualDatasetOptions {
  outputDirectory: string;
  biddingPolicy: BiddingQCounterfactualPolicy;
  biddingPolicyArtifact: NonPlayingRlPolicyArtifactOptions;
  fixedAdjutantPolicy?: NonPlayingAdjutantRlPolicy;
  fixedAdjutantPolicyArtifact?: NonPlayingRlPolicyArtifactOptions;
  fixedExchangePolicy?: NonPlayingExchangeRlPolicy;
  fixedExchangePolicyArtifact?: NonPlayingRlPolicyArtifactOptions;
  playingPolicy: FixedPlayingPolicy;
  playingPolicyArtifact: NonPlayingRlPolicyArtifactOptions;
  startSeed: number;
  logicalSeedCount: number;
  maxSourceStates?: number;
  repeats: number;
  gamesPerShard?: number;
  randomSeed: number;
  randomLegalBidCount?: number;
  inferenceDevice?: "cpu" | "auto" | "cuda";
  actionPlanId?: BiddingQCounterfactualActionPlanId;
  rewardId?: typeof BIDDING_Q_COUNTERFACTUAL_REWARD_ID;
  simulationBackend?: typeof BIDDING_Q_COUNTERFACTUAL_SIMULATION_BACKEND;
  sourceCommit?: string;
  onProgress?: (progress: DatasetGenerationProgress) => void;
}

export interface GenerateBiddingQCounterfactualDatasetResult {
  outputDirectory: string;
  manifest: BiddingQCounterfactualDatasetManifest;
  summary: BiddingQCounterfactualDatasetSummary;
}

export type BiddingQTerminalRole =
  | "napoleon"
  | "adjutant"
  | "citizen"
  | "napoleon-adjutant"
  | "all-pass-starter"
  | "all-pass-other";

export interface BiddingQSemanticAction {
  type: "pass" | "bid";
  targetPointCards?: number;
  suit?: Suit;
}

export interface BiddingQCounterfactualSample {
  sampleType: typeof BIDDING_Q_COUNTERFACTUAL_DATASET_SAMPLE_TYPE;
  schemaVersion: typeof BIDDING_Q_COUNTERFACTUAL_SAMPLE_SCHEMA_VERSION;
  stateKey: string;
  sourceSeed: number;
  sourceGameSeed: number;
  candidateSeatIndex: number;
  actingPlayerId: PlayerId;
  actingPlayerIndex: number;
  biddingStep: number;
  sourceSelectedActionIndex: number;
  sourceSelectedAction: BiddingQSemanticAction;
  modelInput: readonly number[];
  legalBidMask: readonly number[];
  forcedActionIndex: number;
  forcedAction: BiddingQSemanticAction;
  strongestSuit: Suit;
  strongestSuitScore: number;
  actionPlanId: BiddingQCounterfactualActionPlanId;
  repeatIndex: number;
  rolloutSeed: number;
  terminalReward: number;
  rawTerminalReward: number;
  terminalRole: BiddingQTerminalRole;
  contractSuccess: boolean;
  resultType: GameResult["resultType"];
  finalRole: BiddingQTerminalRole;
  candidateFinalTeam: "napoleon-team" | "alliance" | "no-contract";
  napoleonSidePointCards: number | null;
  coalitionSidePointCards: number | null;
  candidateTeamPointCards: number | null;
  teamPointCardsRegressionMask: boolean;
  finalDeclaredTarget: number | null;
  finalDeclaredSuit: Suit | null;
  contractMargin: number | null;
  frozenBiddingOpponentCounts: BiddingQOpponentCounts;
  opponentConfigurationKey: string;
  result: BiddingQResultSummary;
  provenance: BiddingQSampleProvenance;
}

export interface BiddingQResultSummary {
  winner?: "napoleon-team" | "alliance";
  targetPointCards?: number;
  napoleonTeamPointCards?: number;
  alliancePointCards?: number;
  napoleonPlayerId?: PlayerId;
  adjutantPlayerId?: PlayerId | null;
  starterPlayerId?: PlayerId;
}

export interface BiddingQOpponentCounts {
  ruleBased: number;
  conservative: number;
}

export interface BiddingQSampleProvenance {
  sourceStateKey: string;
  sourceSeed: number;
  sourceGameSeed: number;
  sourceBiddingStep: number;
  replayMatchedModelInput: true;
  replayMatchedLegalBidMask: true;
  forcedOnce: true;
}

export interface BiddingQFrozenOpponentMixMetadata {
  type: "mixed-frozen-bidding";
  mixingRuleVersion: typeof BIDDING_Q_COUNTERFACTUAL_OPPONENT_MIX_RULE_VERSION;
  selectionUnit: "game-seat";
  topology: "candidate-x1-frozen-x4-v1";
  selectionSeed: "sourceSeed";
  ruleBasedWeight: 0.5;
  conservativeWeight: 0.5;
  policies: {
    ruleBased: {
      type: "rule-based";
      id: typeof RULE_BASED_BIDDING_BASELINE_ID;
      version: typeof RULE_BASED_AGENT_VERSION;
    };
    conservative: {
      type: "conservative-bidding";
      id: "conservative-bidding-v1";
    };
  };
}

export interface BiddingQFrozenOpponentPolicyMetadata {
  type: "rule-based" | "conservative-bidding";
  id: typeof RULE_BASED_BIDDING_BASELINE_ID | "conservative-bidding-v1";
  version?: typeof RULE_BASED_AGENT_VERSION;
}

export interface BiddingQCounterfactualDatasetManifest {
  datasetSchemaVersion: typeof BIDDING_Q_COUNTERFACTUAL_DATASET_SCHEMA_VERSION;
  generatorVersion: typeof BIDDING_Q_COUNTERFACTUAL_DATASET_GENERATOR_VERSION;
  format: typeof DATASET_FORMAT;
  sampleType: typeof BIDDING_Q_COUNTERFACTUAL_DATASET_SAMPLE_TYPE;
  sampleSchemaVersion: typeof BIDDING_Q_COUNTERFACTUAL_SAMPLE_SCHEMA_VERSION;
  compactObservation: {
    phase: "bidding";
    encoderSchemaVersion: typeof BIDDING_ENCODER_SCHEMA_VERSION;
    modelInputSchemaVersion: typeof BIDDING_MODEL_INPUT_SCHEMA_VERSION;
    modelInputFeatureCount: typeof BIDDING_MODEL_INPUT_FEATURE_COUNT;
  };
  actionMapping: {
    id: typeof BIDDING_Q_COUNTERFACTUAL_ACTION_MAPPING_ID;
    actionCount: typeof BIDDING_ACTION_COUNT;
    passActionIndex: 0;
    bidTargets: readonly number[];
    suitOrder: readonly Suit[];
  };
  reward: {
    id: typeof BIDDING_Q_COUNTERFACTUAL_REWARD_ID;
    type: typeof BIDDING_Q_COUNTERFACTUAL_REWARD_TYPE;
    version: typeof BIDDING_Q_COUNTERFACTUAL_REWARD_VERSION;
    napoleonWinMultiplier: 2;
    napoleonAdjutantWinMultiplier: 3;
    contractLossReward: -1;
    nonContractReward: 0;
  };
  terminalRewardTransform: {
    id: typeof BIDDING_Q_COUNTERFACTUAL_TERMINAL_REWARD_TRANSFORM_ID;
    type: "identity";
    version: 1;
    formula: "terminal_reward = raw_bidding_q_reward";
  };
  actionPlan: {
    id: BiddingQCounterfactualActionPlanId;
    version: typeof BIDDING_Q_COUNTERFACTUAL_ACTION_PLAN_VERSION;
    randomLegalBidCount: number;
  };
  predictionTarget: {
    id: typeof BIDDING_Q_COUNTERFACTUAL_TEAM_POINT_CARDS_TARGET_ID;
    version: 1;
    roleLabel: "finalRole";
    valueLabel: "candidate-team-point-card-count";
    noContractHandling: "masked-null";
    candidateTeamDefinition: {
      napoleon: "napoleon-side";
      adjutant: "napoleon-side";
      citizen: "coalition-side";
      "napoleon-adjutant": "napoleon-side";
      noContract: "masked";
    };
  };
  repeats: number;
  sourceStates: number;
  forcedStateActionPairs: number;
  sampleCount: number;
  startSeed: number;
  endSeed: number;
  logicalSeedCount: number;
  actualSourceGameCount: number;
  candidateSeatRotation: readonly number[];
  gamesPerShard: number;
  shardCount: number;
  playerCount: 5;
  cardCount: typeof CARD_COUNT;
  cardIds: readonly string[];
  cardIdsSha256: string;
  simulation: {
    backend: typeof BIDDING_Q_COUNTERFACTUAL_SIMULATION_BACKEND;
    inferenceDevice?: "cpu" | "auto" | "cuda";
  };
  opponentMix: BiddingQFrozenOpponentMixMetadata;
  behaviorPolicy: NonPlayingRlPolicyArtifactManifest;
  fixedPlayingPolicy: NonPlayingRlPolicyArtifactManifest;
  fixedAdjutantPolicy?: NonPlayingRlPolicyArtifactManifest;
  fixedExchangePolicy?: NonPlayingRlPolicyArtifactManifest;
  sourceCommit: string | null;
  summary: BiddingQCounterfactualDatasetSummary;
  shards: readonly DatasetShardManifest[];
}

export interface BiddingQCounterfactualDatasetSummary {
  totalSourceStates: number;
  totalForcedStateActionPairs: number;
  totalRolloutSamples: number;
  fallbackCount: number;
  illegalActionCount: number;
  passCount: number;
  bidCount: number;
  suitCounts: Record<Suit, number>;
  targetCounts: Record<string, number>;
  actionIndexCounts: Record<string, number>;
  strongestSuitCounts: Record<Suit, number>;
  strongestByForcedSuitCounts: Record<Suit, Record<Suit, number>>;
  legalButNeverSampledActionCount: number;
  legalButNeverSampledActionIndexes: readonly number[];
  terminalReward: {
    mean: number | null;
    std: number | null;
    min: number | null;
    max: number | null;
  };
  contractSuccessBySuit: Record<Suit, BiddingQRateSummary>;
  contractSuccessByTarget: Record<string, BiddingQRateSummary>;
  resultTypeCounts: Record<GameResult["resultType"], number>;
  terminalRoleCounts: Record<BiddingQTerminalRole, number>;
  finalRoleByForcedActionType: {
    pass: Record<BiddingQTerminalRole, number>;
    bid: Record<BiddingQTerminalRole, number>;
  };
  finalRoleBySuit: Record<Suit, Record<BiddingQTerminalRole, number>>;
  finalRoleByTarget: Record<string, Record<BiddingQTerminalRole, number>>;
  finalRoleByOpponentConfiguration: Record<string, Record<BiddingQTerminalRole, number>>;
  candidateTeamPointCards: {
    count: number;
    mean: number | null;
    std: number | null;
    min: number | null;
    max: number | null;
  };
  contractMargin: {
    count: number;
    mean: number | null;
    std: number | null;
    min: number | null;
    max: number | null;
  };
}

export interface BiddingQRateSummary {
  count: number;
  successCount: number;
  successRate: number | null;
}

interface SourceBiddingState {
  stateKey: string;
  sourceSeed: number;
  sourceGameSeed: number;
  candidateSeatIndex: number;
  actingPlayerId: PlayerId;
  actingPlayerIndex: number;
  biddingStep: number;
  modelInput: readonly number[];
  legalBidMask: readonly number[];
  selectedActionIndex: number;
  selectedAction: BiddingQSemanticAction;
  topPolicyActionIndex: number;
  strongestSuit: Suit;
  strongestSuitScore: number;
}

interface SourceBiddingDraft {
  step: number;
  playerId: PlayerId;
  modelInput: readonly number[];
  legalBidMask: readonly number[];
  selectedActionIndex: number;
  topPolicyActionIndex: number;
}

interface PlannedStateAction {
  source: SourceBiddingState;
  forcedActionIndex: number;
}

interface ForcedRolloutResult {
  sample: BiddingQCounterfactualSample;
  fallbackCount: number;
  illegalActionCount: number;
}

export async function generateBiddingQCounterfactualDataset(
  options: GenerateBiddingQCounterfactualDatasetOptions
): Promise<GenerateBiddingQCounterfactualDatasetResult> {
  validateGenerateBiddingQCounterfactualDatasetOptions(options);
  const outputDirectory = resolve(options.outputDirectory);
  await ensureOutputDoesNotExist(outputDirectory);
  await mkdir(dirname(outputDirectory), { recursive: true });
  const tempDirectory = await mkdtemp(
    join(dirname(outputDirectory), `.${basename(outputDirectory)}.tmp-`)
  );

  try {
    const sourceStates = await collectSourceStates(options);
    const plannedActions = createForcedActionPlan(sourceStates, {
      randomSeed: options.randomSeed,
      randomLegalBidCount: options.randomLegalBidCount ?? DEFAULT_RANDOM_LEGAL_BID_COUNT,
      actionPlanId: options.actionPlanId
    });
    const samples: BiddingQCounterfactualSample[] = [];
    let fallbackCount = 0;
    let illegalActionCount = 0;
    let completedRollouts = 0;

    for (const planned of plannedActions) {
      for (let repeatIndex = 0; repeatIndex < options.repeats; repeatIndex += 1) {
        const rolloutSeed = deriveCounterfactualRolloutSeed({
          randomSeed: options.randomSeed,
          source: planned.source,
          forcedActionIndex: planned.forcedActionIndex,
          repeatIndex
        });
        const result = await runForcedRollout({
          options,
          source: planned.source,
          forcedActionIndex: planned.forcedActionIndex,
          repeatIndex,
          rolloutSeed
        });
        samples.push(result.sample);
        fallbackCount += result.fallbackCount;
        illegalActionCount += result.illegalActionCount;
        completedRollouts += 1;
        options.onProgress?.({
          totalGames: plannedActions.length * options.repeats,
          completedGames: completedRollouts,
          sampleCount: samples.length,
          completedShards: 0,
          currentSeed: planned.source.sourceSeed
        });
      }
    }

    const shards = await writeSampleShards(tempDirectory, samples, options.gamesPerShard ?? DEFAULT_GAMES_PER_SHARD);
    const summary = summarizeBiddingQCounterfactualSamples(samples, {
      sourceStates,
      plannedActions,
      fallbackCount,
      illegalActionCount
    });
    const manifest = await createBiddingQCounterfactualManifest({
      options,
      sourceStates,
      plannedActions,
      samples,
      shards,
      summary
    });
    validateBiddingQCounterfactualDatasetManifest(manifest);
    await writeFile(join(tempDirectory, "manifest.json"), serializeManifest(manifest), "utf8");
    await writeFile(join(tempDirectory, "summary.json"), serializeManifest(summary), "utf8");
    await rename(tempDirectory, outputDirectory);
    return {
      outputDirectory,
      manifest,
      summary
    };
  } catch (error) {
    await rm(tempDirectory, { recursive: true, force: true }).catch(() => undefined);
    throw error;
  }
}

export function validateGenerateBiddingQCounterfactualDatasetOptions(
  options: GenerateBiddingQCounterfactualDatasetOptions
): void {
  validateOutputDirectory(options.outputDirectory);
  validateUint32("startSeed", options.startSeed);
  validatePositiveInteger("logicalSeedCount", options.logicalSeedCount);
  validatePositiveInteger("repeats", options.repeats);
  validateUint32("randomSeed", options.randomSeed);
  if (options.maxSourceStates !== undefined) {
    validatePositiveInteger("maxSourceStates", options.maxSourceStates);
  }
  if (options.gamesPerShard !== undefined) {
    validatePositiveInteger("gamesPerShard", options.gamesPerShard);
  }
  if (options.randomLegalBidCount !== undefined) {
    validateNonNegativeInteger("randomLegalBidCount", options.randomLegalBidCount);
  }
  if (
    options.actionPlanId !== undefined &&
    options.actionPlanId !== BIDDING_Q_COUNTERFACTUAL_ACTION_PLAN_ID &&
    options.actionPlanId !== BIDDING_Q_COUNTERFACTUAL_ROLE_VALUE_ACTION_PLAN_ID
  ) {
    throw new Error("Unsupported bidding Q counterfactual action plan id.");
  }
  if (
    options.rewardId !== undefined &&
    options.rewardId !== BIDDING_Q_COUNTERFACTUAL_REWARD_ID
  ) {
    throw new Error("Unsupported bidding Q counterfactual reward id.");
  }
  if (
    options.simulationBackend !== undefined &&
    options.simulationBackend !== BIDDING_Q_COUNTERFACTUAL_SIMULATION_BACKEND
  ) {
    throw new Error("Unsupported bidding Q counterfactual simulation backend.");
  }
}

export function validateBiddingQCounterfactualDatasetManifest(
  manifest: BiddingQCounterfactualDatasetManifest
): void {
  if (
    manifest.datasetSchemaVersion !== BIDDING_Q_COUNTERFACTUAL_DATASET_SCHEMA_VERSION ||
    manifest.generatorVersion !== BIDDING_Q_COUNTERFACTUAL_DATASET_GENERATOR_VERSION ||
    manifest.format !== DATASET_FORMAT ||
    manifest.sampleType !== BIDDING_Q_COUNTERFACTUAL_DATASET_SAMPLE_TYPE ||
    manifest.sampleSchemaVersion !== BIDDING_Q_COUNTERFACTUAL_SAMPLE_SCHEMA_VERSION
  ) {
    throw new Error("Bidding Q counterfactual manifest schema mismatch.");
  }
  if (
    manifest.compactObservation.phase !== "bidding" ||
    manifest.compactObservation.encoderSchemaVersion !== BIDDING_ENCODER_SCHEMA_VERSION ||
    manifest.compactObservation.modelInputSchemaVersion !== BIDDING_MODEL_INPUT_SCHEMA_VERSION ||
    manifest.compactObservation.modelInputFeatureCount !== BIDDING_MODEL_INPUT_FEATURE_COUNT
  ) {
    throw new Error("Bidding Q counterfactual manifest compact observation mismatch.");
  }
  if (
    manifest.actionMapping.id !== BIDDING_Q_COUNTERFACTUAL_ACTION_MAPPING_ID ||
    manifest.actionMapping.actionCount !== BIDDING_ACTION_COUNT ||
    manifest.actionMapping.passActionIndex !== 0 ||
    !sameNumberArray(manifest.actionMapping.bidTargets, TARGETS) ||
    !sameStringArray(manifest.actionMapping.suitOrder, SUITS)
  ) {
    throw new Error("Bidding Q counterfactual manifest action mapping mismatch.");
  }
  if (
    manifest.reward.id !== BIDDING_Q_COUNTERFACTUAL_REWARD_ID ||
    manifest.reward.type !== BIDDING_Q_COUNTERFACTUAL_REWARD_TYPE ||
    manifest.reward.version !== BIDDING_Q_COUNTERFACTUAL_REWARD_VERSION ||
    manifest.reward.contractLossReward !== -1 ||
    manifest.reward.nonContractReward !== 0
  ) {
    throw new Error("Bidding Q counterfactual manifest reward mismatch.");
  }
  if (
    manifest.terminalRewardTransform.id !== BIDDING_Q_COUNTERFACTUAL_TERMINAL_REWARD_TRANSFORM_ID ||
    manifest.terminalRewardTransform.type !== "identity"
  ) {
    throw new Error("Bidding Q counterfactual manifest reward transform mismatch.");
  }
  if (
    (
      manifest.actionPlan.id !== BIDDING_Q_COUNTERFACTUAL_ACTION_PLAN_ID &&
      manifest.actionPlan.id !== BIDDING_Q_COUNTERFACTUAL_ROLE_VALUE_ACTION_PLAN_ID
    ) ||
    manifest.actionPlan.version !== BIDDING_Q_COUNTERFACTUAL_ACTION_PLAN_VERSION
  ) {
    throw new Error("Bidding Q counterfactual manifest action plan mismatch.");
  }
  if (
    manifest.predictionTarget.id !== BIDDING_Q_COUNTERFACTUAL_TEAM_POINT_CARDS_TARGET_ID ||
    manifest.predictionTarget.version !== 1 ||
    manifest.predictionTarget.noContractHandling !== "masked-null"
  ) {
    throw new Error("Bidding Q counterfactual manifest prediction target mismatch.");
  }
  if (manifest.simulation.backend !== BIDDING_Q_COUNTERFACTUAL_SIMULATION_BACKEND) {
    throw new Error("Bidding Q counterfactual manifest simulation backend mismatch.");
  }
  validateBiddingQFrozenOpponentMixMetadata(manifest.opponentMix);
  if (manifest.sampleCount !== manifest.summary.totalRolloutSamples) {
    throw new Error("Bidding Q counterfactual manifest sample count mismatch.");
  }
  if (manifest.sourceStates !== manifest.summary.totalSourceStates) {
    throw new Error("Bidding Q counterfactual manifest source state count mismatch.");
  }
  if (manifest.forcedStateActionPairs !== manifest.summary.totalForcedStateActionPairs) {
    throw new Error("Bidding Q counterfactual manifest forced pair count mismatch.");
  }
  if (manifest.shardCount !== manifest.shards.length) {
    throw new Error("Bidding Q counterfactual manifest shard count mismatch.");
  }
  if (manifest.sampleCount !== sum(manifest.shards.map((shard) => shard.sampleCount))) {
    throw new Error("Bidding Q counterfactual manifest shard sample count mismatch.");
  }
  if (manifest.cardIdsSha256 !== calculateCardIdsSha256()) {
    throw new Error("Bidding Q counterfactual manifest card id hash mismatch.");
  }
  if (!sameStringArray(manifest.cardIds, CARD_IDS)) {
    throw new Error("Bidding Q counterfactual manifest card ids mismatch.");
  }
}

export function validateBiddingQCounterfactualSample(
  sample: BiddingQCounterfactualSample
): void {
  if (
    sample.sampleType !== BIDDING_Q_COUNTERFACTUAL_DATASET_SAMPLE_TYPE ||
    sample.schemaVersion !== BIDDING_Q_COUNTERFACTUAL_SAMPLE_SCHEMA_VERSION
  ) {
    throw new Error("Bidding Q counterfactual sample schema mismatch.");
  }
  if (sample.modelInput.length !== BIDDING_MODEL_INPUT_FEATURE_COUNT) {
    throw new Error("Bidding Q counterfactual sample modelInput length mismatch.");
  }
  if (sample.legalBidMask.length !== BIDDING_ACTION_COUNT) {
    throw new Error("Bidding Q counterfactual sample legalBidMask length mismatch.");
  }
  validateBiddingActionIndex(sample.forcedActionIndex);
  if (!isLegalMaskValue(sample.legalBidMask[sample.forcedActionIndex])) {
    throw new Error("Bidding Q counterfactual sample forced action must be legal.");
  }
  if (!Number.isFinite(sample.terminalReward)) {
    throw new Error("Bidding Q counterfactual sample terminalReward must be finite.");
  }
  if (sample.finalRole !== sample.terminalRole) {
    throw new Error("Bidding Q counterfactual sample finalRole must match terminalRole.");
  }
  if (sample.resultType === "all-pass") {
    if (
      sample.candidateFinalTeam !== "no-contract" ||
      sample.teamPointCardsRegressionMask ||
      sample.candidateTeamPointCards !== null ||
      sample.napoleonSidePointCards !== null ||
      sample.coalitionSidePointCards !== null ||
      sample.finalDeclaredTarget !== null ||
      sample.finalDeclaredSuit !== null ||
      sample.contractMargin !== null
    ) {
      throw new Error("Bidding Q counterfactual all-pass point-card target must be masked.");
    }
  } else {
    if (
      sample.candidateFinalTeam === "no-contract" ||
      !sample.teamPointCardsRegressionMask ||
      sample.candidateTeamPointCards === null ||
      sample.napoleonSidePointCards === null ||
      sample.coalitionSidePointCards === null ||
      sample.finalDeclaredTarget === null ||
      sample.finalDeclaredSuit === null
    ) {
      throw new Error("Bidding Q counterfactual standard result is missing point-card target.");
    }
  }
  if (sample.rawTerminalReward !== sample.terminalReward) {
    throw new Error("Bidding Q counterfactual sample uses identity reward transform.");
  }
  if (!sample.provenance.replayMatchedModelInput || !sample.provenance.replayMatchedLegalBidMask) {
    throw new Error("Bidding Q counterfactual sample replay parity was not confirmed.");
  }
  if (!sample.provenance.forcedOnce) {
    throw new Error("Bidding Q counterfactual sample forcedOnce must be true.");
  }
}

export function calculateBiddingQCounterfactualTerminalReward(input: {
  result: GameResult;
  actingPlayerId: PlayerId;
}): {
  reward: number;
  terminalRole: BiddingQTerminalRole;
  contractSuccess: boolean;
} {
  const role = terminalRoleForResult(input.result, input.actingPlayerId);
  if (input.result.resultType === "all-pass") {
    return {
      reward: 0,
      terminalRole: role,
      contractSuccess: false
    };
  }
  const contractSuccess = input.result.winner === "napoleon-team";
  if (role !== "napoleon" && role !== "napoleon-adjutant") {
    return {
      reward: 0,
      terminalRole: role,
      contractSuccess
    };
  }
  if (!contractSuccess) {
    return {
      reward: -1,
      terminalRole: role,
      contractSuccess
    };
  }
  return {
    reward: role === "napoleon-adjutant"
      ? 3 * input.result.targetPointCards
      : 2 * input.result.targetPointCards,
    terminalRole: role,
    contractSuccess
  };
}

export function encodeBiddingQActionIndex(targetPointCards: number, suit: Suit): number {
  if (targetPointCards < minBidTargetPointCards || targetPointCards > maxBidTargetPointCards) {
    throw new Error("Bidding Q targetPointCards is out of range.");
  }
  const suitIndex = SUITS.indexOf(suit);
  if (suitIndex === -1) {
    throw new Error("Bidding Q suit is unsupported.");
  }
  return 1 + (targetPointCards - minBidTargetPointCards) * SUITS.length + suitIndex;
}

export function decodeBiddingQActionIndex(actionIndex: number): BiddingQSemanticAction {
  validateBiddingActionIndex(actionIndex);
  if (actionIndex === 0) {
    return { type: "pass" };
  }
  const offset = actionIndex - 1;
  return {
    type: "bid",
    targetPointCards: minBidTargetPointCards + Math.floor(offset / SUITS.length),
    suit: SUITS[offset % SUITS.length]
  };
}

export function summarizeBiddingQCounterfactualSamples(
  samples: readonly BiddingQCounterfactualSample[],
  context: {
    sourceStates: readonly SourceBiddingState[];
    plannedActions: readonly PlannedStateAction[];
    fallbackCount: number;
    illegalActionCount: number;
  }
): BiddingQCounterfactualDatasetSummary {
  const suitCounts = emptySuitCounts();
  const targetCounts = emptyTargetCounts();
  const actionIndexCounts = Object.fromEntries(ACTION_INDEXES.map((index) => [String(index), 0]));
  const strongestSuitCounts = emptySuitCounts();
  const strongestByForcedSuitCounts = Object.fromEntries(
    SUITS.map((suit) => [suit, emptySuitCounts()])
  ) as Record<Suit, Record<Suit, number>>;
  const contractSuccessBySuit = Object.fromEntries(
    SUITS.map((suit) => [suit, { count: 0, successCount: 0, successRate: null }])
  ) as Record<Suit, BiddingQRateSummary>;
  const contractSuccessByTarget = Object.fromEntries(
    TARGETS.map((target) => [String(target), { count: 0, successCount: 0, successRate: null }])
  ) as Record<string, BiddingQRateSummary>;
  const resultTypeCounts: Record<GameResult["resultType"], number> = {
    standard: 0,
    "all-pass": 0
  };
  const terminalRoleCounts = emptyTerminalRoleCounts();
  const finalRoleByForcedActionType = {
    pass: emptyTerminalRoleCounts(),
    bid: emptyTerminalRoleCounts()
  };
  const finalRoleBySuit = Object.fromEntries(
    SUITS.map((suit) => [suit, emptyTerminalRoleCounts()])
  ) as Record<Suit, Record<BiddingQTerminalRole, number>>;
  const finalRoleByTarget = Object.fromEntries(
    TARGETS.map((target) => [String(target), emptyTerminalRoleCounts()])
  ) as Record<string, Record<BiddingQTerminalRole, number>>;
  const finalRoleByOpponentConfiguration: Record<string, Record<BiddingQTerminalRole, number>> = {};
  const legalActionsSeen = new Set<number>();
  let passCount = 0;
  let bidCount = 0;
  const rewards: number[] = [];
  const candidateTeamPointCards: number[] = [];
  const contractMargins: number[] = [];

  for (const source of context.sourceStates) {
    for (const [index, value] of source.legalBidMask.entries()) {
      if (isLegalMaskValue(value)) {
        legalActionsSeen.add(index);
      }
    }
    strongestSuitCounts[source.strongestSuit] += 1;
  }

  const sampledActions = new Set<number>();
  for (const sample of samples) {
    validateBiddingQCounterfactualSample(sample);
    actionIndexCounts[String(sample.forcedActionIndex)] += 1;
    sampledActions.add(sample.forcedActionIndex);
    resultTypeCounts[sample.resultType] += 1;
    terminalRoleCounts[sample.terminalRole] += 1;
    finalRoleByForcedActionType[sample.forcedAction.type][sample.finalRole] += 1;
    const opponentCounts =
      finalRoleByOpponentConfiguration[sample.opponentConfigurationKey] ?? emptyTerminalRoleCounts();
    opponentCounts[sample.finalRole] += 1;
    finalRoleByOpponentConfiguration[sample.opponentConfigurationKey] = opponentCounts;
    rewards.push(sample.terminalReward);
    if (sample.teamPointCardsRegressionMask) {
      if (sample.candidateTeamPointCards === null) {
        throw new Error("Regression sample is missing candidateTeamPointCards.");
      }
      candidateTeamPointCards.push(sample.candidateTeamPointCards);
    }
    if (sample.contractMargin !== null) {
      contractMargins.push(sample.contractMargin);
    }
    const action = sample.forcedAction;
    if (action.type === "pass") {
      passCount += 1;
      continue;
    }
    bidCount += 1;
    if (action.suit === undefined || action.targetPointCards === undefined) {
      throw new Error("Bid sample semantic action is incomplete.");
    }
    suitCounts[action.suit] += 1;
    targetCounts[String(action.targetPointCards)] += 1;
    finalRoleBySuit[action.suit][sample.finalRole] += 1;
    finalRoleByTarget[String(action.targetPointCards)][sample.finalRole] += 1;
    strongestByForcedSuitCounts[sample.strongestSuit][action.suit] += 1;
    const suitRate = contractSuccessBySuit[action.suit];
    suitRate.count += 1;
    suitRate.successCount += sample.contractSuccess ? 1 : 0;
    const targetRate = contractSuccessByTarget[String(action.targetPointCards)];
    targetRate.count += 1;
    targetRate.successCount += sample.contractSuccess ? 1 : 0;
  }
  for (const value of Object.values(contractSuccessBySuit)) {
    value.successRate = safeRate(value.successCount, value.count);
  }
  for (const value of Object.values(contractSuccessByTarget)) {
    value.successRate = safeRate(value.successCount, value.count);
  }

  const legalButNeverSampledActionIndexes = [...legalActionsSeen]
    .filter((index) => !sampledActions.has(index))
    .sort((left, right) => left - right);
  return {
    totalSourceStates: context.sourceStates.length,
    totalForcedStateActionPairs: context.plannedActions.length,
    totalRolloutSamples: samples.length,
    fallbackCount: context.fallbackCount,
    illegalActionCount: context.illegalActionCount,
    passCount,
    bidCount,
    suitCounts,
    targetCounts,
    actionIndexCounts,
    strongestSuitCounts,
    strongestByForcedSuitCounts,
    legalButNeverSampledActionCount: legalButNeverSampledActionIndexes.length,
    legalButNeverSampledActionIndexes,
    terminalReward: summarizeNumbers(rewards),
    contractSuccessBySuit,
    contractSuccessByTarget,
    resultTypeCounts,
    terminalRoleCounts,
    finalRoleByForcedActionType,
    finalRoleBySuit,
    finalRoleByTarget,
    finalRoleByOpponentConfiguration,
    candidateTeamPointCards: {
      count: candidateTeamPointCards.length,
      ...summarizeNumbers(candidateTeamPointCards)
    },
    contractMargin: {
      count: contractMargins.length,
      ...summarizeNumbers(contractMargins)
    }
  };
}

async function collectSourceStates(
  options: GenerateBiddingQCounterfactualDatasetOptions
): Promise<SourceBiddingState[]> {
  const states: SourceBiddingState[] = [];
  const maxSourceStates = options.maxSourceStates ?? DEFAULT_MAX_SOURCE_STATES;
  for (let seedOffset = 0; seedOffset < options.logicalSeedCount; seedOffset += 1) {
    const sourceSeed = options.startSeed + seedOffset;
    for (const candidateSeatIndex of NON_PLAYING_RL_ROTATION_OFFSETS) {
      const drafts = await runSourceGame({ options, sourceSeed, candidateSeatIndex });
      for (const draft of drafts) {
        const strongest = strongestSuitForModelInput(draft.modelInput);
        states.push({
          stateKey: createStateKey({
            sourceSeed,
            candidateSeatIndex,
            actingPlayerId: draft.playerId,
            biddingStep: draft.step,
            modelInput: draft.modelInput
          }),
          sourceSeed,
          sourceGameSeed: sourceSeed,
          candidateSeatIndex,
          actingPlayerId: draft.playerId,
          actingPlayerIndex: candidateSeatIndex,
          biddingStep: draft.step,
          modelInput: draft.modelInput,
          legalBidMask: draft.legalBidMask,
          selectedActionIndex: draft.selectedActionIndex,
          selectedAction: decodeBiddingQActionIndex(draft.selectedActionIndex),
          topPolicyActionIndex: draft.topPolicyActionIndex,
          strongestSuit: strongest.suit,
          strongestSuitScore: strongest.score
        });
        if (states.length >= maxSourceStates) {
          return states;
        }
      }
    }
  }
  return states;
}

async function runSourceGame(input: {
  options: GenerateBiddingQCounterfactualDatasetOptions;
  sourceSeed: number;
  candidateSeatIndex: number;
}): Promise<SourceBiddingDraft[]> {
  const drafts: SourceBiddingDraft[] = [];
  await runAutomatedGame({
    seed: input.sourceSeed,
    createAgent: ({ rng, playerIndex }) => {
      if (playerIndex === input.candidateSeatIndex) {
        return new SourceCandidateAgent({
          options: input.options,
          rng,
          recordDraft: (draft) => drafts.push(draft)
        });
      }
      return createFrozenAgent({
        options: input.options,
        rng,
        sourceSeed: input.sourceSeed,
        candidateSeatIndex: input.candidateSeatIndex,
        playerIndex
      });
    }
  });
  return drafts;
}

function createForcedActionPlan(
  sourceStates: readonly SourceBiddingState[],
  options: {
    randomSeed: number;
    randomLegalBidCount: number;
    actionPlanId?: BiddingQCounterfactualActionPlanId;
  }
): PlannedStateAction[] {
  const planned: PlannedStateAction[] = [];
  const targetCounts = emptyTargetCounts();
  for (const source of sourceStates) {
    if (options.actionPlanId === BIDDING_Q_COUNTERFACTUAL_ROLE_VALUE_ACTION_PLAN_ID) {
      for (const actionIndex of legalActionIndexes(source.legalBidMask)) {
        planned.push({ source, forcedActionIndex: actionIndex });
      }
      continue;
    }
    const indexes = new Set<number>();
    addIfLegal(indexes, source.legalBidMask, 0);
    const lowestTarget = lowestLegalTarget(source.legalBidMask);
    if (lowestTarget !== null) {
      for (const suit of SUITS) {
        addIfLegal(indexes, source.legalBidMask, encodeBiddingQActionIndex(lowestTarget, suit));
      }
      addIfLegal(
        indexes,
        source.legalBidMask,
        encodeBiddingQActionIndex(lowestTarget, source.strongestSuit)
      );
    }
    addIfLegal(indexes, source.legalBidMask, source.topPolicyActionIndex);

    const missingTarget = TARGETS.find((target) => targetCounts[String(target)] === 0);
    if (missingTarget !== undefined) {
      const supplement = firstLegalBidForTarget(source.legalBidMask, missingTarget, source.strongestSuit);
      if (supplement !== null) {
        indexes.add(supplement);
      }
    }

    for (const actionIndex of selectRandomLegalBidIndexes(source, options)) {
      indexes.add(actionIndex);
    }
    const sorted = [...indexes].sort((left, right) => left - right);
    for (const actionIndex of sorted) {
      const decoded = decodeBiddingQActionIndex(actionIndex);
      if (decoded.type === "bid" && decoded.targetPointCards !== undefined) {
        targetCounts[String(decoded.targetPointCards)] += 1;
      }
      planned.push({ source, forcedActionIndex: actionIndex });
    }
  }
  return planned;
}

async function runForcedRollout(input: {
  options: GenerateBiddingQCounterfactualDatasetOptions;
  source: SourceBiddingState;
  forcedActionIndex: number;
  repeatIndex: number;
  rolloutSeed: number;
}): Promise<ForcedRolloutResult> {
  if (!isLegalMaskValue(input.source.legalBidMask[input.forcedActionIndex])) {
    throw new Error(`Forced action ${input.forcedActionIndex} is illegal for ${input.source.stateKey}.`);
  }
  const metrics = {
    fallbackCount: 0,
    illegalActionCount: 0
  };
  let forcedCount = 0;
  const record = await runAutomatedGame({
    seed: input.source.sourceGameSeed,
    createAgent: ({ rng, playerIndex }) => {
      if (playerIndex === input.source.candidateSeatIndex) {
        return new ForcedCandidateAgent({
          options: input.options,
          source: input.source,
          forcedActionIndex: input.forcedActionIndex,
          repeatRng: createSeededRandom(input.rolloutSeed),
          rng,
          onForced: () => {
            forcedCount += 1;
          }
        });
      }
      return createFrozenAgent({
        options: input.options,
        rng,
        sourceSeed: input.source.sourceSeed,
        candidateSeatIndex: input.source.candidateSeatIndex,
        playerIndex
      });
    }
  });
  if (forcedCount !== 1) {
    throw new Error(`Forced action was applied ${forcedCount} times for ${input.source.stateKey}.`);
  }
  const reward = calculateBiddingQCounterfactualTerminalReward({
    result: record.result,
    actingPlayerId: input.source.actingPlayerId
  });
  const outcome = summarizeCandidatePointCardOutcome({
    result: record.result,
    decisions: record.decisions,
    actingPlayerId: input.source.actingPlayerId,
    terminalRole: reward.terminalRole
  });
  const opponentCounts = frozenOpponentCounts({
    seed: input.source.sourceSeed,
    candidateSeatIndex: input.source.candidateSeatIndex
  });
  const sample: BiddingQCounterfactualSample = {
    sampleType: BIDDING_Q_COUNTERFACTUAL_DATASET_SAMPLE_TYPE,
    schemaVersion: BIDDING_Q_COUNTERFACTUAL_SAMPLE_SCHEMA_VERSION,
    stateKey: input.source.stateKey,
    sourceSeed: input.source.sourceSeed,
    sourceGameSeed: input.source.sourceGameSeed,
    candidateSeatIndex: input.source.candidateSeatIndex,
    actingPlayerId: input.source.actingPlayerId,
    actingPlayerIndex: input.source.actingPlayerIndex,
    biddingStep: input.source.biddingStep,
    sourceSelectedActionIndex: input.source.selectedActionIndex,
    sourceSelectedAction: input.source.selectedAction,
    modelInput: [...input.source.modelInput],
    legalBidMask: [...input.source.legalBidMask],
    forcedActionIndex: input.forcedActionIndex,
    forcedAction: decodeBiddingQActionIndex(input.forcedActionIndex),
    strongestSuit: input.source.strongestSuit,
    strongestSuitScore: input.source.strongestSuitScore,
    actionPlanId: input.options.actionPlanId ?? BIDDING_Q_COUNTERFACTUAL_ACTION_PLAN_ID,
    repeatIndex: input.repeatIndex,
    rolloutSeed: input.rolloutSeed,
    terminalReward: reward.reward,
    rawTerminalReward: reward.reward,
    terminalRole: reward.terminalRole,
    contractSuccess: reward.contractSuccess,
    resultType: record.result.resultType,
    finalRole: reward.terminalRole,
    candidateFinalTeam: outcome.candidateFinalTeam,
    napoleonSidePointCards: outcome.napoleonSidePointCards,
    coalitionSidePointCards: outcome.coalitionSidePointCards,
    candidateTeamPointCards: outcome.candidateTeamPointCards,
    teamPointCardsRegressionMask: outcome.teamPointCardsRegressionMask,
    finalDeclaredTarget: outcome.finalDeclaredTarget,
    finalDeclaredSuit: outcome.finalDeclaredSuit,
    contractMargin: outcome.contractMargin,
    frozenBiddingOpponentCounts: opponentCounts,
    opponentConfigurationKey: opponentConfigurationKey(opponentCounts),
    result: summarizeResult(record.result),
    provenance: {
      sourceStateKey: input.source.stateKey,
      sourceSeed: input.source.sourceSeed,
      sourceGameSeed: input.source.sourceGameSeed,
      sourceBiddingStep: input.source.biddingStep,
      replayMatchedModelInput: true,
      replayMatchedLegalBidMask: true,
      forcedOnce: true
    }
  };
  validateBiddingQCounterfactualSample(sample);
  return {
    sample,
    fallbackCount: metrics.fallbackCount,
    illegalActionCount: metrics.illegalActionCount
  };
}

class SourceCandidateAgent implements Agent {
  private readonly ruleBasedAgent: RuleBasedAgent;

  constructor(
    private readonly input: {
      options: GenerateBiddingQCounterfactualDatasetOptions;
      rng: () => number;
      recordDraft: (draft: SourceBiddingDraft) => void;
    }
  ) {
    this.ruleBasedAgent = new RuleBasedAgent(input.rng);
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
      case "choosing-adjutant":
        return selectAdjutantAction({
          observation,
          policy: this.input.options.fixedAdjutantPolicy,
          fallback: this.ruleBasedAgent
        });
      case "exchanging":
        return selectExchangeAction({
          observation,
          policy: this.input.options.fixedExchangePolicy,
          fallback: this.ruleBasedAgent
        });
      case "playing":
        return selectPlayingAction({
          observation,
          context,
          policy: this.input.options.playingPolicy
        });
      case "finished":
        return this.ruleBasedAgent.selectAction(observation);
    }
  }

  private async selectBiddingAction(observation: PlayerObservation): Promise<GameAction> {
    const encoded = encodeBiddingObservation(
      observation,
      observation.view.players.map((player) => player.id)
    );
    const { modelInput, legalBidMask } = createBiddingModelInput(encoded);
    const logits = await this.input.options.biddingPolicy.predictLogits(modelInput);
    const selection = sampleMaskedCategoricalAction({
      logits,
      legalMask: legalBidMask,
      rng: this.input.rng
    });
    const topPolicyActionIndex = selectLegalIndex(logits, legalBidMask);
    const selectedAction = decodeBiddingAction(selection.selectedIndex, observation.playerId);
    const legalAction = findMatchingLegalAction(observation, selectedAction);
    this.input.recordDraft({
      step: nextDecisionStep(observation),
      playerId: observation.playerId,
      modelInput: [...modelInput],
      legalBidMask: [...legalBidMask],
      selectedActionIndex: selection.selectedIndex,
      topPolicyActionIndex
    });
    return legalAction;
  }
}

class ForcedCandidateAgent implements Agent {
  private readonly ruleBasedAgent: RuleBasedAgent;
  private forced = false;

  constructor(
    private readonly input: {
      options: GenerateBiddingQCounterfactualDatasetOptions;
      source: SourceBiddingState;
      forcedActionIndex: number;
      repeatRng: () => number;
      rng: () => number;
      onForced: () => void;
    }
  ) {
    this.ruleBasedAgent = new RuleBasedAgent(input.rng);
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
      case "choosing-adjutant":
        return selectAdjutantAction({
          observation,
          policy: this.input.options.fixedAdjutantPolicy,
          fallback: this.ruleBasedAgent
        });
      case "exchanging":
        return selectExchangeAction({
          observation,
          policy: this.input.options.fixedExchangePolicy,
          fallback: this.ruleBasedAgent
        });
      case "playing":
        return selectPlayingAction({
          observation,
          context,
          policy: this.input.options.playingPolicy
        });
      case "finished":
        return this.ruleBasedAgent.selectAction(observation);
    }
  }

  private async selectBiddingAction(observation: PlayerObservation): Promise<GameAction> {
    const encoded = encodeBiddingObservation(
      observation,
      observation.view.players.map((player) => player.id)
    );
    const { modelInput, legalBidMask } = createBiddingModelInput(encoded);
    const logits = await this.input.options.biddingPolicy.predictLogits(modelInput);
    const step = nextDecisionStep(observation);
    if (
      !this.forced &&
      step === this.input.source.biddingStep &&
      observation.playerId === this.input.source.actingPlayerId
    ) {
      assertSameNumberArray(Array.from(modelInput), this.input.source.modelInput, "modelInput", 1e-6);
      assertSameNumberArray(legalBidMask, this.input.source.legalBidMask, "legalBidMask", 0);
      if (!isLegalMaskValue(legalBidMask[this.input.forcedActionIndex])) {
        throw new Error(`Forced action ${this.input.forcedActionIndex} is illegal at replay.`);
      }
      consumeSamplingRngIfNeeded(logits, legalBidMask, this.input.rng);
      const forcedAction = decodeBiddingAction(this.input.forcedActionIndex, observation.playerId);
      const legalAction = findMatchingLegalAction(observation, forcedAction);
      this.forced = true;
      this.input.onForced();
      return legalAction;
    }
    const selection = sampleMaskedCategoricalAction({
      logits,
      legalMask: legalBidMask,
      rng: this.forced ? this.input.repeatRng : this.input.rng
    });
    return findMatchingLegalAction(
      observation,
      decodeBiddingAction(selection.selectedIndex, observation.playerId)
    );
  }
}

class FrozenCounterfactualAgent implements Agent {
  private readonly ruleBasedAgent: RuleBasedAgent;
  private readonly conservativeBiddingAgent: ConservativeBiddingAgent;

  constructor(
    private readonly input: {
      options: GenerateBiddingQCounterfactualDatasetOptions;
      rng: () => number;
      biddingPolicyType: "rule-based" | "conservative-bidding";
    }
  ) {
    this.ruleBasedAgent = new RuleBasedAgent(input.rng);
    this.conservativeBiddingAgent = new ConservativeBiddingAgent(input.rng);
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
        return this.input.biddingPolicyType === "conservative-bidding"
          ? this.conservativeBiddingAgent.selectAction(observation)
          : this.ruleBasedAgent.selectAction(observation);
      case "choosing-adjutant":
        return selectAdjutantAction({
          observation,
          policy: this.input.options.fixedAdjutantPolicy,
          fallback: this.ruleBasedAgent
        });
      case "exchanging":
        return selectExchangeAction({
          observation,
          policy: this.input.options.fixedExchangePolicy,
          fallback: this.ruleBasedAgent
        });
      case "playing":
        return selectPlayingAction({
          observation,
          context,
          policy: this.input.options.playingPolicy
        });
      case "finished":
        return this.ruleBasedAgent.selectAction(observation);
    }
  }
}

function createFrozenAgent(input: {
  options: GenerateBiddingQCounterfactualDatasetOptions;
  rng: () => number;
  sourceSeed: number;
  candidateSeatIndex: number;
  playerIndex: number;
}): Agent {
  const policy = selectBiddingQFrozenOpponentPolicy({
    seed: input.sourceSeed,
    candidateSeatIndex: input.candidateSeatIndex,
    playerIndex: input.playerIndex
  });
  return new FrozenCounterfactualAgent({
    options: input.options,
    rng: input.rng,
    biddingPolicyType: policy.type
  });
}

export function createBiddingQFrozenOpponentMixMetadata(): BiddingQFrozenOpponentMixMetadata {
  return {
    type: "mixed-frozen-bidding",
    mixingRuleVersion: BIDDING_Q_COUNTERFACTUAL_OPPONENT_MIX_RULE_VERSION,
    selectionUnit: "game-seat",
    topology: "candidate-x1-frozen-x4-v1",
    selectionSeed: "sourceSeed",
    ruleBasedWeight: 0.5,
    conservativeWeight: 0.5,
    policies: {
      ruleBased: {
        type: "rule-based",
        id: RULE_BASED_BIDDING_BASELINE_ID,
        version: RULE_BASED_AGENT_VERSION
      },
      conservative: {
        type: "conservative-bidding",
        id: "conservative-bidding-v1"
      }
    }
  };
}

export function selectBiddingQFrozenOpponentPolicy(input: {
  seed: number;
  candidateSeatIndex: number;
  playerIndex: number;
}): BiddingQFrozenOpponentPolicyMetadata {
  if (input.playerIndex === input.candidateSeatIndex) {
    throw new Error("candidate seat cannot be assigned a frozen bidding opponent policy.");
  }
  const digest = createHash("sha256")
    .update(
      `${BIDDING_Q_COUNTERFACTUAL_OPPONENT_MIX_RULE_VERSION}:${input.seed}:${input.candidateSeatIndex}:${input.playerIndex}`
    )
    .digest();
  const bucket = digest.readUInt32BE(0) % 2;
  return bucket === 0
    ? {
        type: "rule-based",
        id: RULE_BASED_BIDDING_BASELINE_ID,
        version: RULE_BASED_AGENT_VERSION
      }
    : {
        type: "conservative-bidding",
        id: "conservative-bidding-v1"
      };
}

function frozenOpponentCounts(input: {
  seed: number;
  candidateSeatIndex: number;
}): BiddingQOpponentCounts {
  const counts: BiddingQOpponentCounts = { ruleBased: 0, conservative: 0 };
  for (let playerIndex = 0; playerIndex < PLAYER_COUNT; playerIndex += 1) {
    if (playerIndex === input.candidateSeatIndex) {
      continue;
    }
    const policy = selectBiddingQFrozenOpponentPolicy({
      seed: input.seed,
      candidateSeatIndex: input.candidateSeatIndex,
      playerIndex
    });
    if (policy.type === "rule-based") {
      counts.ruleBased += 1;
    } else {
      counts.conservative += 1;
    }
  }
  return counts;
}

function opponentConfigurationKey(counts: BiddingQOpponentCounts): string {
  return `ruleBased=${counts.ruleBased},conservative=${counts.conservative}`;
}

async function selectAdjutantAction(input: {
  observation: PlayerObservation;
  policy: NonPlayingAdjutantRlPolicy | undefined;
  fallback: RuleBasedAgent;
}): Promise<GameAction> {
  if (input.policy === undefined) {
    return input.fallback.selectAction(input.observation);
  }
  if (input.observation.publicActionHistory === undefined) {
    throw new Error("Fixed adjutant policy input requires publicActionHistory.");
  }
  const absolutePlayerIds = input.observation.view.players.map((player) => player.id);
  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, input.observation.playerId);
  const biddingHistory = encodeBiddingHistoryFromPublicActions(
    input.observation.publicActionHistory,
    relativePlayerIds
  );
  const encoded = encodeAdjutantObservation(
    input.observation,
    absolutePlayerIds,
    biddingHistory
  );
  const { modelInput, legalAdjutantMask } = createAdjutantModelInput(encoded);
  const logits = await input.policy.predictLogits(modelInput);
  const selectedIndex = selectLegalIndex(logits, legalAdjutantMask);
  return findMatchingLegalAction(
    input.observation,
    decodeAdjutantAction(selectedIndex, input.observation.playerId)
  );
}

async function selectExchangeAction(input: {
  observation: PlayerObservation;
  policy: NonPlayingExchangeRlPolicy | undefined;
  fallback: RuleBasedAgent;
}): Promise<GameAction> {
  if (input.policy === undefined) {
    return input.fallback.selectAction(input.observation);
  }
  if (input.observation.publicActionHistory === undefined) {
    throw new Error("Fixed exchange policy input requires publicActionHistory.");
  }
  const absolutePlayerIds = input.observation.view.players.map((player) => player.id);
  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, input.observation.playerId);
  const biddingHistory = encodeBiddingHistoryFromPublicActions(
    input.observation.publicActionHistory,
    relativePlayerIds
  );
  const encoded = encodeExchangeObservation(
    input.observation,
    absolutePlayerIds,
    biddingHistory
  );
  const { modelInput, legalDiscardCardMask } = createExchangeModelInput(encoded);
  const logits = await input.policy.predictLogits(modelInput);
  const selectedCardIds = topLegalIndexes(logits, legalDiscardCardMask, 3).map((index) =>
    getCardId(index)
  );
  return {
    type: "discard-cards",
    playerId: input.observation.playerId,
    cardIds: selectedCardIds
  };
}

async function selectPlayingAction(input: {
  observation: PlayerObservation;
  context?: { actualState: ActualCardState; playerIds: readonly PlayerId[] };
  policy: FixedPlayingPolicy;
}): Promise<GameAction> {
  if (input.observation.publicActionHistory === undefined) {
    throw new Error("Bidding Q fixed playing policy requires publicActionHistory.");
  }
  const absolutePlayerIds =
    input.context?.playerIds ?? input.observation.view.players.map((player) => player.id);
  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, input.observation.playerId);
  const biddingHistory = encodeBiddingHistoryFromPublicActions(
    input.observation.publicActionHistory,
    relativePlayerIds
  );
  const encoded = encodePlayingObservation(input.observation, absolutePlayerIds, biddingHistory);
  const { modelInput, legalPlayMask } = createPlayingModelInput(encoded);
  const logits = await input.policy.predictLogits(modelInput);
  const selectedIndex = selectLegalIndex(logits, legalPlayMask);
  const selectedCardId = getCardId(selectedIndex);
  const action = input.observation.legalActions.find(
    (candidate) => candidate.type === "play-card" && candidate.cardId === selectedCardId
  );
  if (action === undefined) {
    throw new Error(`Fixed playing policy selected illegal card index ${selectedIndex}.`);
  }
  return action;
}

async function createBiddingQCounterfactualManifest(input: {
  options: GenerateBiddingQCounterfactualDatasetOptions;
  sourceStates: readonly SourceBiddingState[];
  plannedActions: readonly PlannedStateAction[];
  samples: readonly BiddingQCounterfactualSample[];
  shards: readonly DatasetShardManifest[];
  summary: BiddingQCounterfactualDatasetSummary;
}): Promise<BiddingQCounterfactualDatasetManifest> {
  const behaviorPolicy = await createPolicyArtifactManifest({
    type: "bidding-onnx",
    artifact: input.options.biddingPolicyArtifact,
    policy: input.options.biddingPolicy
  });
  const fixedPlayingPolicy = await createPolicyArtifactManifest({
    type: "playing-onnx",
    artifact: input.options.playingPolicyArtifact,
    policy: input.options.playingPolicy
  });
  const fixedAdjutantPolicy =
    input.options.fixedAdjutantPolicy !== undefined &&
    input.options.fixedAdjutantPolicyArtifact !== undefined
      ? await createPolicyArtifactManifest({
          type: "adjutant-onnx",
          artifact: input.options.fixedAdjutantPolicyArtifact,
          policy: input.options.fixedAdjutantPolicy
        })
      : undefined;
  const fixedExchangePolicy =
    input.options.fixedExchangePolicy !== undefined &&
    input.options.fixedExchangePolicyArtifact !== undefined
      ? await createPolicyArtifactManifest({
          type: "exchange-onnx",
          artifact: input.options.fixedExchangePolicyArtifact,
          policy: input.options.fixedExchangePolicy
        })
      : undefined;
  return {
    datasetSchemaVersion: BIDDING_Q_COUNTERFACTUAL_DATASET_SCHEMA_VERSION,
    generatorVersion: BIDDING_Q_COUNTERFACTUAL_DATASET_GENERATOR_VERSION,
    format: DATASET_FORMAT,
    sampleType: BIDDING_Q_COUNTERFACTUAL_DATASET_SAMPLE_TYPE,
    sampleSchemaVersion: BIDDING_Q_COUNTERFACTUAL_SAMPLE_SCHEMA_VERSION,
    compactObservation: {
      phase: "bidding",
      encoderSchemaVersion: BIDDING_ENCODER_SCHEMA_VERSION,
      modelInputSchemaVersion: BIDDING_MODEL_INPUT_SCHEMA_VERSION,
      modelInputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT
    },
    actionMapping: {
      id: BIDDING_Q_COUNTERFACTUAL_ACTION_MAPPING_ID,
      actionCount: BIDDING_ACTION_COUNT,
      passActionIndex: 0,
      bidTargets: TARGETS,
      suitOrder: SUITS
    },
    reward: {
      id: BIDDING_Q_COUNTERFACTUAL_REWARD_ID,
      type: BIDDING_Q_COUNTERFACTUAL_REWARD_TYPE,
      version: BIDDING_Q_COUNTERFACTUAL_REWARD_VERSION,
      napoleonWinMultiplier: 2,
      napoleonAdjutantWinMultiplier: 3,
      contractLossReward: -1,
      nonContractReward: 0
    },
    terminalRewardTransform: {
      id: BIDDING_Q_COUNTERFACTUAL_TERMINAL_REWARD_TRANSFORM_ID,
      type: "identity",
      version: 1,
      formula: "terminal_reward = raw_bidding_q_reward"
    },
    actionPlan: {
      id: input.options.actionPlanId ?? BIDDING_Q_COUNTERFACTUAL_ACTION_PLAN_ID,
      version: BIDDING_Q_COUNTERFACTUAL_ACTION_PLAN_VERSION,
      randomLegalBidCount: input.options.randomLegalBidCount ?? DEFAULT_RANDOM_LEGAL_BID_COUNT
    },
    predictionTarget: {
      id: BIDDING_Q_COUNTERFACTUAL_TEAM_POINT_CARDS_TARGET_ID,
      version: 1,
      roleLabel: "finalRole",
      valueLabel: "candidate-team-point-card-count",
      noContractHandling: "masked-null",
      candidateTeamDefinition: {
        napoleon: "napoleon-side",
        adjutant: "napoleon-side",
        citizen: "coalition-side",
        "napoleon-adjutant": "napoleon-side",
        noContract: "masked"
      }
    },
    repeats: input.options.repeats,
    sourceStates: input.sourceStates.length,
    forcedStateActionPairs: input.plannedActions.length,
    sampleCount: input.samples.length,
    startSeed: input.options.startSeed,
    endSeed: input.options.startSeed + input.options.logicalSeedCount - 1,
    logicalSeedCount: input.options.logicalSeedCount,
    actualSourceGameCount: input.options.logicalSeedCount * NON_PLAYING_RL_ROTATION_OFFSETS.length,
    candidateSeatRotation: NON_PLAYING_RL_ROTATION_OFFSETS,
    gamesPerShard: input.options.gamesPerShard ?? DEFAULT_GAMES_PER_SHARD,
    shardCount: input.shards.length,
    playerCount: PLAYER_COUNT,
    cardCount: CARD_COUNT,
    cardIds: CARD_IDS,
    cardIdsSha256: calculateCardIdsSha256(),
    simulation: {
      backend: BIDDING_Q_COUNTERFACTUAL_SIMULATION_BACKEND,
      inferenceDevice: input.options.inferenceDevice
    },
    opponentMix: createBiddingQFrozenOpponentMixMetadata(),
    behaviorPolicy,
    fixedPlayingPolicy,
    ...(fixedAdjutantPolicy === undefined ? {} : { fixedAdjutantPolicy }),
    ...(fixedExchangePolicy === undefined ? {} : { fixedExchangePolicy }),
    sourceCommit: input.options.sourceCommit ?? null,
    summary: input.summary,
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
    artifactId: input.artifact.artifactId ?? input.type,
    onnxFileName: basename(input.artifact.onnxPath),
    metadataFileName: basename(input.artifact.metadataPath),
    onnxSha256: await sha256File(input.artifact.onnxPath),
    metadataSha256: await sha256File(input.artifact.metadataPath),
    requestedInferenceDevice: input.policy.runtime?.requestedInferenceDevice,
    resolvedInferenceDevice: input.policy.runtime?.resolvedInferenceDevice,
    executionProvider: input.policy.runtime?.executionProvider,
    metadata: input.policy.metadata
  };
}

async function writeSampleShards(
  outputDirectory: string,
  samples: readonly BiddingQCounterfactualSample[],
  samplesPerShard: number
): Promise<DatasetShardManifest[]> {
  const shards: DatasetShardManifest[] = [];
  for (let start = 0; start < samples.length; start += samplesPerShard) {
    const chunk = samples.slice(start, start + samplesPerShard);
    const file = `shard-${String(shards.length).padStart(5, "0")}.jsonl`;
    const body = chunk.map(serializeBiddingQCounterfactualSample).join("");
    await writeFile(join(outputDirectory, file), body, "utf8");
    const seeds = chunk.map((sample) => sample.sourceSeed);
    shards.push({
      file,
      startSeed: Math.min(...seeds),
      endSeed: Math.max(...seeds),
      gameCount: new Set(chunk.map((sample) => sample.stateKey)).size,
      sampleCount: chunk.length,
      byteLength: Buffer.byteLength(body),
      sha256: sha256Utf8(body)
    });
  }
  return shards;
}

export function serializeBiddingQCounterfactualSample(
  sample: BiddingQCounterfactualSample
): string {
  validateBiddingQCounterfactualSample(sample);
  return `${JSON.stringify(sample)}\n`;
}

function strongestSuitForModelInput(modelInput: readonly number[]): { suit: Suit; score: number } {
  const hand = modelInput.slice(0, CARD_COUNT).flatMap((value, index) => {
    if (Number(value) !== 1) {
      return [];
    }
    const card = CARD_BY_ID.get(CARD_IDS[index]);
    if (card === undefined) {
      throw new Error(`Unknown card id in model input: ${CARD_IDS[index]}.`);
    }
    return [card];
  });
  let bestSuit = SUITS[0];
  let bestScore = evaluateHandForTrump(hand, bestSuit);
  for (const suit of SUITS.slice(1)) {
    const score = evaluateHandForTrump(hand, suit);
    if (score > bestScore) {
      bestSuit = suit;
      bestScore = score;
    }
  }
  return { suit: bestSuit, score: bestScore };
}

function terminalRoleForResult(result: GameResult, actingPlayerId: PlayerId): BiddingQTerminalRole {
  if (result.resultType === "all-pass") {
    return actingPlayerId === result.starterPlayerId ? "all-pass-starter" : "all-pass-other";
  }
  if (result.napoleonPlayerId === actingPlayerId && result.adjutantPlayerId === actingPlayerId) {
    return "napoleon-adjutant";
  }
  if (result.napoleonPlayerId === actingPlayerId) {
    return "napoleon";
  }
  if (result.adjutantPlayerId === actingPlayerId) {
    return "adjutant";
  }
  return "citizen";
}

function summarizeCandidatePointCardOutcome(input: {
  result: GameResult;
  decisions: readonly { phase: string; action: GameAction }[];
  actingPlayerId: PlayerId;
  terminalRole: BiddingQTerminalRole;
}): {
  candidateFinalTeam: "napoleon-team" | "alliance" | "no-contract";
  napoleonSidePointCards: number | null;
  coalitionSidePointCards: number | null;
  candidateTeamPointCards: number | null;
  teamPointCardsRegressionMask: boolean;
  finalDeclaredTarget: number | null;
  finalDeclaredSuit: Suit | null;
  contractMargin: number | null;
} {
  if (input.result.resultType === "all-pass") {
    return {
      candidateFinalTeam: "no-contract",
      napoleonSidePointCards: null,
      coalitionSidePointCards: null,
      candidateTeamPointCards: null,
      teamPointCardsRegressionMask: false,
      finalDeclaredTarget: null,
      finalDeclaredSuit: null,
      contractMargin: null
    };
  }
  const candidateFinalTeam = input.terminalRole === "citizen" ? "alliance" : "napoleon-team";
  const candidateTeamPointCards =
    candidateFinalTeam === "napoleon-team"
      ? input.result.napoleonTeamPointCards
      : input.result.alliancePointCards;
  const finalBid = finalDeclaredBid(input.decisions);
  if (finalBid === null) {
    throw new Error("Standard result must have a final bid.");
  }
  return {
    candidateFinalTeam,
    napoleonSidePointCards: input.result.napoleonTeamPointCards,
    coalitionSidePointCards: input.result.alliancePointCards,
    candidateTeamPointCards,
    teamPointCardsRegressionMask: true,
    finalDeclaredTarget: input.result.targetPointCards,
    finalDeclaredSuit: finalBid.suit,
    contractMargin:
      candidateFinalTeam === "napoleon-team"
        ? input.result.napoleonTeamPointCards - input.result.targetPointCards
        : null
  };
}

function finalDeclaredBid(
  decisions: readonly { phase: string; action: GameAction }[]
): { targetPointCards: number; suit: Suit } | null {
  for (const decision of [...decisions].reverse()) {
    if (decision.phase !== "bidding" || decision.action.type !== "bid") {
      continue;
    }
    return {
      targetPointCards: decision.action.targetPointCards,
      suit: decision.action.suit
    };
  }
  return null;
}

function summarizeResult(result: GameResult): BiddingQResultSummary {
  if (result.resultType === "all-pass") {
    return {
      starterPlayerId: result.starterPlayerId
    };
  }
  return {
    winner: result.winner,
    targetPointCards: result.targetPointCards,
    napoleonTeamPointCards: result.napoleonTeamPointCards,
    alliancePointCards: result.alliancePointCards,
    napoleonPlayerId: result.napoleonPlayerId,
    adjutantPlayerId: result.adjutantPlayerId
  };
}

function selectRandomLegalBidIndexes(
  source: SourceBiddingState,
  options: { randomSeed: number; randomLegalBidCount: number }
): number[] {
  const legalBids = legalActionIndexes(source.legalBidMask).filter((index) => index !== 0);
  const rng = createSeededRandom(
    stableUint32(`${options.randomSeed}:${source.stateKey}:random-legal-bids`)
  );
  const pool = [...legalBids];
  const selected: number[] = [];
  while (pool.length > 0 && selected.length < options.randomLegalBidCount) {
    const index = Math.floor(rng() * pool.length);
    selected.push(pool.splice(index, 1)[0]);
  }
  return selected;
}

function lowestLegalTarget(legalBidMask: readonly number[]): number | null {
  for (const target of TARGETS) {
    if (SUITS.some((suit) => isLegalMaskValue(legalBidMask[encodeBiddingQActionIndex(target, suit)]))) {
      return target;
    }
  }
  return null;
}

function firstLegalBidForTarget(
  legalBidMask: readonly number[],
  target: number,
  preferredSuit: Suit
): number | null {
  const suitOrder = [preferredSuit, ...SUITS.filter((suit) => suit !== preferredSuit)];
  for (const suit of suitOrder) {
    const actionIndex = encodeBiddingQActionIndex(target, suit);
    if (isLegalMaskValue(legalBidMask[actionIndex])) {
      return actionIndex;
    }
  }
  return null;
}

function addIfLegal(target: Set<number>, legalBidMask: readonly number[], actionIndex: number): void {
  if (isLegalMaskValue(legalBidMask[actionIndex])) {
    target.add(actionIndex);
  }
}

function deriveCounterfactualRolloutSeed(input: {
  randomSeed: number;
  source: SourceBiddingState;
  forcedActionIndex: number;
  repeatIndex: number;
}): number {
  return stableUint32([
    input.randomSeed,
    input.source.stateKey,
    input.forcedActionIndex,
    input.repeatIndex
  ].join(":"));
}

function createStateKey(input: {
  sourceSeed: number;
  candidateSeatIndex: number;
  actingPlayerId: PlayerId;
  biddingStep: number;
  modelInput: readonly number[];
}): string {
  const digest = createHash("sha256")
    .update(input.modelInput.join(","))
    .digest("hex")
    .slice(0, 16);
  return [
    `seed-${input.sourceSeed}`,
    `seat-${input.candidateSeatIndex}`,
    `player-${input.actingPlayerId}`,
    `step-${input.biddingStep}`,
    digest
  ].join(":");
}

function sampleMaskedCategoricalAction(input: {
  logits: Float32Array | readonly number[];
  legalMask: readonly number[];
  rng: () => number;
}): { selectedIndex: number; logProbability: number } {
  const distribution = createMaskedCategoricalDistribution(input.logits, input.legalMask);
  if (distribution.legalIndexes.length === 1) {
    return {
      selectedIndex: distribution.legalIndexes[0],
      logProbability: 0
    };
  }
  const value = input.rng();
  let cumulative = 0;
  for (let index = 0; index < distribution.legalIndexes.length; index += 1) {
    cumulative += distribution.probabilities[index];
    if (value < cumulative) {
      return {
        selectedIndex: distribution.legalIndexes[index],
        logProbability: distribution.logProbabilities[index]
      };
    }
  }
  const last = distribution.legalIndexes.length - 1;
  return {
    selectedIndex: distribution.legalIndexes[last],
    logProbability: distribution.logProbabilities[last]
  };
}

function consumeSamplingRngIfNeeded(
  logits: Float32Array | readonly number[],
  legalMask: readonly number[],
  rng: () => number
): void {
  const distribution = createMaskedCategoricalDistribution(logits, legalMask);
  if (distribution.legalIndexes.length > 1) {
    rng();
  }
}

function createMaskedCategoricalDistribution(
  logits: Float32Array | readonly number[],
  legalMask: readonly number[]
): {
  legalIndexes: number[];
  probabilities: number[];
  logProbabilities: number[];
} {
  const legalIndexes = legalActionIndexes(legalMask);
  if (legalIndexes.length === 0) {
    throw new Error("legalBidMask must contain at least one legal action.");
  }
  if (legalIndexes.length === 1) {
    return {
      legalIndexes,
      probabilities: [1],
      logProbabilities: [0]
    };
  }
  const legalLogits = legalIndexes.map((index) => Number(logits[index]));
  const maxLogit = Math.max(...legalLogits);
  const expValues = legalLogits.map((logit) => Math.exp(logit - maxLogit));
  const expSum = sum(expValues);
  const logDenominator = maxLogit + Math.log(expSum);
  return {
    legalIndexes,
    probabilities: expValues.map((value) => value / expSum),
    logProbabilities: legalLogits.map((logit) => logit - logDenominator)
  };
}

function selectLegalIndex(logits: Float32Array | readonly number[], legalMask: readonly number[]): number {
  const legal = legalActionIndexes(legalMask);
  if (legal.length === 0) {
    throw new Error("legal mask is empty.");
  }
  return legal.reduce((best, index) => {
    const value = Number(logits[index]);
    const bestValue = Number(logits[best]);
    if (!Number.isFinite(value)) {
      throw new Error(`logits[${index}] must be finite.`);
    }
    return value > bestValue ? index : best;
  }, legal[0]);
}

function topLegalIndexes(
  logits: Float32Array | readonly number[],
  legalMask: readonly number[],
  count: number
): number[] {
  return legalActionIndexes(legalMask)
    .map((index) => ({ index, value: Number(logits[index]) }))
    .sort((left, right) => right.value - left.value || left.index - right.index)
    .slice(0, count)
    .map((entry) => entry.index)
    .sort((left, right) => left - right);
}

function legalActionIndexes(legalMask: readonly number[]): number[] {
  if (legalMask.length !== BIDDING_ACTION_COUNT && legalMask.length !== CARD_COUNT) {
    throw new Error("legal mask length is unsupported.");
  }
  return [...legalMask.entries()].flatMap(([index, value]) =>
    isLegalMaskValue(value) ? [index] : []
  );
}

function findMatchingLegalAction(observation: PlayerObservation, selectedAction: GameAction): GameAction {
  const legalAction = observation.legalActions.find((action) => actionsEqual(action, selectedAction));
  if (legalAction === undefined) {
    throw new Error(`Selected action outside legal actions: ${JSON.stringify(selectedAction)}.`);
  }
  return legalAction;
}

function actionsEqual(left: GameAction, right: GameAction): boolean {
  if (left.type !== right.type || left.playerId !== right.playerId) {
    return false;
  }
  switch (left.type) {
    case "bid":
      return right.type === "bid" &&
        left.suit === right.suit &&
        left.targetPointCards === right.targetPointCards;
    case "pass":
      return right.type === "pass";
    case "play-card":
      return right.type === "play-card" && left.cardId === right.cardId;
    case "choose-adjutant":
      return right.type === "choose-adjutant" && left.cardId === right.cardId;
    case "discard-cards":
      return right.type === "discard-cards" &&
        sameStringArray([...left.cardIds].sort(), [...right.cardIds].sort());
  }
  return false;
}

function nextDecisionStep(observation: PlayerObservation): number {
  return (observation.publicActionHistory ?? []).length + 1;
}

function emptySuitCounts(): Record<Suit, number> {
  return {
    spades: 0,
    hearts: 0,
    diamonds: 0,
    clubs: 0
  };
}

function emptyTargetCounts(): Record<string, number> {
  return Object.fromEntries(TARGETS.map((target) => [String(target), 0]));
}

function emptyTerminalRoleCounts(): Record<BiddingQTerminalRole, number> {
  return {
    napoleon: 0,
    adjutant: 0,
    citizen: 0,
    "napoleon-adjutant": 0,
    "all-pass-starter": 0,
    "all-pass-other": 0
  };
}

function summarizeNumbers(values: readonly number[]): BiddingQCounterfactualDatasetSummary["terminalReward"] {
  if (values.length === 0) {
    return {
      mean: null,
      std: null,
      min: null,
      max: null
    };
  }
  const mean = sum(values) / values.length;
  const variance = sum(values.map((value) => (value - mean) ** 2)) / values.length;
  return {
    mean,
    std: Math.sqrt(variance),
    min: Math.min(...values),
    max: Math.max(...values)
  };
}

function validateBiddingQFrozenOpponentMixMetadata(
  metadata: BiddingQFrozenOpponentMixMetadata
): void {
  if (
    metadata.type !== "mixed-frozen-bidding" ||
    metadata.mixingRuleVersion !== BIDDING_Q_COUNTERFACTUAL_OPPONENT_MIX_RULE_VERSION ||
    metadata.selectionUnit !== "game-seat" ||
    metadata.topology !== "candidate-x1-frozen-x4-v1" ||
    metadata.selectionSeed !== "sourceSeed" ||
    metadata.ruleBasedWeight !== 0.5 ||
    metadata.conservativeWeight !== 0.5 ||
    metadata.policies.ruleBased.type !== "rule-based" ||
    metadata.policies.ruleBased.id !== RULE_BASED_BIDDING_BASELINE_ID ||
    metadata.policies.ruleBased.version !== RULE_BASED_AGENT_VERSION ||
    metadata.policies.conservative.type !== "conservative-bidding" ||
    metadata.policies.conservative.id !== "conservative-bidding-v1"
  ) {
    throw new Error("Bidding Q counterfactual frozen opponent mix mismatch.");
  }
}

async function ensureOutputDoesNotExist(outputDirectory: string): Promise<void> {
  try {
    await stat(outputDirectory);
  } catch (error) {
    if (isNodeErrorWithCode(error, "ENOENT")) {
      return;
    }
    throw error;
  }
  throw new Error(`Output directory already exists: ${outputDirectory}`);
}

async function sha256File(path: string): Promise<string> {
  const content = await readFile(path);
  return createHash("sha256").update(content).digest("hex");
}

function sha256Utf8(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function stableUint32(value: string): number {
  return createHash("sha256").update(value).digest().readUInt32BE(0);
}

function isLegalMaskValue(value: unknown): boolean {
  return value === 1 || value === true;
}

function validateBiddingActionIndex(value: number): void {
  if (!Number.isInteger(value) || value < 0 || value >= BIDDING_ACTION_COUNT) {
    throw new Error(`Bidding action index must be between 0 and ${BIDDING_ACTION_COUNT - 1}.`);
  }
}

function validateOutputDirectory(value: string): void {
  if (value.length === 0) {
    throw new Error("outputDirectory must not be empty.");
  }
}

function validateUint32(name: string, value: number): void {
  if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
    throw new Error(`${name} must be an unsigned 32-bit integer.`);
  }
}

function validatePositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function validateNonNegativeInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${name} must be a non-negative integer.`);
  }
}

function assertSameNumberArray(
  left: readonly number[],
  right: readonly number[],
  label: string,
  tolerance: number
): void {
  if (left.length !== right.length) {
    throw new Error(`${label} length mismatch: ${left.length} !== ${right.length}.`);
  }
  for (let index = 0; index < left.length; index += 1) {
    if (Math.abs(Number(left[index]) - Number(right[index])) > tolerance) {
      throw new Error(`${label} mismatch at ${index}: ${left[index]} !== ${right[index]}.`);
    }
  }
}

function sameNumberArray(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function sameStringArray(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function safeRate(numerator: number, denominator: number): number | null {
  return denominator === 0 ? null : numerator / denominator;
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function isNodeErrorWithCode(error: unknown, code: string): boolean {
  return typeof error === "object" &&
    error !== null &&
    "code" in error &&
    (error as { code: unknown }).code === code;
}
