import { createHash } from "node:crypto";
import { lstat, mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import {
  RuleBasedAgent,
  createSeededRandom,
  deriveSeed,
  normalizeSeed
} from "@napoleon/ai";
import type {
  ActualCardState,
  Agent,
  PlayerObservation,
  PublicActionRecord
} from "@napoleon/ai";
import {
  CARD_COUNT,
  CARD_IDS,
  EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
  PLAYER_COUNT,
  createExchangeModelInput,
  createRelativePlayerOrder,
  encodeBiddingHistoryFromPublicActions,
  encodeExchangeObservation
} from "@napoleon/ai-observation";
import {
  advanceToNextTrick,
  applyAction,
  clearLatestEvent,
  createInitialGame,
  createPlayerView,
  getLegalActions,
  getSeiJackCardId,
  getUraJackCardId,
  isPointCard,
  jokerCardId,
  orumaCardId,
  yoromekiCardId
} from "@napoleon/game-core";
import type {
  Card,
  DiscardCardsAction,
  GameAction,
  GameState,
  PlayerId,
  Suit,
  StandardGameResult
} from "@napoleon/game-core";
import type { DatasetShardManifest } from "./types.js";
import {
  calculateCardIdsSha256,
  serializeManifest,
  sha256Utf8
} from "./serialization.js";

export const EXCHANGE_COUNTERFACTUAL_DATASET_FORMAT = "jsonl" as const;
export const EXCHANGE_COUNTERFACTUAL_DATASET_SAMPLE_TYPE =
  "exchange-counterfactual-value-v1" as const;
export const EXCHANGE_COUNTERFACTUAL_DATASET_SCHEMA_VERSION = 1 as const;
export const EXCHANGE_COUNTERFACTUAL_SAMPLE_SCHEMA_VERSION = 1 as const;
export const EXCHANGE_COUNTERFACTUAL_DATASET_GENERATOR_VERSION = 1 as const;
export const EXCHANGE_COUNTERFACTUAL_TEACHER_ID =
  "exchange-discard-combination-playing-rollout-v1" as const;
export const EXCHANGE_COUNTERFACTUAL_DISCARD_ACTION_SPACE_ID =
  "exchange-unordered-13c3-discard-combinations-v1" as const;
export const EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT = 286 as const;

const DEFAULT_PLAYER_IDS: readonly PlayerId[] = [
  "player-0",
  "player-1",
  "player-2",
  "player-3",
  "player-4"
];
const DEFAULT_STATES_PER_SHARD = 10;
const DEFAULT_MAX_DEAL_ATTEMPTS_FACTOR = 25;
const DEFAULT_MAX_DECISION_STEPS = 1000;
const COMBINATION_DISCARD_COUNT = 3;
const UINT32_MAX = 0xffffffff;

export interface ExchangeCounterfactualPolicyMetadata {
  id: string;
  description?: string;
  version?: string | number;
  provenance?: unknown;
}

export interface ExchangeCounterfactualAgentContext {
  playerId: PlayerId;
  playerIndex: number;
  rng: () => number;
  seed: number;
}

export interface GenerateExchangeCounterfactualDatasetOptions {
  outputDirectory: string;
  sourceStateCount: number;
  startSeed: number;
  statesPerShard?: number;
  maxDealAttempts?: number;
  maxDecisionSteps?: number;
  playerIds?: readonly PlayerId[];
  sourceCommit?: string | null;
  biddingPolicy?: ExchangeCounterfactualPolicyMetadata;
  adjutantPolicy?: ExchangeCounterfactualPolicyMetadata;
  playingPolicy?: ExchangeCounterfactualPolicyMetadata;
  playingPolicyDeterministic?: boolean;
  createBiddingAgent?: (context: ExchangeCounterfactualAgentContext) => Agent;
  createAdjutantAgent?: (context: ExchangeCounterfactualAgentContext) => Agent;
  createPlayingAgent?: (context: ExchangeCounterfactualAgentContext) => Agent;
  onProgress?: (progress: ExchangeCounterfactualGenerationProgress) => void;
}

export interface ExchangeCounterfactualGenerationProgress {
  completedSourceStates: number;
  requestedSourceStates: number;
  sampleCount: number;
  rolloutCount: number;
  completedShards: number;
  currentSeed: number;
  dealAttempts: number;
}

export interface ExchangeCounterfactualSample {
  sampleType: typeof EXCHANGE_COUNTERFACTUAL_DATASET_SAMPLE_TYPE;
  schemaVersion: typeof EXCHANGE_COUNTERFACTUAL_SAMPLE_SCHEMA_VERSION;
  sourceStateKey: string;
  fixedHandId: string;
  dealSeed: number;
  sourceIndex: number;
  candidateIndex: number;
  candidateKey: string;
  napoleonPlayerId: PlayerId;
  napoleonSeatIndex: number;
  contractTargetPointCards: number;
  contractSuit: string;
  calledAdjutantCardId: string;
  pickupHandCardIds: readonly string[];
  modelInput: readonly number[];
  legalDiscardCardMask: readonly number[];
  candidateDiscardCardIds: readonly string[];
  candidateDiscardMask: readonly number[];
  buriedPointCardCount: number;
  buriedTrumpCount: number;
  buriedSpecialCards: ExchangeCounterfactualSpecialBuriedFlags;
  contractMargin: number;
  contractSuccess: boolean;
  napoleonSidePointCards: number;
  napoleonRawReward: number;
  napoleonRelativeReward: number;
  ruleBasedDiscardCardIds: readonly string[];
  isRuleBasedAction: boolean;
  ruleBasedCandidateIndex: number;
  hiddenDealChecksum: string;
  biddingHistoryHash: string;
  biddingHistoryActionCount: number;
  invariantChecks: ExchangeCounterfactualInvariantChecks;
}

export interface ExchangeCounterfactualSpecialBuriedFlags {
  joker: boolean;
  oruma: boolean;
  yoromeki: boolean;
  seiJack: boolean;
  uraJack: boolean;
  calledAdjutant: boolean;
}

export interface ExchangeCounterfactualInvariantChecks {
  sourcePhaseExchanging: boolean;
  napoleonHandCount13: boolean;
  candidateCount286: boolean;
  candidateDiscardCount3: boolean;
  candidateDiscardUnique: boolean;
  candidateDiscardFromPickupHand: boolean;
  unorderedCombinationCanonical: boolean;
  modelInputFeatureCount2671: boolean;
  legalDiscardMaskMatchesPickupHand: boolean;
  sameSourceHiddenDeal: boolean;
  standardGameResult: boolean;
}

export interface ExchangeCounterfactualDatasetSummary {
  sourceStateCount: number;
  uniqueDealSeedCount: number;
  uniqueSourceStateCount: number;
  uniqueFixedHandCount: number;
  sampleCount: number;
  rolloutCount: number;
  candidateCountPerState: NumericSummary;
  marginSpread: NumericSummary;
  relativeRewardSpread: NumericSummary;
  ruleBasedMatchRate: number;
  ruleBasedMarginRegret: NumericSummary;
  ruleBasedRelativeRewardRegret: NumericSummary;
  ruleBasedRankPercentile: NumericSummary;
  bestBuriedPointCardCount: NumericSummary;
  ruleBasedBuriedPointCardCount: NumericSummary;
  bestSpecialCardBuryRate: ExchangeCounterfactualSpecialBuryRates;
  ruleBasedSpecialCardBuryRate: ExchangeCounterfactualSpecialBuryRates;
  rankingStability: ExchangeCounterfactualRankingStability;
  invariantFailureCount: number;
  sourceStateDiagnostics: readonly ExchangeCounterfactualSourceDiagnostic[];
}

export interface NumericSummary {
  count: number;
  min: number;
  max: number;
  mean: number;
  median: number;
}

export interface ExchangeCounterfactualSpecialBuryRates {
  joker: number;
  oruma: number;
  yoromeki: number;
  seiJack: number;
  uraJack: number;
  calledAdjutant: number;
  any: number;
}

export interface ExchangeCounterfactualRankingStability {
  method: "fixed-source-single-rollout-deterministic-policy";
  repeatedRolloutsPerCandidate: 1;
  deterministicPlayingPolicy: boolean;
  exactRepeatRankCorrelation: 1 | null;
  note: string;
}

export interface ExchangeCounterfactualSourceDiagnostic {
  sourceStateKey: string;
  dealSeed: number;
  candidateCount: number;
  ruleBasedCandidateIndex: number;
  ruleBasedBestMatch: boolean;
  ruleBasedRankByRelativeReward: number;
  ruleBasedRankPercentile: number;
  bestCandidateIndex: number;
  bestRelativeReward: number;
  ruleBasedRelativeReward: number;
  ruleBasedRelativeRewardRegret: number;
  bestMargin: number;
  ruleBasedMargin: number;
  ruleBasedMarginRegret: number;
  marginSpread: number;
  relativeRewardSpread: number;
  bestBuriedPointCardCount: number;
  ruleBasedBuriedPointCardCount: number;
  bestBuriedSpecialCards: ExchangeCounterfactualSpecialBuriedFlags;
  ruleBasedBuriedSpecialCards: ExchangeCounterfactualSpecialBuriedFlags;
}

export interface ExchangeCounterfactualDatasetManifest {
  datasetSchemaVersion: typeof EXCHANGE_COUNTERFACTUAL_DATASET_SCHEMA_VERSION;
  generatorVersion: typeof EXCHANGE_COUNTERFACTUAL_DATASET_GENERATOR_VERSION;
  format: typeof EXCHANGE_COUNTERFACTUAL_DATASET_FORMAT;
  sampleType: typeof EXCHANGE_COUNTERFACTUAL_DATASET_SAMPLE_TYPE;
  sampleSchemaVersion: typeof EXCHANGE_COUNTERFACTUAL_SAMPLE_SCHEMA_VERSION;
  teacherId: typeof EXCHANGE_COUNTERFACTUAL_TEACHER_ID;
  discardActionSpaceId: typeof EXCHANGE_COUNTERFACTUAL_DISCARD_ACTION_SPACE_ID;
  discardCombinationCount: typeof EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT;
  sourceFlow: readonly string[];
  modelInput: {
    observation: "existing-exchange-observation";
    featureCount: typeof EXCHANGE_MODEL_INPUT_FEATURE_COUNT;
    hiddenOpponentHandsIncluded: false;
  };
  teacherUsesCompleteHiddenState: true;
  permutationActionsIncluded: false;
  startSeed: number;
  endSeed: number;
  sourceStateCount: number;
  requestedSourceStateCount: number;
  sampleCount: number;
  rolloutCount: number;
  statesPerShard: number;
  dealAttemptCount: number;
  sourceCommit: string | null;
  playerCount: typeof PLAYER_COUNT;
  cardCount: typeof CARD_COUNT;
  cardIds: readonly string[];
  cardIdsSha256: string;
  biddingPolicy: ExchangeCounterfactualPolicyMetadata;
  adjutantPolicy: ExchangeCounterfactualPolicyMetadata;
  playingPolicy: ExchangeCounterfactualPolicyMetadata;
  reward: {
    raw: "napoleon contract success ? 2 * target : 0";
    relative: "napoleon contract success ? 7 * target / 4 : -3 * target / 4";
  };
  summary: ExchangeCounterfactualDatasetSummary;
  shardCount: number;
  shards: readonly DatasetShardManifest[];
}

export interface GenerateExchangeCounterfactualDatasetResult {
  outputDirectory: string;
  manifest: ExchangeCounterfactualDatasetManifest;
}

interface SourceExchangeState {
  sourceIndex: number;
  dealSeed: number;
  state: GameState;
  publicActionHistory: readonly PublicActionRecord[];
  hiddenDealChecksum: string;
  sourceStateKey: string;
  fixedHandId: string;
  modelInput: readonly number[];
  legalDiscardCardMask: readonly number[];
  pickupHandCards: readonly Card[];
  pickupHandCardIds: readonly string[];
  ruleBasedDiscardCardIds: readonly string[];
  ruleBasedCandidateIndex: number;
}

interface CompleteInfoActionAgent {
  selectActionWithContext: (
    observation: PlayerObservation,
    context: {
      actualState: ActualCardState;
      playerIds: readonly PlayerId[];
    }
  ) => Promise<GameAction>;
}

export async function generateExchangeCounterfactualDataset(
  options: GenerateExchangeCounterfactualDatasetOptions
): Promise<GenerateExchangeCounterfactualDatasetResult> {
  const validated = validateGenerateExchangeCounterfactualDatasetOptions(options);
  const outputDirectory = resolve(validated.outputDirectory);
  const outputParentDirectory = dirname(outputDirectory);
  await mkdir(outputParentDirectory, { recursive: true });
  await assertOutputDirectoryDoesNotExist(outputDirectory);
  const tempDirectory = await mkdtemp(
    join(outputParentDirectory, `.${basename(outputDirectory)}-staging-`)
  );
  const shards: DatasetShardManifest[] = [];
  let currentShardRows: ExchangeCounterfactualSample[] = [];
  let currentShardStartSeed = validated.startSeed;
  let dealAttempts = 0;
  let seed = validated.startSeed;
  let completedSourceStates = 0;
  let sampleCount = 0;
  let invariantFailureCount = 0;
  const sourceDiagnostics: ExchangeCounterfactualSourceDiagnostic[] = [];
  const dealSeeds = new Set<number>();
  const fixedHandIds = new Set<string>();

  try {
    while (completedSourceStates < validated.sourceStateCount) {
      if (dealAttempts >= validated.maxDealAttempts) {
        throw new Error(
          `Unable to collect ${validated.sourceStateCount} exchange states within ${validated.maxDealAttempts} deal attempts.`
        );
      }

      const source = await tryCreateSourceExchangeState({
        ...validated,
        dealSeed: seed,
        sourceIndex: completedSourceStates
      });
      dealAttempts += 1;
      seed += 1;

      if (source === null) {
        continue;
      }

      const sourceSamples = await createSourceSamples(source, validated);
      currentShardRows.push(...sourceSamples);
      sampleCount += sourceSamples.length;
      invariantFailureCount += sourceSamples.filter((sample) =>
        Object.values(sample.invariantChecks).some((value) => value === false)
      ).length;
      sourceDiagnostics.push(createSourceDiagnostic(sourceSamples));
      dealSeeds.add(source.dealSeed);
      fixedHandIds.add(source.fixedHandId);
      completedSourceStates += 1;

      if (currentShardRows.length >= validated.statesPerShard * EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT) {
        shards.push(
          await writeJsonlShard({
            directory: tempDirectory,
            rows: currentShardRows,
            shardIndex: shards.length,
            startSeed: currentShardStartSeed,
            endSeed: source.dealSeed,
            gameCount: currentShardRows.length / EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT
          })
        );
        currentShardRows = [];
        currentShardStartSeed = seed;
      }

      validated.onProgress?.({
        completedSourceStates,
        requestedSourceStates: validated.sourceStateCount,
        sampleCount,
        rolloutCount: sampleCount,
        completedShards: shards.length,
        currentSeed: seed - 1,
        dealAttempts
      });
    }

    if (currentShardRows.length > 0) {
      shards.push(
        await writeJsonlShard({
          directory: tempDirectory,
          rows: currentShardRows,
          shardIndex: shards.length,
          startSeed: currentShardStartSeed,
          endSeed: seed - 1,
          gameCount: currentShardRows.length / EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT
        })
      );
    }

    const summary = createSummaryFromDiagnostics({
      diagnostics: sourceDiagnostics,
      sampleCount,
      uniqueDealSeedCount: dealSeeds.size,
      uniqueFixedHandCount: fixedHandIds.size,
      invariantFailureCount,
      playingPolicyDeterministic: validated.playingPolicyDeterministic
    });
    const manifest = createManifest({
      options: validated,
      endSeed: seed - 1,
      dealAttempts,
      sampleCount,
      summary,
      shards
    });

    await writeFile(join(tempDirectory, "manifest.json"), serializeManifest(manifest), "utf8");
    await rename(tempDirectory, outputDirectory);

    return {
      outputDirectory,
      manifest
    };
  } catch (error) {
    await rm(tempDirectory, { recursive: true, force: true });
    throw error;
  }
}

export function validateGenerateExchangeCounterfactualDatasetOptions(
  options: GenerateExchangeCounterfactualDatasetOptions
): Required<Omit<
  GenerateExchangeCounterfactualDatasetOptions,
  | "createBiddingAgent"
  | "createAdjutantAgent"
  | "createPlayingAgent"
  | "onProgress"
>> & Pick<
  GenerateExchangeCounterfactualDatasetOptions,
  "createBiddingAgent" | "createAdjutantAgent" | "createPlayingAgent" | "onProgress"
> {
  if (options.outputDirectory.trim() === "") {
    throw new Error("outputDirectory is required.");
  }
  if (!Number.isInteger(options.sourceStateCount) || options.sourceStateCount <= 0) {
    throw new Error(`sourceStateCount must be a positive integer: ${options.sourceStateCount}`);
  }
  if (!Number.isInteger(options.startSeed) || options.startSeed < 0) {
    throw new Error(`startSeed must be a non-negative integer: ${options.startSeed}`);
  }

  const statesPerShard = options.statesPerShard ?? DEFAULT_STATES_PER_SHARD;
  if (!Number.isInteger(statesPerShard) || statesPerShard <= 0) {
    throw new Error(`statesPerShard must be a positive integer: ${statesPerShard}`);
  }

  const maxDealAttempts = options.maxDealAttempts ??
    options.sourceStateCount * DEFAULT_MAX_DEAL_ATTEMPTS_FACTOR;
  if (!Number.isInteger(maxDealAttempts) || maxDealAttempts < options.sourceStateCount) {
    throw new Error(
      `maxDealAttempts must be an integer >= sourceStateCount: ${maxDealAttempts}`
    );
  }
  validateSeedHeadroom(options.startSeed, maxDealAttempts);

  const maxDecisionSteps = options.maxDecisionSteps ?? DEFAULT_MAX_DECISION_STEPS;
  if (!Number.isInteger(maxDecisionSteps) || maxDecisionSteps <= 0) {
    throw new Error(`maxDecisionSteps must be a positive integer: ${maxDecisionSteps}`);
  }

  const playerIds = options.playerIds ?? DEFAULT_PLAYER_IDS;
  if (playerIds.length !== PLAYER_COUNT || new Set(playerIds).size !== playerIds.length) {
    throw new Error(`playerIds must contain ${PLAYER_COUNT} unique ids.`);
  }

  return {
    outputDirectory: options.outputDirectory,
    sourceStateCount: options.sourceStateCount,
    startSeed: normalizeSeed(options.startSeed),
    statesPerShard,
    maxDealAttempts,
    maxDecisionSteps,
    playerIds,
    sourceCommit: options.sourceCommit ?? null,
    biddingPolicy: options.biddingPolicy ??
      (options.createBiddingAgent === undefined
        ? {
            id: "rule-based-source-bidding-v1",
            description: "Default RuleBasedAgent source bidding policy"
          }
        : {
            id: "custom-source-bidding-policy-unknown-v1",
            description: "Caller supplied createBiddingAgent without explicit biddingPolicy metadata"
          }),
    adjutantPolicy: options.adjutantPolicy ??
      (options.createAdjutantAgent === undefined
        ? {
            id: "rule-based-adjutant-v1",
            description: "RuleBasedAgent adjutant selection"
          }
        : {
            id: "custom-adjutant-policy-unknown-v1",
            description: "Caller supplied createAdjutantAgent without explicit adjutantPolicy metadata"
          }),
    playingPolicy: options.playingPolicy ??
      (options.createPlayingAgent === undefined
        ? {
            id: "rule-based-playing-rollout-v1",
            description: "RuleBasedAgent fixed playing rollout policy"
          }
        : {
            id: "custom-playing-rollout-policy-unknown-v1",
            description: "Caller supplied createPlayingAgent without explicit playingPolicy metadata"
          }),
    playingPolicyDeterministic: options.playingPolicyDeterministic ?? false,
    createBiddingAgent: options.createBiddingAgent,
    createAdjutantAgent: options.createAdjutantAgent,
    createPlayingAgent: options.createPlayingAgent,
    onProgress: options.onProgress
  };
}

function validateSeedHeadroom(startSeed: number, maxDealAttempts: number): void {
  if (startSeed + maxDealAttempts - 1 > UINT32_MAX) {
    throw new Error(
      `startSeed + maxDealAttempts - 1 must be <= ${UINT32_MAX} to keep deal seeds in uint32 range.`
    );
  }
}

async function assertOutputDirectoryDoesNotExist(outputDirectory: string): Promise<void> {
  try {
    await lstat(outputDirectory);
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") {
      return;
    }
    throw error;
  }

  throw new Error(`outputDirectory already exists: ${outputDirectory}`);
}

export function enumerateExchangeDiscardCombinations(
  hand: readonly Card[]
): readonly (readonly string[])[] {
  const sortedHand = [...hand].sort(compareCardsByEncoderOrder);
  const combinations: string[][] = [];

  for (let first = 0; first < sortedHand.length - 2; first += 1) {
    for (let second = first + 1; second < sortedHand.length - 1; second += 1) {
      for (let third = second + 1; third < sortedHand.length; third += 1) {
        combinations.push([
          sortedHand[first].id,
          sortedHand[second].id,
          sortedHand[third].id
        ]);
      }
    }
  }

  return combinations;
}

export function summarizeExchangeCounterfactualSamples(
  samples: readonly ExchangeCounterfactualSample[]
): ExchangeCounterfactualDatasetSummary {
  const bySource = new Map<string, ExchangeCounterfactualSample[]>();
  for (const sample of samples) {
    const rows = bySource.get(sample.sourceStateKey) ?? [];
    rows.push(sample);
    bySource.set(sample.sourceStateKey, rows);
  }

  const diagnostics = Array.from(bySource.values()).map((rows) => createSourceDiagnostic(rows));

  const invariantFailureCount = samples.filter((sample) =>
    Object.values(sample.invariantChecks).some((value) => value === false)
  ).length;

  return createSummaryFromDiagnostics({
    diagnostics,
    uniqueDealSeedCount: new Set(samples.map((sample) => sample.dealSeed)).size,
    uniqueFixedHandCount: new Set(samples.map((sample) => sample.fixedHandId)).size,
    sampleCount: samples.length,
    invariantFailureCount,
    playingPolicyDeterministic: false
  });
}

function createSourceDiagnostic(
  rows: readonly ExchangeCounterfactualSample[]
): ExchangeCounterfactualSourceDiagnostic {
  const sortedByReward = [...rows].sort(compareCandidateOutcome);
  const best = sortedByReward[0];
  const ruleBased = rows.find((sample) => sample.isRuleBasedAction);
  if (best === undefined || ruleBased === undefined) {
    throw new Error("Source state is missing best or RuleBased sample.");
  }
  const ruleBasedRank = rows.filter((sample) =>
    isOutcomeStrictlyBetter(sample, ruleBased)
  ).length + 1;
  const rewardValues = rows.map((sample) => sample.napoleonRelativeReward);
  const marginValues = rows.map((sample) => sample.contractMargin);

  return {
    sourceStateKey: best.sourceStateKey,
    dealSeed: best.dealSeed,
    candidateCount: rows.length,
    ruleBasedCandidateIndex: ruleBased.candidateIndex,
    ruleBasedBestMatch: sameOutcome(ruleBased, best),
    ruleBasedRankByRelativeReward: ruleBasedRank,
    ruleBasedRankPercentile: rows.length === 1
      ? 1
      : (rows.length - ruleBasedRank) / (rows.length - 1),
    bestCandidateIndex: best.candidateIndex,
    bestRelativeReward: best.napoleonRelativeReward,
    ruleBasedRelativeReward: ruleBased.napoleonRelativeReward,
    ruleBasedRelativeRewardRegret: best.napoleonRelativeReward - ruleBased.napoleonRelativeReward,
    bestMargin: best.contractMargin,
    ruleBasedMargin: ruleBased.contractMargin,
    ruleBasedMarginRegret: best.contractMargin - ruleBased.contractMargin,
    marginSpread: Math.max(...marginValues) - Math.min(...marginValues),
    relativeRewardSpread: Math.max(...rewardValues) - Math.min(...rewardValues),
    bestBuriedPointCardCount: best.buriedPointCardCount,
    ruleBasedBuriedPointCardCount: ruleBased.buriedPointCardCount,
    bestBuriedSpecialCards: best.buriedSpecialCards,
    ruleBasedBuriedSpecialCards: ruleBased.buriedSpecialCards
  };
}

function createSummaryFromDiagnostics(input: {
  diagnostics: readonly ExchangeCounterfactualSourceDiagnostic[];
  uniqueDealSeedCount: number;
  uniqueFixedHandCount: number;
  sampleCount: number;
  invariantFailureCount: number;
  playingPolicyDeterministic: boolean;
}): ExchangeCounterfactualDatasetSummary {
  const diagnostics = input.diagnostics;
  return {
    sourceStateCount: diagnostics.length,
    uniqueDealSeedCount: input.uniqueDealSeedCount,
    uniqueSourceStateCount: diagnostics.length,
    uniqueFixedHandCount: input.uniqueFixedHandCount,
    sampleCount: input.sampleCount,
    rolloutCount: input.sampleCount,
    candidateCountPerState: summarizeNumbers(diagnostics.map((diagnostic) => diagnostic.candidateCount)),
    marginSpread: summarizeNumbers(diagnostics.map((diagnostic) => diagnostic.marginSpread)),
    relativeRewardSpread: summarizeNumbers(
      diagnostics.map((diagnostic) => diagnostic.relativeRewardSpread)
    ),
    ruleBasedMatchRate: diagnostics.length === 0
      ? 0
      : diagnostics.filter((diagnostic) => diagnostic.ruleBasedBestMatch).length / diagnostics.length,
    ruleBasedMarginRegret: summarizeNumbers(
      diagnostics.map((diagnostic) => diagnostic.ruleBasedMarginRegret)
    ),
    ruleBasedRelativeRewardRegret: summarizeNumbers(
      diagnostics.map((diagnostic) => diagnostic.ruleBasedRelativeRewardRegret)
    ),
    ruleBasedRankPercentile: summarizeNumbers(
      diagnostics.map((diagnostic) => diagnostic.ruleBasedRankPercentile)
    ),
    bestBuriedPointCardCount: summarizeNumbers(
      diagnostics.map((diagnostic) => diagnostic.bestBuriedPointCardCount)
    ),
    ruleBasedBuriedPointCardCount: summarizeNumbers(
      diagnostics.map((diagnostic) => diagnostic.ruleBasedBuriedPointCardCount)
    ),
    bestSpecialCardBuryRate: summarizeSpecialBuryRates(
      diagnostics.map((diagnostic) => diagnostic.bestBuriedSpecialCards)
    ),
    ruleBasedSpecialCardBuryRate: summarizeSpecialBuryRates(
      diagnostics.map((diagnostic) => diagnostic.ruleBasedBuriedSpecialCards)
    ),
    rankingStability: {
      method: "fixed-source-single-rollout-deterministic-policy",
      repeatedRolloutsPerCandidate: 1,
      deterministicPlayingPolicy: input.playingPolicyDeterministic,
      exactRepeatRankCorrelation: input.playingPolicyDeterministic ? 1 : null,
      note: input.playingPolicyDeterministic
        ? "Source state and candidate discard are fixed; no hidden reshuffle is performed. With deterministic playing policy selection, rerunning the same state/candidate gives the same ranking, so diagnostic budget is spent on unique source states."
        : "Source state and candidate discard are fixed; no hidden reshuffle is performed. Ranking repeat correlation was not measured because the configured playing policy is not declared deterministic."
    },
    invariantFailureCount: input.invariantFailureCount,
    sourceStateDiagnostics: diagnostics
  };
}

function createManifest(input: {
  options: ReturnType<typeof validateGenerateExchangeCounterfactualDatasetOptions>;
  endSeed: number;
  dealAttempts: number;
  sampleCount: number;
  summary: ExchangeCounterfactualDatasetSummary;
  shards: readonly DatasetShardManifest[];
}): ExchangeCounterfactualDatasetManifest {
  return {
    datasetSchemaVersion: EXCHANGE_COUNTERFACTUAL_DATASET_SCHEMA_VERSION,
    generatorVersion: EXCHANGE_COUNTERFACTUAL_DATASET_GENERATOR_VERSION,
    format: EXCHANGE_COUNTERFACTUAL_DATASET_FORMAT,
    sampleType: EXCHANGE_COUNTERFACTUAL_DATASET_SAMPLE_TYPE,
    sampleSchemaVersion: EXCHANGE_COUNTERFACTUAL_SAMPLE_SCHEMA_VERSION,
    teacherId: EXCHANGE_COUNTERFACTUAL_TEACHER_ID,
    discardActionSpaceId: EXCHANGE_COUNTERFACTUAL_DISCARD_ACTION_SPACE_ID,
    discardCombinationCount: EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT,
    sourceFlow: [
      "initial deal",
      "bidding",
      "contract established",
      `adjutant selection (${input.options.adjutantPolicy.id})`,
      "kitty 3-card pickup",
      "exchanging"
    ],
    modelInput: {
      observation: "existing-exchange-observation",
      featureCount: EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
      hiddenOpponentHandsIncluded: false
    },
    teacherUsesCompleteHiddenState: true,
    permutationActionsIncluded: false,
    startSeed: input.options.startSeed,
    endSeed: input.endSeed,
    sourceStateCount: input.summary.sourceStateCount,
    requestedSourceStateCount: input.options.sourceStateCount,
    sampleCount: input.sampleCount,
    rolloutCount: input.sampleCount,
    statesPerShard: input.options.statesPerShard,
    dealAttemptCount: input.dealAttempts,
    sourceCommit: input.options.sourceCommit,
    playerCount: PLAYER_COUNT,
    cardCount: CARD_COUNT,
    cardIds: CARD_IDS,
    cardIdsSha256: calculateCardIdsSha256(),
    biddingPolicy: input.options.biddingPolicy,
    adjutantPolicy: input.options.adjutantPolicy,
    playingPolicy: input.options.playingPolicy,
    reward: {
      raw: "napoleon contract success ? 2 * target : 0",
      relative: "napoleon contract success ? 7 * target / 4 : -3 * target / 4"
    },
    summary: input.summary,
    shardCount: input.shards.length,
    shards: input.shards
  };
}

async function tryCreateSourceExchangeState(input: ReturnType<
  typeof validateGenerateExchangeCounterfactualDatasetOptions
> & {
  dealSeed: number;
  sourceIndex: number;
}): Promise<SourceExchangeState | null> {
  let state = createInitialGame({
    playerIds: input.playerIds,
    rng: createSeededRandom(deriveSeed(input.dealSeed, "game"))
  });
  const biddingAgents = createAgents(input, input.dealSeed, "bidding");
  const publicActionHistory: PublicActionRecord[] = [];
  let decisionStep = 0;

  while (state.phase === "bidding" && !state.isGameOver) {
    if (decisionStep >= input.maxDecisionSteps) {
      throw new Error(`Source bidding exceeded ${input.maxDecisionSteps} decision steps.`);
    }
    const playerId = state.currentPlayerId;
    const agent = requiredAgent(biddingAgents, playerId);
    const legalActions = getLegalActions(state, playerId);
    const observation = createObservation({
      state,
      playerId,
      legalActions,
      publicActionHistory
    });
    const action = await selectAgentAction(agent, observation, {
      actualState: captureActualCardState(state),
      playerIds: input.playerIds
    });
    assertLegalAction(action, legalActions, state.phase);
    decisionStep += 1;
    publicActionHistory.push({
      step: decisionStep,
      playerId,
      phase: "bidding",
      action: action as PublicActionRecord["action"]
    });
    state = applyAction(clearLatestEvent(state), action);
  }

  if (state.phase !== "choosing-adjutant" || state.contract === null || state.isGameOver) {
    return null;
  }

  const napoleonPlayerId = state.contract.napoleonPlayerId;
  const adjutantAgent = createAgent(input, input.dealSeed, "adjutant", napoleonPlayerId);
  const adjutantLegalActions = getLegalActions(state, napoleonPlayerId);
  const adjutantObservation = createObservation({
    state,
    playerId: napoleonPlayerId,
    legalActions: adjutantLegalActions,
    publicActionHistory
  });
  const adjutantAction = await selectAgentAction(adjutantAgent, adjutantObservation, {
    actualState: captureActualCardState(state),
    playerIds: input.playerIds
  });
  assertLegalAction(adjutantAction, adjutantLegalActions, state.phase);
  state = applyAction(clearLatestEvent(state), adjutantAction);

  if (state.phase !== "exchanging" || state.contract === null || state.adjutant === null) {
    throw new Error(`Expected exchange phase after adjutant selection for seed ${input.dealSeed}.`);
  }

  const exchangeLegalActions = enumerateExchangeLegalActions(state, napoleonPlayerId);
  const exchangeObservation = createObservation({
    state,
    playerId: napoleonPlayerId,
    legalActions: exchangeLegalActions,
    publicActionHistory
  });
  const absolutePlayerIds = input.playerIds;
  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, napoleonPlayerId);
  const biddingHistory = encodeBiddingHistoryFromPublicActions(publicActionHistory, relativePlayerIds);
  const encodedExchangeObservation = encodeExchangeObservation(
    exchangeObservation,
    absolutePlayerIds,
    biddingHistory
  );
  const { modelInput, legalDiscardCardMask } = createExchangeModelInput(encodedExchangeObservation);
  const self = getPlayerHand(state, napoleonPlayerId);
  const pickupHandCards = [...self].sort(compareCardsByEncoderOrder);
  const pickupHandCardIds = pickupHandCards.map((card) => card.id);
  const combinations = enumerateExchangeDiscardCombinations(pickupHandCards);
  if (combinations.length !== EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT) {
    throw new Error(`Expected 286 exchange candidates, got ${combinations.length}.`);
  }

  const ruleBasedAction = await new RuleBasedAgent(
    createSeededRandom(deriveSeed(input.dealSeed, "rule-based-discard-diagnostic"))
  ).selectAction(exchangeObservation);
  if (ruleBasedAction.type !== "discard-cards") {
    throw new Error(`RuleBased exchange selected ${ruleBasedAction.type}.`);
  }
  const ruleBasedDiscardCardIds = canonicalCardIds(ruleBasedAction.cardIds);
  const ruleBasedCandidateIndex = combinations.findIndex((candidate) =>
    sameStringArray(candidate, ruleBasedDiscardCardIds)
  );
  if (ruleBasedCandidateIndex === -1) {
    throw new Error("RuleBased discard action was not found among enumerated candidates.");
  }

  const hiddenDealChecksum = createHiddenDealChecksum(state);
  const sourceStateKey = sha256Utf8(JSON.stringify({
    dealSeed: input.dealSeed,
    contract: state.contract,
    adjutant: state.adjutant,
    pickupHandCardIds,
    publicActionHistory,
    hiddenDealChecksum
  })).slice(0, 24);
  const fixedHandId = sha256Utf8(JSON.stringify({
    napoleonSeatIndex: input.playerIds.indexOf(napoleonPlayerId),
    pickupHandCardIds
  })).slice(0, 24);

  return {
    sourceIndex: input.sourceIndex,
    dealSeed: input.dealSeed,
    state,
    publicActionHistory,
    hiddenDealChecksum,
    sourceStateKey,
    fixedHandId,
    modelInput: Array.from(modelInput),
    legalDiscardCardMask: [...legalDiscardCardMask],
    pickupHandCards,
    pickupHandCardIds,
    ruleBasedDiscardCardIds,
    ruleBasedCandidateIndex
  };
}

async function createSourceSamples(
  source: SourceExchangeState,
  options: ReturnType<typeof validateGenerateExchangeCounterfactualDatasetOptions>
): Promise<readonly ExchangeCounterfactualSample[]> {
  const combinations = enumerateExchangeDiscardCombinations(source.pickupHandCards);
  const candidateKeys = new Set(combinations.map((combo) => combo.join("|")));
  const invariantBase = {
    sourcePhaseExchanging: source.state.phase === "exchanging",
    napoleonHandCount13: source.pickupHandCardIds.length === 13,
    candidateCount286: combinations.length === EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT,
    modelInputFeatureCount2671: source.modelInput.length === EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
    legalDiscardMaskMatchesPickupHand: sameNumberArray(
      source.legalDiscardCardMask,
      createCardMask(source.pickupHandCardIds)
    ),
    sameSourceHiddenDeal: true
  };

  if (candidateKeys.size !== EXCHANGE_COUNTERFACTUAL_COMBINATION_COUNT) {
    throw new Error(`Duplicate exchange candidates for source ${source.sourceStateKey}.`);
  }

  const samples: ExchangeCounterfactualSample[] = [];
  for (const [candidateIndex, candidateDiscardCardIds] of combinations.entries()) {
    const result = await rolloutDiscardCandidate(source, options, candidateDiscardCardIds);
    const buriedCards = candidateDiscardCardIds.map((cardId) => {
      const card = source.pickupHandCards.find((candidate) => candidate.id === cardId);
      if (card === undefined) {
        throw new Error(`Candidate card ${cardId} is missing from pickup hand.`);
      }
      return card;
    });
    const contract = requiredContract(source.state);
    const candidateDiscardMask = createCardMask(candidateDiscardCardIds);
    const candidateKey = sha256Utf8(JSON.stringify({
      sourceStateKey: source.sourceStateKey,
      candidateDiscardCardIds
    })).slice(0, 24);
    const invariantChecks: ExchangeCounterfactualInvariantChecks = {
      ...invariantBase,
      candidateDiscardCount3: candidateDiscardCardIds.length === COMBINATION_DISCARD_COUNT,
      candidateDiscardUnique: new Set(candidateDiscardCardIds).size === COMBINATION_DISCARD_COUNT,
      candidateDiscardFromPickupHand: candidateDiscardCardIds.every((cardId) =>
        source.pickupHandCardIds.includes(cardId)
      ),
      unorderedCombinationCanonical: sameStringArray(
        candidateDiscardCardIds,
        canonicalCardIds(candidateDiscardCardIds)
      ),
      standardGameResult: result.resultType === "standard"
    };
    const buriedPointCardCount = buriedCards.filter(isPointCard).length;
    const contractSuccess = result.napoleonTeamPointCards >= contract.targetPointCards;

    samples.push({
      sampleType: EXCHANGE_COUNTERFACTUAL_DATASET_SAMPLE_TYPE,
      schemaVersion: EXCHANGE_COUNTERFACTUAL_SAMPLE_SCHEMA_VERSION,
      sourceStateKey: source.sourceStateKey,
      fixedHandId: source.fixedHandId,
      dealSeed: source.dealSeed,
      sourceIndex: source.sourceIndex,
      candidateIndex,
      candidateKey,
      napoleonPlayerId: contract.napoleonPlayerId,
      napoleonSeatIndex: options.playerIds.indexOf(contract.napoleonPlayerId),
      contractTargetPointCards: contract.targetPointCards,
      contractSuit: contract.trumpSuit,
      calledAdjutantCardId: requiredAdjutant(source.state).calledCardId,
      pickupHandCardIds: source.pickupHandCardIds,
      modelInput: source.modelInput,
      legalDiscardCardMask: source.legalDiscardCardMask,
      candidateDiscardCardIds,
      candidateDiscardMask,
      buriedPointCardCount,
      buriedTrumpCount: buriedCards.filter((card) =>
        card.type === "standard" && card.suit === contract.trumpSuit
      ).length,
      buriedSpecialCards: createSpecialBuriedFlags(
        candidateDiscardCardIds,
        contract.trumpSuit,
        requiredAdjutant(source.state).calledCardId
      ),
      contractMargin: result.napoleonTeamPointCards - contract.targetPointCards,
      contractSuccess,
      napoleonSidePointCards: result.napoleonTeamPointCards,
      napoleonRawReward: contractSuccess ? 2 * contract.targetPointCards : 0,
      napoleonRelativeReward: contractSuccess
        ? (7 * contract.targetPointCards) / 4
        : (-3 * contract.targetPointCards) / 4,
      ruleBasedDiscardCardIds: source.ruleBasedDiscardCardIds,
      isRuleBasedAction: candidateIndex === source.ruleBasedCandidateIndex,
      ruleBasedCandidateIndex: source.ruleBasedCandidateIndex,
      hiddenDealChecksum: source.hiddenDealChecksum,
      biddingHistoryHash: sha256Utf8(JSON.stringify(source.publicActionHistory)),
      biddingHistoryActionCount: source.publicActionHistory.length,
      invariantChecks
    });
  }

  return samples;
}

async function rolloutDiscardCandidate(
  source: SourceExchangeState,
  options: ReturnType<typeof validateGenerateExchangeCounterfactualDatasetOptions>,
  candidateDiscardCardIds: readonly string[]
): Promise<StandardGameResult> {
  let state = applyAction(clearLatestEvent(source.state), {
    type: "discard-cards",
    playerId: requiredContract(source.state).napoleonPlayerId,
    cardIds: candidateDiscardCardIds
  });
  const playingAgents = createAgents(options, source.dealSeed, `playout:${candidateDiscardCardIds.join(",")}`);
  let decisionStep = source.publicActionHistory.length;

  while (!state.isGameOver) {
    if (state.isTrickComplete) {
      state = advanceToNextTrick(clearLatestEvent(state));
      continue;
    }
    if (decisionStep >= options.maxDecisionSteps) {
      throw new Error(`Playing rollout exceeded ${options.maxDecisionSteps} decision steps.`);
    }

    const playerId = state.currentPlayerId;
    const agent = requiredAgent(playingAgents, playerId);
    const legalActions = getLegalActions(state, playerId);
    const observation = createObservation({
      state,
      playerId,
      legalActions,
      publicActionHistory: source.publicActionHistory
    });
    const action = await selectAgentAction(agent, observation, {
      actualState: captureActualCardState(state),
      playerIds: options.playerIds
    });
    assertLegalAction(action, legalActions, state.phase);
    decisionStep += 1;
    state = applyAction(clearLatestEvent(state), action);
  }

  if (state.result?.resultType !== "standard") {
    throw new Error(`Exchange rollout ended with non-standard result: ${state.result?.resultType ?? "null"}`);
  }

  return state.result;
}

function createAgents(
  options: ReturnType<typeof validateGenerateExchangeCounterfactualDatasetOptions>,
  seed: number,
  phaseLabel: "bidding" | "adjutant" | string
): ReadonlyMap<PlayerId, Agent> {
  return new Map(options.playerIds.map((playerId, playerIndex) => [
    playerId,
    createAgent(options, seed, phaseLabel, playerId, playerIndex)
  ]));
}

function createAgent(
  options: ReturnType<typeof validateGenerateExchangeCounterfactualDatasetOptions>,
  seed: number,
  phaseLabel: "bidding" | "adjutant" | string,
  playerId: PlayerId,
  playerIndex = options.playerIds.indexOf(playerId)
): Agent {
  const context: ExchangeCounterfactualAgentContext = {
    playerId,
    playerIndex,
    rng: createSeededRandom(deriveSeed(seed, `agent:${phaseLabel}:${playerId}`)),
    seed
  };

  if (phaseLabel === "bidding") {
    return options.createBiddingAgent?.(context) ?? new RuleBasedAgent(context.rng);
  }
  if (phaseLabel === "adjutant") {
    return options.createAdjutantAgent?.(context) ?? new RuleBasedAgent(context.rng);
  }
  return options.createPlayingAgent?.(context) ?? new RuleBasedAgent(context.rng);
}

function createObservation(input: {
  state: GameState;
  playerId: PlayerId;
  legalActions: readonly GameAction[];
  publicActionHistory: readonly PublicActionRecord[];
}): PlayerObservation {
  return {
    playerId: input.playerId,
    view: {
      ...createPlayerView(input.state, input.playerId),
      legalActions: input.legalActions
    },
    legalActions: input.legalActions,
    publicActionHistory: input.publicActionHistory
  };
}

function enumerateExchangeLegalActions(
  state: GameState,
  playerId: PlayerId
): readonly DiscardCardsAction[] {
  const hand = getPlayerHand(state, playerId);
  return enumerateExchangeDiscardCombinations(hand).map((cardIds): DiscardCardsAction => ({
    type: "discard-cards",
    playerId,
    cardIds
  }));
}

async function selectAgentAction(
  agent: Agent,
  observation: PlayerObservation,
  context: {
    actualState: ActualCardState;
    playerIds: readonly PlayerId[];
  }
): Promise<GameAction> {
  if (isCompleteInfoActionAgent(agent)) {
    return agent.selectActionWithContext(observation, context);
  }

  return agent.selectAction(observation);
}

function isCompleteInfoActionAgent(agent: Agent): agent is Agent & CompleteInfoActionAgent {
  return typeof (agent as Partial<CompleteInfoActionAgent>).selectActionWithContext === "function";
}

function captureActualCardState(state: GameState): ActualCardState {
  return {
    hands: Object.fromEntries(
      state.players.map((player) => [player.id, player.hand.map((card) => card.id)])
    ),
    unusedCardIds: state.unusedCards.map((card) => card.id),
    excludedCardIds: state.excludedCards.map((card) => card.id),
    awardedPointCardIds: Object.fromEntries(
      state.players.map((player) => [
        player.id,
        state.awardedPointCards
          .filter((award) => award.playerId === player.id)
          .flatMap((award) => award.cards.map((card) => card.id))
      ])
    ),
    currentTrickCardIds: state.currentTrick.map((playedCard) => playedCard.card.id),
    completedTrickCardIds: state.completedTricks.flatMap((trick) =>
      trick.cards.map((playedCard) => playedCard.card.id)
    )
  };
}

function createHiddenDealChecksum(state: GameState): string {
  return sha256Utf8(JSON.stringify({
    phase: state.phase,
    currentPlayerId: state.currentPlayerId,
    contract: state.contract,
    adjutant: state.adjutant,
    bidding: state.bidding,
    players: state.players.map((player) => ({
      id: player.id,
      hand: canonicalCardIds(player.hand.map((card) => card.id))
    })),
    unusedCards: canonicalCardIds(state.unusedCards.map((card) => card.id)),
    awardedPointCards: state.awardedPointCards.map((award) => ({
      playerId: award.playerId,
      cards: canonicalCardIds(award.cards.map((card) => card.id))
    })),
    excludedCards: canonicalCardIds(state.excludedCards.map((card) => card.id))
  }));
}

function createCardMask(cardIds: readonly string[]): readonly number[] {
  const mask = Array(CARD_IDS.length).fill(0);
  for (const cardId of cardIds) {
    const index = CARD_IDS.indexOf(cardId);
    if (index === -1) {
      throw new Error(`Unknown card id: ${cardId}`);
    }
    mask[index] = 1;
  }
  return mask;
}

function createSpecialBuriedFlags(
  cardIds: readonly string[],
  trumpSuit: Suit,
  calledAdjutantCardId: string
): ExchangeCounterfactualSpecialBuriedFlags {
  const idSet = new Set(cardIds);
  return {
    joker: idSet.has(jokerCardId),
    oruma: idSet.has(orumaCardId),
    yoromeki: idSet.has(yoromekiCardId),
    seiJack: idSet.has(getSeiJackCardId(trumpSuit)),
    uraJack: idSet.has(getUraJackCardId(trumpSuit)),
    calledAdjutant: idSet.has(calledAdjutantCardId)
  };
}

function getPlayerHand(state: GameState, playerId: PlayerId): readonly Card[] {
  const player = state.players.find((candidate) => candidate.id === playerId);
  if (player === undefined) {
    throw new Error(`Unknown player id: ${playerId}`);
  }
  return player.hand;
}

function requiredContract(state: GameState): NonNullable<GameState["contract"]> {
  if (state.contract === null) {
    throw new Error("Expected contract.");
  }
  return state.contract;
}

function requiredAdjutant(state: GameState): NonNullable<GameState["adjutant"]> {
  if (state.adjutant === null) {
    throw new Error("Expected adjutant.");
  }
  return state.adjutant;
}

function requiredAgent(agents: ReadonlyMap<PlayerId, Agent>, playerId: PlayerId): Agent {
  const agent = agents.get(playerId);
  if (agent === undefined) {
    throw new Error(`No agent for ${playerId}.`);
  }
  return agent;
}

function assertLegalAction(
  action: GameAction,
  legalActions: readonly GameAction[],
  phase: GameState["phase"]
): void {
  if (!legalActions.some((legalAction) => actionsEqual(action, legalAction))) {
    throw new Error(
      [
        "Agent selected an illegal action.",
        `phase=${phase}`,
        `action=${JSON.stringify(action)}`,
        `legalActions=${JSON.stringify(legalActions)}`
      ].join(" ")
    );
  }
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
    case "choose-adjutant":
      return right.type === "choose-adjutant" && left.cardId === right.cardId;
    case "discard-cards":
      return right.type === "discard-cards" && sameStringArray(
        canonicalCardIds(left.cardIds),
        canonicalCardIds(right.cardIds)
      );
    case "play-card":
      return right.type === "play-card" && left.cardId === right.cardId;
  }
}

function compareCandidateOutcome(
  left: ExchangeCounterfactualSample,
  right: ExchangeCounterfactualSample
): number {
  return right.napoleonRelativeReward - left.napoleonRelativeReward ||
    right.contractMargin - left.contractMargin ||
    right.napoleonSidePointCards - left.napoleonSidePointCards ||
    left.candidateIndex - right.candidateIndex;
}

function isOutcomeStrictlyBetter(
  left: ExchangeCounterfactualSample,
  right: ExchangeCounterfactualSample
): boolean {
  return left.napoleonRelativeReward > right.napoleonRelativeReward ||
    (left.napoleonRelativeReward === right.napoleonRelativeReward &&
      left.contractMargin > right.contractMargin) ||
    (left.napoleonRelativeReward === right.napoleonRelativeReward &&
      left.contractMargin === right.contractMargin &&
      left.napoleonSidePointCards > right.napoleonSidePointCards);
}

function sameOutcome(
  left: ExchangeCounterfactualSample,
  right: ExchangeCounterfactualSample
): boolean {
  return left.napoleonRelativeReward === right.napoleonRelativeReward &&
    left.contractMargin === right.contractMargin &&
    left.napoleonSidePointCards === right.napoleonSidePointCards;
}

function compareCardsByEncoderOrder(left: Card, right: Card): number {
  return CARD_IDS.indexOf(left.id) - CARD_IDS.indexOf(right.id);
}

function canonicalCardIds(cardIds: readonly string[]): readonly string[] {
  return [...cardIds].sort((left, right) => CARD_IDS.indexOf(left) - CARD_IDS.indexOf(right));
}

function sameStringArray(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function sameNumberArray(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function summarizeNumbers(values: readonly number[]): NumericSummary {
  if (values.length === 0) {
    return { count: 0, min: 0, max: 0, mean: 0, median: 0 };
  }
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
  return {
    count: values.length,
    min: sorted[0],
    max: sorted[sorted.length - 1],
    mean: values.reduce((total, value) => total + value, 0) / values.length,
    median
  };
}

function summarizeSpecialBuryRates(
  flags: readonly ExchangeCounterfactualSpecialBuriedFlags[]
): ExchangeCounterfactualSpecialBuryRates {
  if (flags.length === 0) {
    return {
      joker: 0,
      oruma: 0,
      yoromeki: 0,
      seiJack: 0,
      uraJack: 0,
      calledAdjutant: 0,
      any: 0
    };
  }
  const count = (field: keyof ExchangeCounterfactualSpecialBuriedFlags) =>
    flags.filter((flag) => flag[field]).length / flags.length;
  return {
    joker: count("joker"),
    oruma: count("oruma"),
    yoromeki: count("yoromeki"),
    seiJack: count("seiJack"),
    uraJack: count("uraJack"),
    calledAdjutant: count("calledAdjutant"),
    any: flags.filter((flag) => Object.values(flag).some(Boolean)).length / flags.length
  };
}

function isNodeError(error: unknown): error is NodeJS.ErrnoException {
  return error instanceof Error && "code" in error;
}

async function writeJsonlShard(input: {
  directory: string;
  rows: readonly ExchangeCounterfactualSample[];
  shardIndex: number;
  startSeed: number;
  endSeed: number;
  gameCount: number;
}): Promise<DatasetShardManifest> {
  await mkdir(input.directory, { recursive: true });
  const file = `shard-${String(input.shardIndex).padStart(5, "0")}.jsonl`;
  const body = input.rows.map((row) => `${JSON.stringify(row)}\n`).join("");
  await writeFile(join(input.directory, file), body, "utf8");
  return {
    file,
    startSeed: input.startSeed,
    endSeed: input.endSeed,
    gameCount: input.gameCount,
    sampleCount: input.rows.length,
    byteLength: Buffer.byteLength(body, "utf8"),
    sha256: createHash("sha256").update(body, "utf8").digest("hex")
  };
}
