import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import {
  ConservativeBiddingAgent,
  RuleBasedAgent,
  createSeededRandom,
  deriveSeed,
  evaluateHandForTrump
} from "@napoleon/ai";
import type { Agent, PlayerObservation } from "@napoleon/ai";
import {
  advanceToNextTrick,
  applyAction,
  clearLatestEvent,
  createContractEstablishedState,
  createDeck,
  createInitialGame,
  createPlayerView,
  getLegalActions
} from "@napoleon/game-core";
import type { Card, GameAction, GameResult, PlayerId, Suit } from "@napoleon/game-core";
import {
  BIDDING_ACTION_COUNT,
  BIDDING_ENCODER_SCHEMA_VERSION,
  BIDDING_HISTORY_SUIT_ORDER,
  BIDDING_MODEL_INPUT_FEATURE_COUNT,
  BIDDING_MODEL_INPUT_SCHEMA_VERSION,
  CARD_COUNT,
  CARD_IDS,
  createBiddingModelInput,
  decodeBiddingAction,
  encodeBiddingObservation
} from "@napoleon/ai-observation";
import { DATASET_FORMAT } from "./schema.js";
import type { DatasetShardManifest } from "./types.js";
import { calculateCardIdsSha256, serializeManifest } from "./serialization.js";
import {
  BIDDING_Q_COUNTERFACTUAL_ACTION_MAPPING_ID,
  BIDDING_Q_COUNTERFACTUAL_OPPONENT_MIX_RULE_VERSION,
  decodeBiddingQActionIndex
} from "./generateBiddingQCounterfactualDataset.js";

export const FIXED_HAND_BIDDING_MARGIN_DATASET_SAMPLE_TYPE =
  "fixed-hand-bidding-margin-sample" as const;
export const FIXED_HAND_BIDDING_MARGIN_SAMPLE_SCHEMA_VERSION = 1 as const;
export const FIXED_HAND_BIDDING_MARGIN_DATASET_SCHEMA_VERSION = 1 as const;
export const FIXED_HAND_BIDDING_MARGIN_DATASET_GENERATOR_VERSION = 1 as const;
export const FIXED_HAND_BIDDING_MARGIN_TEACHER_ID =
  "fixed-hand-action-hidden-deal-empirical-margin-mean-v1" as const;
export const NAPOLEON_FIXED_MARGIN_DATASET_SAMPLE_TYPE =
  "napoleon-fixed-contract-margin-sample" as const;
export const NAPOLEON_FIXED_MARGIN_SAMPLE_SCHEMA_VERSION = 1 as const;
export const NAPOLEON_FIXED_MARGIN_DATASET_SCHEMA_VERSION = 1 as const;
export const NAPOLEON_FIXED_MARGIN_DATASET_GENERATOR_VERSION = 1 as const;
export const NAPOLEON_FIXED_MARGIN_TEACHER_ID =
  "napoleon-fixed-contract-hidden-deal-empirical-margin-mean-v1" as const;

const PLAYER_COUNT = 5;
const PLAYER_IDS = ["player-0", "player-1", "player-2", "player-3", "player-4"] as const;
const TARGETS = [13, 14, 15, 16, 17, 18, 19] as const;
const SUITS = BIDDING_HISTORY_SUIT_ORDER;
const CARD_BY_ID = new Map(createDeck().map((card) => [card.id, card]));

export interface FixedHandBiddingActionSpec {
  actionIndex: number;
  targetPointCards: number;
  suit: Suit;
  label?: string;
  sourceNnMu?: number | null;
  sourceNnSigma?: number | null;
  sourceNnPWin?: number | null;
}

export interface FixedHandSpec {
  fixedHandId: string;
  handIds: readonly string[];
  candidateSeatIndex: number;
  sourceStateKey?: string | null;
  sourceSeed?: number | null;
  sourceBiddingStep?: number | null;
  strongestSuit: Suit;
  strongestSuitScore: number;
  reason?: string | null;
  actions: readonly FixedHandBiddingActionSpec[];
}

export interface GenerateFixedHandBiddingMarginDatasetOptions {
  outputDirectory: string;
  pairCount: number;
  repeats: number;
  randomSeed: number;
  candidateSeatIndex?: number;
  actionCountPerHand?: number;
  gamesPerShard?: number;
  reservedHands?: readonly FixedHandSpec[];
  reserveHandsForFinal?: boolean;
  sourceCommit?: string | null;
  onProgress?: (progress: {
    totalPairs: number;
    completedPairs: number;
    totalRollouts: number;
    completedRollouts: number;
    sampleCount: number;
  }) => void;
}

export type GenerateNapoleonFixedMarginDatasetOptions =
  GenerateFixedHandBiddingMarginDatasetOptions;

export interface FixedHandBiddingMarginSample {
  sampleType: typeof FIXED_HAND_BIDDING_MARGIN_DATASET_SAMPLE_TYPE;
  schemaVersion: typeof FIXED_HAND_BIDDING_MARGIN_SAMPLE_SCHEMA_VERSION;
  fixedHandId: string;
  handIds: readonly string[];
  candidateSeatIndex: number;
  candidatePlayerId: PlayerId;
  sourceStateKey: string | null;
  sourceSeed: number | null;
  sourceBiddingStep: number | null;
  strongestSuit: Suit;
  strongestSuitScore: number;
  forcedActionIndex: number;
  forcedTargetPointCards: number;
  forcedSuit: Suit;
  forcedActionLabel: string;
  modelInput: readonly number[];
  legalBidMask: readonly number[];
  rolloutCount: number;
  empiricalMarginMean: number;
  empiricalMarginStd: number;
  empiricalWinRate: number;
  empiricalMarginMin: number;
  empiricalMarginMax: number;
  marginHistogram: Record<string, number>;
  resultTypeCounts: Record<string, number>;
  finalRoleCounts: Record<string, number>;
  sourceNnMu: number | null;
  sourceNnSigma: number | null;
  sourceNnPWin: number | null;
  splitHint: "final-diagnostic" | null;
}

export interface NapoleonFixedMarginRawRollout {
  sampleType: "napoleon-fixed-contract-margin-raw-rollout";
  schemaVersion: 1;
  fixedHandId: string;
  handIds: readonly string[];
  candidateSeatIndex: number;
  candidatePlayerId: PlayerId;
  forcedActionIndex: number;
  forcedTargetPointCards: number;
  forcedSuit: Suit;
  repeatIndex: number;
  hiddenDealSeed: number;
  finalAdjutant: {
    calledCardId: string;
    playerId: PlayerId | null;
  };
  napoleonSidePointCards: number;
  coalitionSidePointCards: number;
  contractMargin: number;
  contractSuccess: boolean;
  candidateRawReward: number;
  candidateRelativeReward: number;
  resultType: GameResult["resultType"];
  finalRole: string;
  invariantChecks: {
    candidateRoleNapoleon: true;
    contractOwnerCandidate: true;
    targetMatches: true;
    suitMatches: true;
    downstreamBiddingActionCount: 0;
    candidateHandFixedAtStart: true;
    deckConservation: true;
  };
}

export interface NapoleonFixedMarginSample
  extends Omit<
    FixedHandBiddingMarginSample,
    "sampleType" | "schemaVersion" | "finalRoleCounts"
  > {
  sampleType: typeof NAPOLEON_FIXED_MARGIN_DATASET_SAMPLE_TYPE;
  schemaVersion: typeof NAPOLEON_FIXED_MARGIN_SAMPLE_SCHEMA_VERSION;
  rawRolloutShard: string;
  finalRoleCounts: Record<"napoleon" | "napoleon-adjutant", number>;
  invariantFailures: Record<string, number>;
}

export interface FixedHandBiddingMarginDatasetManifest {
  datasetSchemaVersion: typeof FIXED_HAND_BIDDING_MARGIN_DATASET_SCHEMA_VERSION;
  generatorVersion: typeof FIXED_HAND_BIDDING_MARGIN_DATASET_GENERATOR_VERSION;
  format: typeof DATASET_FORMAT;
  sampleType: typeof FIXED_HAND_BIDDING_MARGIN_DATASET_SAMPLE_TYPE;
  sampleSchemaVersion: typeof FIXED_HAND_BIDDING_MARGIN_SAMPLE_SCHEMA_VERSION;
  teacher: {
    id: typeof FIXED_HAND_BIDDING_MARGIN_TEACHER_ID;
    primaryLabel: "empiricalMarginMean";
    stdLabel: "empiricalMarginStd";
    winRateLabel: "empiricalWinRate";
    repeatsPerPair: number;
  };
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
  fixedCondition: {
    fixed: readonly string[];
    varied: readonly string[];
    note: string;
  };
  opponentMix: {
    mixingRuleVersion: typeof BIDDING_Q_COUNTERFACTUAL_OPPONENT_MIX_RULE_VERSION;
    ruleBasedWeight: 0.5;
    conservativeWeight: 0.5;
  };
  pairCount: number;
  fixedHandCount: number;
  rolloutCount: number;
  randomSeed: number;
  sourceCommit: string | null;
  playerCount: 5;
  cardCount: typeof CARD_COUNT;
  cardIds: readonly string[];
  cardIdsSha256: string;
  summary: FixedHandBiddingMarginDatasetSummary;
  shardCount: number;
  shards: readonly DatasetShardManifest[];
}

export interface NapoleonFixedMarginDatasetManifest
  extends Omit<
    FixedHandBiddingMarginDatasetManifest,
    | "datasetSchemaVersion"
    | "generatorVersion"
    | "sampleType"
    | "sampleSchemaVersion"
    | "teacher"
    | "fixedCondition"
    | "opponentMix"
    | "summary"
    | "shards"
  > {
  datasetSchemaVersion: typeof NAPOLEON_FIXED_MARGIN_DATASET_SCHEMA_VERSION;
  generatorVersion: typeof NAPOLEON_FIXED_MARGIN_DATASET_GENERATOR_VERSION;
  sampleType: typeof NAPOLEON_FIXED_MARGIN_DATASET_SAMPLE_TYPE;
  sampleSchemaVersion: typeof NAPOLEON_FIXED_MARGIN_SAMPLE_SCHEMA_VERSION;
  teacher: {
    id: typeof NAPOLEON_FIXED_MARGIN_TEACHER_ID;
    primaryLabel: "empiricalMarginMean";
    stdLabel: "empiricalMarginStd";
    winRateLabel: "empiricalWinRate";
    repeatsPerPair: number;
  };
  fixedCondition: {
    fixed: readonly string[];
    varied: readonly string[];
    note: string;
  };
  opponentMix: {
    mixingRuleVersion: typeof BIDDING_Q_COUNTERFACTUAL_OPPONENT_MIX_RULE_VERSION;
    ruleBasedWeight: number;
    conservativeWeight: number;
  };
  summary: FixedHandBiddingMarginDatasetSummary;
  shards: readonly DatasetShardManifest[];
  rawRolloutShards: readonly DatasetShardManifest[];
}

export interface FixedHandBiddingMarginDatasetSummary {
  pairCount: number;
  fixedHandCount: number;
  rolloutCount: number;
  repeatCountMin: number;
  repeatCountMax: number;
  empiricalMarginMean: NumericSummary;
  empiricalMarginStd: NumericSummary;
  empiricalWinRate: NumericSummary;
  suitCounts: Record<Suit, number>;
  targetCounts: Record<string, number>;
  splitHintCounts: Record<string, number>;
}

export interface NumericSummary {
  count: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
}

interface RolloutRow {
  fixedHandId: string;
  contractMargin: number;
  contractSuccess: boolean;
  resultType: GameResult["resultType"];
  finalRole: string;
  modelInput: readonly number[];
  legalBidMask: readonly number[];
}

interface NapoleonFixedRolloutRow extends RolloutRow {
  raw: NapoleonFixedMarginRawRollout;
}

type ForcedActionWithMetadata = GameAction & {
  __forced?: true;
  __modelInput?: readonly number[];
  __legalBidMask?: readonly number[];
};
type PublicBiddingAction = Extract<GameAction, { type: "bid" | "pass" }>;

export async function generateFixedHandBiddingMarginDataset(
  options: GenerateFixedHandBiddingMarginDatasetOptions
): Promise<{
  outputDirectory: string;
  manifest: FixedHandBiddingMarginDatasetManifest;
  samples: readonly FixedHandBiddingMarginSample[];
}> {
  validateGenerateFixedHandBiddingMarginDatasetOptions(options);
  const outputDirectory = resolve(options.outputDirectory);
  const tempDirectory = await mkdtemp(join(dirname(outputDirectory), `.${basename(outputDirectory)}.tmp-`));
  try {
    const specs = createFixedHandActionPlan(options);
    const samples: FixedHandBiddingMarginSample[] = [];
    let completedPairs = 0;
    let completedRollouts = 0;
    for (const spec of specs) {
      for (const action of spec.actions) {
        const rows: RolloutRow[] = [];
        for (let repeatIndex = 0; repeatIndex < options.repeats; repeatIndex += 1) {
          const reshuffleSeed = stableUint32([
            options.randomSeed,
            spec.fixedHandId,
            action.actionIndex,
            repeatIndex
          ].join(":"));
          rows.push(await runFixedHandBidRollout({ spec, action, repeatIndex, reshuffleSeed }));
          completedRollouts += 1;
        }
        samples.push(aggregateFixedHandBidRollouts({
          spec,
          action,
          rows,
          splitHint: options.reserveHandsForFinal && spec.reason?.includes("issue409")
            ? "final-diagnostic"
            : null
        }));
        completedPairs += 1;
        options.onProgress?.({
          totalPairs: specs.reduce((sum, item) => sum + item.actions.length, 0),
          completedPairs,
          totalRollouts: specs.reduce((sum, item) => sum + item.actions.length, 0) * options.repeats,
          completedRollouts,
          sampleCount: samples.length
        });
      }
    }
    const shards = await writeSampleShards(tempDirectory, samples, options.gamesPerShard ?? 1000);
    const summary = summarizeFixedHandBiddingMarginSamples(samples);
    const manifest = createFixedHandBiddingMarginManifest({
      options,
      samples,
      shards,
      summary
    });
    await writeFile(join(tempDirectory, "manifest.json"), serializeManifest(manifest), "utf8");
    await writeFile(join(tempDirectory, "summary.json"), serializeManifest(summary), "utf8");
    await rename(tempDirectory, outputDirectory);
    return { outputDirectory, manifest, samples };
  } catch (error) {
    await rm(tempDirectory, { recursive: true, force: true }).catch(() => undefined);
    throw error;
  }
}

export async function generateNapoleonFixedMarginDataset(
  options: GenerateNapoleonFixedMarginDatasetOptions
): Promise<{
  outputDirectory: string;
  manifest: NapoleonFixedMarginDatasetManifest;
  samples: readonly NapoleonFixedMarginSample[];
  rawRollouts: readonly NapoleonFixedMarginRawRollout[];
}> {
  validateGenerateFixedHandBiddingMarginDatasetOptions(options);
  const outputDirectory = resolve(options.outputDirectory);
  const tempDirectory = await mkdtemp(join(dirname(outputDirectory), `.${basename(outputDirectory)}.tmp-`));
  try {
    const specs = createFixedHandActionPlan(options);
    const totalPairs = specs.reduce((sum, item) => sum + item.actions.length, 0);
    const samples: NapoleonFixedMarginSample[] = [];
    const rawRollouts: NapoleonFixedMarginRawRollout[] = [];
    let completedPairs = 0;
    let completedRollouts = 0;
    for (const spec of specs) {
      for (const action of spec.actions) {
        const rows: NapoleonFixedRolloutRow[] = [];
        for (let repeatIndex = 0; repeatIndex < options.repeats; repeatIndex += 1) {
          const reshuffleSeed = stableUint32([
            options.randomSeed,
            "napoleon-fixed",
            spec.fixedHandId,
            action.actionIndex,
            repeatIndex
          ].join(":"));
          const row = await runNapoleonFixedContractRollout({
            spec,
            action,
            repeatIndex,
            reshuffleSeed
          });
          rows.push(row);
          rawRollouts.push(row.raw);
          completedRollouts += 1;
        }
        samples.push(aggregateNapoleonFixedRollouts({
          spec,
          action,
          rows,
          splitHint: options.reserveHandsForFinal && spec.reason?.includes("issue")
            ? "final-diagnostic"
            : null,
          rawRolloutShard: ""
        }));
        completedPairs += 1;
        options.onProgress?.({
          totalPairs,
          completedPairs,
          totalRollouts: totalPairs * options.repeats,
          completedRollouts,
          sampleCount: samples.length
        });
      }
    }
    const rawShards = await writeJsonlShards(
      tempDirectory,
      rawRollouts,
      options.gamesPerShard ?? 1000,
      "raw-rollouts"
    );
    const rowsPerShard = options.gamesPerShard ?? 1000;
    const sampleShardByFixedPair = new Map<string, string>();
    rawRollouts.forEach((row, index) => {
      const key = napoleonFixedPairKey(row.fixedHandId, row.forcedActionIndex);
      if (!sampleShardByFixedPair.has(key)) {
        sampleShardByFixedPair.set(key, rawRolloutShardForIndex(index, rowsPerShard));
      }
    });
    const samplesWithRawShard = samples.map((sample) => ({
      ...sample,
      rawRolloutShard:
        sampleShardByFixedPair.get(napoleonFixedPairKey(sample.fixedHandId, sample.forcedActionIndex)) ??
        "raw-rollouts-00000.jsonl"
    }));
    const shards = await writeJsonlShards(
      tempDirectory,
      samplesWithRawShard,
      options.gamesPerShard ?? 1000,
      "shard"
    );
    const summary = summarizeFixedHandBiddingMarginSamples(samplesWithRawShard);
    const manifest = createNapoleonFixedMarginManifest({
      options,
      samples: samplesWithRawShard,
      shards,
      rawShards,
      summary
    });
    await writeFile(join(tempDirectory, "manifest.json"), serializeManifest(manifest), "utf8");
    await writeFile(join(tempDirectory, "summary.json"), serializeManifest(summary), "utf8");
    await rename(tempDirectory, outputDirectory);
    return { outputDirectory, manifest, samples: samplesWithRawShard, rawRollouts };
  } catch (error) {
    await rm(tempDirectory, { recursive: true, force: true }).catch(() => undefined);
    throw error;
  }
}

export function validateGenerateFixedHandBiddingMarginDatasetOptions(
  options: GenerateFixedHandBiddingMarginDatasetOptions
): void {
  if (!options.outputDirectory) {
    throw new Error("outputDirectory is required.");
  }
  validatePositiveInteger("pairCount", options.pairCount);
  validatePositiveInteger("repeats", options.repeats);
  validateUint32("randomSeed", options.randomSeed);
  if (options.candidateSeatIndex !== undefined) {
    validateSeat(options.candidateSeatIndex);
  }
  if (options.actionCountPerHand !== undefined) {
    validatePositiveInteger("actionCountPerHand", options.actionCountPerHand);
  }
  if (options.gamesPerShard !== undefined) {
    validatePositiveInteger("gamesPerShard", options.gamesPerShard);
  }
}

export function createFixedHandActionPlan(
  options: GenerateFixedHandBiddingMarginDatasetOptions
): readonly FixedHandSpec[] {
  const actionCountPerHand = options.actionCountPerHand ?? 4;
  const reserved = [...(options.reservedHands ?? [])].map(normalizeFixedHandSpec);
  const reservedPairCount = reserved.reduce((sum, spec) => sum + spec.actions.length, 0);
  const randomHandCount = Math.max(
    0,
    Math.ceil(Math.max(0, options.pairCount - reservedPairCount) / actionCountPerHand)
  );
  const randomHands = createRandomFixedHands({
    handCount: randomHandCount,
    actionCountPerHand,
    randomSeed: options.randomSeed,
    candidateSeatIndex: options.candidateSeatIndex ?? 0
  });
  const merged = [...reserved, ...randomHands];
  let remainingPairs = options.pairCount;
  const planned: FixedHandSpec[] = [];
  for (const spec of merged) {
    if (remainingPairs <= 0) {
      break;
    }
    const actionLimit = Math.min(spec.actions.length, remainingPairs);
    planned.push({ ...spec, actions: spec.actions.slice(0, actionLimit) });
    remainingPairs -= actionLimit;
  }
  return planned;
}

export function createRandomFixedHands(options: {
  handCount: number;
  actionCountPerHand: number;
  randomSeed: number;
  candidateSeatIndex: number;
}): readonly FixedHandSpec[] {
  const rng = createSeededRandom(deriveSeed(options.randomSeed, "fixed-hand-specs"));
  const hands: FixedHandSpec[] = [];
  const seen = new Set<string>();
  while (hands.length < options.handCount) {
    const deck = [...createDeck()];
    shuffleInPlace(deck, rng);
    const hand = deck.slice(0, 10);
    const handIds = hand.map((card) => card.id).sort();
    const handKey = handIds.join(",");
    if (seen.has(handKey)) {
      continue;
    }
    seen.add(handKey);
    const strength = strongestSuitForHand(hand);
    hands.push({
      fixedHandId: `${hands.length}:${hashText(handKey)}`,
      handIds,
      candidateSeatIndex: options.candidateSeatIndex,
      sourceStateKey: null,
      sourceSeed: null,
      sourceBiddingStep: null,
      strongestSuit: strength.suit,
      strongestSuitScore: strength.score,
      reason: "random-fixed-hand-v1",
      actions: selectBidActionsForHand({
        hand,
        strongestSuit: strength.suit,
        actionCount: options.actionCountPerHand,
        rng
      })
    });
  }
  return hands;
}

export function createFixedHandInitialState(options: {
  seed: number;
  candidateSeatIndex: number;
  handIds: readonly string[];
}): ReturnType<typeof createInitialGame> {
  validateSeat(options.candidateSeatIndex);
  validateFixedHandIds(options.handIds);
  const state = createInitialGame({
    playerIds: PLAYER_IDS as unknown as readonly PlayerId[],
    rng: createSeededRandom(deriveSeed(options.seed, "initial-placeholder"))
  });
  const fixed = options.handIds.map((id) => cardById(id));
  const fixedSet = new Set(options.handIds);
  const remaining = createDeck().filter((card) => !fixedSet.has(card.id));
  shuffleInPlace(remaining, createSeededRandom(deriveSeed(options.seed, "fixed-hand-reshuffle")));
  let offset = 0;
  const players = state.players.map((player, playerIndex) => {
    if (playerIndex === options.candidateSeatIndex) {
      return { ...player, hand: fixed };
    }
    const hand = remaining.slice(offset, offset + 10);
    offset += 10;
    return { ...player, hand };
  });
  const unusedCards = remaining.slice(offset, offset + 3);
  const candidatePlayerId = PLAYER_IDS[options.candidateSeatIndex];
  return {
    ...state,
    players,
    unusedCards,
    currentPlayerId: candidatePlayerId,
    bidding: {
      starterPlayerId: candidatePlayerId,
      highestBid: null,
      consecutivePassCount: 0,
      history: []
    }
  };
}

export function assertDeckConservation(state: ReturnType<typeof createInitialGame>): void {
  const ids = [
    ...state.players.flatMap((player) => player.hand.map((card) => card.id)),
    ...state.unusedCards.map((card) => card.id)
  ];
  if (ids.length !== CARD_COUNT) {
    throw new Error(`expected ${CARD_COUNT} cards, got ${ids.length}.`);
  }
  if (new Set(ids).size !== CARD_COUNT) {
    throw new Error("duplicate card found in fixed-hand state.");
  }
}

function assertNapoleonFixedStartState(options: {
  state: ReturnType<typeof createInitialGame>;
  candidatePlayerId: PlayerId;
  targetPointCards: number;
  suit: Suit;
  handIds: readonly string[];
}): void {
  if (options.state.phase !== "choosing-adjutant") {
    throw new Error(`Napoleon-fixed rollout must start after bidding, got ${options.state.phase}.`);
  }
  if (options.state.bidding !== null) {
    throw new Error("Napoleon-fixed rollout must not retain a bidding state.");
  }
  if (options.state.contract?.napoleonPlayerId !== options.candidatePlayerId) {
    throw new Error("contract owner is not candidate.");
  }
  if (options.state.contract.targetPointCards !== options.targetPointCards) {
    throw new Error("contract target does not match requested target.");
  }
  if (options.state.contract.trumpSuit !== options.suit || options.state.trumpSuit !== options.suit) {
    throw new Error("contract suit does not match requested suit.");
  }
  if (!sameStringArray(handIdsForPlayer(options.state, options.candidatePlayerId), [...options.handIds].sort())) {
    throw new Error("candidate hand is not fixed at Napoleon-fixed rollout start.");
  }
  assertDeckConservation(options.state);
}

function assertFinalDeckConservation(state: ReturnType<typeof createInitialGame>): void {
  const ids = [
    ...state.players.flatMap((player) => player.hand.map((card) => card.id)),
    ...state.unusedCards.map((card) => card.id),
    ...state.excludedCards.map((card) => card.id),
    ...state.awardedPointCards.flatMap((award) => award.cards.map((card) => card.id)),
    ...state.completedTricks.flatMap((trick) => trick.cards.map((played) => played.card.id))
  ];
  if (ids.length !== CARD_COUNT) {
    throw new Error(`expected ${CARD_COUNT} terminal cards, got ${ids.length}.`);
  }
  if (new Set(ids).size !== CARD_COUNT) {
    throw new Error("duplicate card found in terminal state.");
  }
}

function handIdsForPlayer(
  state: ReturnType<typeof createInitialGame>,
  playerId: PlayerId
): string[] {
  const player = state.players.find((candidate) => candidate.id === playerId);
  if (player === undefined) {
    throw new Error(`missing player ${playerId}.`);
  }
  return player.hand.map((card) => card.id).sort();
}

export function aggregateFixedHandBidRollouts(options: {
  spec: FixedHandSpec;
  action: FixedHandBiddingActionSpec;
  rows: readonly RolloutRow[];
  splitHint: "final-diagnostic" | null;
}): FixedHandBiddingMarginSample {
  if (options.rows.length === 0) {
    throw new Error("cannot aggregate an empty rollout group.");
  }
  const margins = options.rows.map((row) => row.contractMargin);
  const first = options.rows[0];
  return {
    sampleType: FIXED_HAND_BIDDING_MARGIN_DATASET_SAMPLE_TYPE,
    schemaVersion: FIXED_HAND_BIDDING_MARGIN_SAMPLE_SCHEMA_VERSION,
    fixedHandId: options.spec.fixedHandId,
    handIds: options.spec.handIds,
    candidateSeatIndex: options.spec.candidateSeatIndex,
    candidatePlayerId: PLAYER_IDS[options.spec.candidateSeatIndex],
    sourceStateKey: options.spec.sourceStateKey ?? null,
    sourceSeed: options.spec.sourceSeed ?? null,
    sourceBiddingStep: options.spec.sourceBiddingStep ?? null,
    strongestSuit: options.spec.strongestSuit,
    strongestSuitScore: options.spec.strongestSuitScore,
    forcedActionIndex: options.action.actionIndex,
    forcedTargetPointCards: options.action.targetPointCards,
    forcedSuit: options.action.suit,
    forcedActionLabel: options.action.label ?? actionLabel(options.action),
    modelInput: first.modelInput,
    legalBidMask: first.legalBidMask,
    rolloutCount: options.rows.length,
    empiricalMarginMean: mean(margins),
    empiricalMarginStd: std(margins),
    empiricalWinRate: mean(options.rows.map((row) => row.contractSuccess ? 1 : 0)),
    empiricalMarginMin: Math.min(...margins),
    empiricalMarginMax: Math.max(...margins),
    marginHistogram: countBy(margins, (value) => String(value)),
    resultTypeCounts: countBy(options.rows, (row) => row.resultType),
    finalRoleCounts: countBy(options.rows, (row) => row.finalRole),
    sourceNnMu: options.action.sourceNnMu ?? null,
    sourceNnSigma: options.action.sourceNnSigma ?? null,
    sourceNnPWin: options.action.sourceNnPWin ?? null,
    splitHint: options.splitHint
  };
}

export function aggregateNapoleonFixedRollouts(options: {
  spec: FixedHandSpec;
  action: FixedHandBiddingActionSpec;
  rows: readonly NapoleonFixedRolloutRow[];
  splitHint: "final-diagnostic" | null;
  rawRolloutShard: string;
}): NapoleonFixedMarginSample {
  if (options.rows.length === 0) {
    throw new Error("cannot aggregate an empty Napoleon-fixed rollout group.");
  }
  const margins = options.rows.map((row) => row.contractMargin);
  const first = options.rows[0];
  return {
    sampleType: NAPOLEON_FIXED_MARGIN_DATASET_SAMPLE_TYPE,
    schemaVersion: NAPOLEON_FIXED_MARGIN_SAMPLE_SCHEMA_VERSION,
    fixedHandId: options.spec.fixedHandId,
    handIds: options.spec.handIds,
    candidateSeatIndex: options.spec.candidateSeatIndex,
    candidatePlayerId: PLAYER_IDS[options.spec.candidateSeatIndex],
    sourceStateKey: options.spec.sourceStateKey ?? null,
    sourceSeed: options.spec.sourceSeed ?? null,
    sourceBiddingStep: options.spec.sourceBiddingStep ?? null,
    strongestSuit: options.spec.strongestSuit,
    strongestSuitScore: options.spec.strongestSuitScore,
    forcedActionIndex: options.action.actionIndex,
    forcedTargetPointCards: options.action.targetPointCards,
    forcedSuit: options.action.suit,
    forcedActionLabel: options.action.label ?? actionLabel(options.action),
    modelInput: first.modelInput,
    legalBidMask: first.legalBidMask,
    rolloutCount: options.rows.length,
    empiricalMarginMean: mean(margins),
    empiricalMarginStd: std(margins),
    empiricalWinRate: mean(options.rows.map((row) => row.contractSuccess ? 1 : 0)),
    empiricalMarginMin: Math.min(...margins),
    empiricalMarginMax: Math.max(...margins),
    marginHistogram: countBy(margins, (value) => String(value)),
    resultTypeCounts: countBy(options.rows, (row) => row.resultType),
    finalRoleCounts: countBy(options.rows, (row) => row.finalRole) as Record<"napoleon" | "napoleon-adjutant", number>,
    invariantFailures: {},
    sourceNnMu: options.action.sourceNnMu ?? null,
    sourceNnSigma: options.action.sourceNnSigma ?? null,
    sourceNnPWin: options.action.sourceNnPWin ?? null,
    splitHint: options.splitHint,
    rawRolloutShard: options.rawRolloutShard
  };
}

export function summarizeFixedHandBiddingMarginSamples(
  samples: readonly (FixedHandBiddingMarginSample | NapoleonFixedMarginSample)[]
): FixedHandBiddingMarginDatasetSummary {
  const repeats = samples.map((sample) => sample.rolloutCount);
  return {
    pairCount: samples.length,
    fixedHandCount: new Set(samples.map((sample) => sample.fixedHandId)).size,
    rolloutCount: sum(repeats),
    repeatCountMin: repeats.length === 0 ? 0 : Math.min(...repeats),
    repeatCountMax: repeats.length === 0 ? 0 : Math.max(...repeats),
    empiricalMarginMean: numericSummary(samples.map((sample) => sample.empiricalMarginMean)),
    empiricalMarginStd: numericSummary(samples.map((sample) => sample.empiricalMarginStd)),
    empiricalWinRate: numericSummary(samples.map((sample) => sample.empiricalWinRate)),
    suitCounts: suitCounts(samples, (sample) => sample.forcedSuit),
    targetCounts: countBy(samples, (sample) => String(sample.forcedTargetPointCards)),
    splitHintCounts: countBy(samples, (sample) => sample.splitHint ?? "none")
  };
}

export function stableUint32(text: string): number {
  return createHash("sha256").update(text).digest().readUInt32BE(0);
}

async function runFixedHandBidRollout(options: {
  spec: FixedHandSpec;
  action: FixedHandBiddingActionSpec;
  repeatIndex: number;
  reshuffleSeed: number;
}): Promise<RolloutRow> {
  const candidatePlayerId = PLAYER_IDS[options.spec.candidateSeatIndex];
  let state = createFixedHandInitialState({
    seed: options.reshuffleSeed,
    candidateSeatIndex: options.spec.candidateSeatIndex,
    handIds: options.spec.handIds
  });
  const publicActionHistory: Array<{
    step: number;
    playerId: PlayerId;
    phase: "bidding";
    action: PublicBiddingAction;
  }> = [];
  const agents = new Map<PlayerId, Agent>(PLAYER_IDS.map((playerId, playerIndex) => [
    playerId,
    playerIndex === options.spec.candidateSeatIndex
      ? new ForcedCandidateAgent({ action: options.action, rngSeed: options.reshuffleSeed })
      : createFrozenAgent({
          sourceSeed: options.spec.sourceSeed ?? options.reshuffleSeed,
          candidateSeatIndex: options.spec.candidateSeatIndex,
          playerIndex,
          rngSeed: options.reshuffleSeed
        })
  ]));
  let forcedModelInput: readonly number[] | null = null;
  let forcedLegalBidMask: readonly number[] | null = null;
  let step = 0;
  while (!state.isGameOver) {
    if (state.isTrickComplete) {
      state = advanceToNextTrick(clearLatestEvent(state));
      continue;
    }
    if (step > 1000) {
      throw new Error("decision limit exceeded.");
    }
    const playerId = state.currentPlayerId;
    const observation = createObservation(state, playerId, publicActionHistory);
    const agent = agents.get(playerId);
    if (agent === undefined) {
      throw new Error(`missing agent for ${playerId}.`);
    }
    const beforePhase = state.phase;
    const action = await agent.selectAction(observation) as ForcedActionWithMetadata;
    if (!observation.legalActions.some((legal) => actionsEqual(legal, action))) {
      throw new Error(`illegal action selected: ${JSON.stringify(action)}`);
    }
    step += 1;
    if (beforePhase === "bidding" && (action.type === "bid" || action.type === "pass")) {
      publicActionHistory.push({ step, playerId, phase: beforePhase, action });
    }
    if (
      playerId === candidatePlayerId &&
      beforePhase === "bidding" &&
      action.type === "bid" &&
      action.__forced === true
    ) {
      forcedModelInput = action.__modelInput ?? null;
      forcedLegalBidMask = action.__legalBidMask ?? null;
      delete action.__forced;
      delete action.__modelInput;
      delete action.__legalBidMask;
    }
    state = applyAction(clearLatestEvent(state), action);
  }
  if (state.result === null || forcedModelInput === null || forcedLegalBidMask === null) {
    throw new Error("fixed-hand rollout did not produce a forced terminal result.");
  }
  const contractMargin = state.result.resultType === "standard"
    ? state.result.napoleonTeamPointCards - state.result.targetPointCards
    : -20;
  return {
    fixedHandId: options.spec.fixedHandId,
    contractMargin,
    contractSuccess: state.result.resultType === "standard" && state.result.winner === "napoleon-team",
    resultType: state.result.resultType,
    finalRole: terminalRoleForResult(state.result, candidatePlayerId),
    modelInput: forcedModelInput,
    legalBidMask: forcedLegalBidMask
  };
}

async function runNapoleonFixedContractRollout(options: {
  spec: FixedHandSpec;
  action: FixedHandBiddingActionSpec;
  repeatIndex: number;
  reshuffleSeed: number;
}): Promise<NapoleonFixedRolloutRow> {
  const candidatePlayerId = PLAYER_IDS[options.spec.candidateSeatIndex];
  const initialState = createFixedHandInitialState({
    seed: options.reshuffleSeed,
    candidateSeatIndex: options.spec.candidateSeatIndex,
    handIds: options.spec.handIds
  });
  assertDeckConservation(initialState);
  const candidateObservation = createObservation(initialState, candidatePlayerId, []);
  const encoded = encodeBiddingObservation(
    candidateObservation,
    candidateObservation.view.players.map((player) => player.id)
  );
  const { modelInput, legalBidMask } = createBiddingModelInput(encoded);
  if (Number(legalBidMask[options.action.actionIndex]) !== 1) {
    throw new Error(`contract action ${options.action.actionIndex} is illegal in fixed-hand context.`);
  }
  const candidateHandBeforeContract = handIdsForPlayer(initialState, candidatePlayerId);
  if (!sameStringArray(candidateHandBeforeContract, [...options.spec.handIds].sort())) {
    throw new Error("candidate hand changed before Napoleon-fixed contract start.");
  }
  let state = createContractEstablishedState(initialState, {
    napoleonPlayerId: candidatePlayerId,
    trumpSuit: options.action.suit,
    targetPointCards: options.action.targetPointCards
  });
  assertNapoleonFixedStartState({
    state,
    candidatePlayerId,
    targetPointCards: options.action.targetPointCards,
    suit: options.action.suit,
    handIds: options.spec.handIds
  });

  const agents = new Map<PlayerId, Agent>(PLAYER_IDS.map((playerId, playerIndex) => [
    playerId,
    new RuleBasedAgent(createSeededRandom(deriveSeed(options.reshuffleSeed, `napoleon-fixed-agent:${playerIndex}`)))
  ]));
  let downstreamBiddingActionCount = 0;
  let step = 0;
  while (!state.isGameOver) {
    if (state.isTrickComplete) {
      state = advanceToNextTrick(clearLatestEvent(state));
      continue;
    }
    if (step > 1000) {
      throw new Error("decision limit exceeded.");
    }
    const playerId = state.currentPlayerId;
    const observation = createObservation(state, playerId, []);
    const agent = agents.get(playerId);
    if (agent === undefined) {
      throw new Error(`missing agent for ${playerId}.`);
    }
    const beforePhase = state.phase;
    if (beforePhase === "bidding") {
      downstreamBiddingActionCount += 1;
    }
    const action = await agent.selectAction(observation);
    if (!observation.legalActions.some((legal) => actionsEqual(legal, action))) {
      throw new Error(`illegal action selected: ${JSON.stringify(action)}`);
    }
    step += 1;
    state = applyAction(clearLatestEvent(state), action);
  }
  if (state.result === null || state.result.resultType !== "standard") {
    throw new Error("Napoleon-fixed rollout did not produce a standard terminal result.");
  }
  if (downstreamBiddingActionCount !== 0) {
    throw new Error(`Napoleon-fixed rollout executed ${downstreamBiddingActionCount} bidding actions.`);
  }
  assertFinalDeckConservation(state);
  const contractMargin = state.result.napoleonTeamPointCards - state.result.targetPointCards;
  const contractSuccess = state.result.winner === "napoleon-team";
  const finalRole = terminalRoleForResult(state.result, candidatePlayerId);
  if (finalRole !== "napoleon" && finalRole !== "napoleon-adjutant") {
    throw new Error(`candidate final role is not Napoleon: ${finalRole}.`);
  }
  if (state.contract?.napoleonPlayerId !== candidatePlayerId) {
    throw new Error("candidate is no longer the contract owner at terminal state.");
  }
  if (
    state.contract.targetPointCards !== options.action.targetPointCards ||
    state.contract.trumpSuit !== options.action.suit
  ) {
    throw new Error("terminal contract target/suit changed.");
  }
  const raw: NapoleonFixedMarginRawRollout = {
    sampleType: "napoleon-fixed-contract-margin-raw-rollout",
    schemaVersion: 1,
    fixedHandId: options.spec.fixedHandId,
    handIds: options.spec.handIds,
    candidateSeatIndex: options.spec.candidateSeatIndex,
    candidatePlayerId,
    forcedActionIndex: options.action.actionIndex,
    forcedTargetPointCards: options.action.targetPointCards,
    forcedSuit: options.action.suit,
    repeatIndex: options.repeatIndex,
    hiddenDealSeed: options.reshuffleSeed,
    finalAdjutant: {
      calledCardId: state.adjutant?.calledCardId ?? "",
      playerId: state.result.adjutantPlayerId
    },
    napoleonSidePointCards: state.result.napoleonTeamPointCards,
    coalitionSidePointCards: state.result.alliancePointCards,
    contractMargin,
    contractSuccess,
    candidateRawReward: contractSuccess ? 2 * options.action.targetPointCards : 0,
    candidateRelativeReward: contractSuccess
      ? (7 * options.action.targetPointCards) / 4
      : (-3 * options.action.targetPointCards) / 4,
    resultType: state.result.resultType,
    finalRole,
    invariantChecks: {
      candidateRoleNapoleon: true,
      contractOwnerCandidate: true,
      targetMatches: true,
      suitMatches: true,
      downstreamBiddingActionCount: 0,
      candidateHandFixedAtStart: true,
      deckConservation: true
    }
  };
  return {
    fixedHandId: options.spec.fixedHandId,
    contractMargin,
    contractSuccess,
    resultType: state.result.resultType,
    finalRole,
    modelInput: Array.from(modelInput),
    legalBidMask: Array.from(legalBidMask),
    raw
  };
}

class ForcedCandidateAgent implements Agent {
  readonly ruleBased: RuleBasedAgent;
  readonly action: FixedHandBiddingActionSpec;
  forced = false;

  constructor(options: { action: FixedHandBiddingActionSpec; rngSeed: number }) {
    this.action = options.action;
    this.ruleBased = new RuleBasedAgent(createSeededRandom(deriveSeed(options.rngSeed, "forced-candidate")));
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    if (observation.view.phase === "bidding" && !this.forced) {
      const encoded = encodeBiddingObservation(
        observation,
        observation.view.players.map((player) => player.id)
      );
      const { modelInput, legalBidMask } = createBiddingModelInput(encoded);
      if (Number(legalBidMask[this.action.actionIndex]) !== 1) {
        throw new Error(`forced action ${this.action.actionIndex} is illegal in fixed-hand context.`);
      }
      const decoded = decodeBiddingAction(this.action.actionIndex, observation.playerId);
      const legalAction = findMatchingLegalAction(observation, decoded);
      this.forced = true;
      return {
        ...legalAction,
        __forced: true,
        __modelInput: Array.from(modelInput),
        __legalBidMask: Array.from(legalBidMask)
      } as ForcedActionWithMetadata;
    }
    return this.ruleBased.selectAction(observation);
  }
}

class FrozenAgent implements Agent {
  readonly ruleBased: RuleBasedAgent;
  readonly conservative: ConservativeBiddingAgent;
  readonly biddingPolicyType: "rule-based" | "conservative-bidding";

  constructor(options: { rngSeed: number; biddingPolicyType: "rule-based" | "conservative-bidding" }) {
    const rng = createSeededRandom(deriveSeed(options.rngSeed, "frozen-agent"));
    this.ruleBased = new RuleBasedAgent(rng);
    this.conservative = new ConservativeBiddingAgent(rng);
    this.biddingPolicyType = options.biddingPolicyType;
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    if (observation.view.phase === "bidding" && this.biddingPolicyType === "conservative-bidding") {
      return this.conservative.selectAction(observation);
    }
    return this.ruleBased.selectAction(observation);
  }
}

function createFrozenAgent(options: {
  sourceSeed: number;
  candidateSeatIndex: number;
  playerIndex: number;
  rngSeed: number;
}): Agent {
  return new FrozenAgent({
    rngSeed: deriveSeed(options.rngSeed, `agent:${options.playerIndex}`),
    biddingPolicyType: selectFrozenPolicy(options)
  });
}

function selectFrozenPolicy(options: {
  sourceSeed: number;
  candidateSeatIndex: number;
  playerIndex: number;
}): "rule-based" | "conservative-bidding" {
  const digest = createHash("sha256")
    .update(`${BIDDING_Q_COUNTERFACTUAL_OPPONENT_MIX_RULE_VERSION}:${options.sourceSeed}:${options.candidateSeatIndex}:${options.playerIndex}`)
    .digest();
  return digest.readUInt32BE(0) % 2 === 0 ? "rule-based" : "conservative-bidding";
}

function createObservation(
  state: ReturnType<typeof createInitialGame>,
  playerId: PlayerId,
  publicActionHistory: readonly { step: number; playerId: PlayerId; phase: "bidding"; action: PublicBiddingAction }[]
): PlayerObservation {
  const view = createPlayerView(state, playerId);
  return {
    playerId,
    view,
    legalActions: getAutomatedLegalActions(state, playerId),
    publicActionHistory: publicActionHistory.map(({ step, playerId: actorId, phase, action }) => ({
      step,
      playerId: actorId,
      phase,
      action
    }))
  };
}

function getAutomatedLegalActions(
  state: ReturnType<typeof createInitialGame>,
  playerId: PlayerId
): GameAction[] {
  if (state.phase !== "exchanging") {
    return [...getLegalActions(state, playerId)];
  }
  const view = createPlayerView(state, playerId);
  const discardCount = view.exchangeRequirement?.discardCount;
  const self = view.players.find((player) => player.id === playerId);
  if (discardCount === undefined || self?.hand === undefined) {
    return [];
  }
  return combinations(self.hand, discardCount).map((cards) => ({
    type: "discard-cards",
    playerId,
    cardIds: cards.map((card) => card.id)
  }));
}

function normalizeFixedHandSpec(spec: FixedHandSpec): FixedHandSpec {
  validateSeat(spec.candidateSeatIndex);
  validateFixedHandIds(spec.handIds);
  const hand = spec.handIds.map((id) => cardById(id));
  const strength = strongestSuitForHand(hand);
  return {
    ...spec,
    strongestSuit: spec.strongestSuit ?? strength.suit,
    strongestSuitScore: spec.strongestSuitScore ?? strength.score,
    actions: spec.actions.map((action) => normalizeBidAction(action))
  };
}

function selectBidActionsForHand(options: {
  hand: readonly Card[];
  strongestSuit: Suit;
  actionCount: number;
  rng: () => number;
}): readonly FixedHandBiddingActionSpec[] {
  const candidates: FixedHandBiddingActionSpec[] = [];
  const alternateSuits = SUITS.filter((suit) => suit !== options.strongestSuit);
  const highStrongestTarget = 15 + Math.floor(options.rng() * 5);
  const randomTarget = TARGETS[Math.floor(options.rng() * TARGETS.length)];
  const randomSuit = SUITS[Math.floor(options.rng() * SUITS.length)];
  candidates.push(normalizeBidAction({ actionIndex: bidActionIndex(13, options.strongestSuit), targetPointCards: 13, suit: options.strongestSuit }));
  candidates.push(normalizeBidAction({ actionIndex: bidActionIndex(highStrongestTarget, options.strongestSuit), targetPointCards: highStrongestTarget, suit: options.strongestSuit }));
  candidates.push(normalizeBidAction({ actionIndex: bidActionIndex(13, alternateSuits[0]), targetPointCards: 13, suit: alternateSuits[0] }));
  candidates.push(normalizeBidAction({ actionIndex: bidActionIndex(randomTarget, randomSuit), targetPointCards: randomTarget, suit: randomSuit }));
  while (candidates.length < options.actionCount * 2) {
    const target = TARGETS[Math.floor(options.rng() * TARGETS.length)];
    const suit = SUITS[Math.floor(options.rng() * SUITS.length)];
    candidates.push(normalizeBidAction({ actionIndex: bidActionIndex(target, suit), targetPointCards: target, suit }));
  }
  const unique = new Map<number, FixedHandBiddingActionSpec>();
  for (const candidate of candidates) {
    unique.set(candidate.actionIndex, candidate);
  }
  return [...unique.values()].slice(0, options.actionCount);
}

function bidActionIndex(targetPointCards: number, suit: Suit): number {
  for (let index = 1; index < BIDDING_ACTION_COUNT; index += 1) {
    const decoded = decodeBiddingQActionIndex(index);
    if (
      decoded.type === "bid" &&
      decoded.targetPointCards === targetPointCards &&
      decoded.suit === suit
    ) {
      return index;
    }
  }
  throw new Error(`cannot encode bid ${targetPointCards} ${suit}.`);
}

function normalizeBidAction(action: FixedHandBiddingActionSpec): FixedHandBiddingActionSpec {
  const decoded = decodeBiddingQActionIndex(action.actionIndex);
  if (decoded.type !== "bid") {
    throw new Error("fixed-hand margin dataset only supports BID actions.");
  }
  if (decoded.targetPointCards !== action.targetPointCards || decoded.suit !== action.suit) {
    throw new Error("actionIndex does not match target/suit.");
  }
  return {
    ...action,
    label: action.label ?? actionLabel(action)
  };
}

function actionLabel(action: FixedHandBiddingActionSpec): string {
  return `${action.suit}${action.targetPointCards}`;
}

function strongestSuitForHand(hand: readonly Card[]): { suit: Suit; score: number } {
  const scores = SUITS.map((suit) => ({
    suit,
    score: evaluateHandForTrump(hand, suit)
  }));
  return scores.sort((left, right) => right.score - left.score)[0];
}

function terminalRoleForResult(result: GameResult, actingPlayerId: PlayerId): string {
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

function actionsEqual(left: GameAction, right: GameAction): boolean {
  if (left.type !== right.type || left.playerId !== right.playerId) {
    return false;
  }
  if (left.type === "bid" && right.type === "bid") {
    return left.suit === right.suit && left.targetPointCards === right.targetPointCards;
  }
  if (left.type === "pass") return true;
  if (left.type === "play-card" && right.type === "play-card") return left.cardId === right.cardId;
  if (left.type === "choose-adjutant" && right.type === "choose-adjutant") {
    return left.cardId === right.cardId;
  }
  if (left.type === "discard-cards" && right.type === "discard-cards") {
    return sameStringArray([...left.cardIds].sort(), [...right.cardIds].sort());
  }
  return false;
}

function findMatchingLegalAction(observation: PlayerObservation, selectedAction: GameAction): GameAction {
  const action = observation.legalActions.find((legal) => actionsEqual(legal, selectedAction));
  if (action === undefined) {
    throw new Error(`selected action outside legal actions ${JSON.stringify(selectedAction)}`);
  }
  return action;
}

async function writeSampleShards(
  directory: string,
  samples: readonly FixedHandBiddingMarginSample[],
  gamesPerShard: number
): Promise<DatasetShardManifest[]> {
  await mkdir(directory, { recursive: true });
  const shards: DatasetShardManifest[] = [];
  for (let start = 0; start < samples.length; start += gamesPerShard) {
    const shardSamples = samples.slice(start, start + gamesPerShard);
    const relativePath = `shard-${String(shards.length).padStart(5, "0")}.jsonl`;
    const body = shardSamples.map((sample) => `${JSON.stringify(sample)}\n`).join("");
    await writeFile(join(directory, relativePath), body, "utf8");
    shards.push({
      file: relativePath,
      startSeed: 0,
      endSeed: 0,
      gameCount: shardSamples.length,
      sampleCount: shardSamples.length,
      byteLength: Buffer.byteLength(body, "utf8"),
      sha256: createHash("sha256").update(body).digest("hex")
    });
  }
  return shards;
}

async function writeJsonlShards<T>(
  directory: string,
  rows: readonly T[],
  rowsPerShard: number,
  filePrefix: string
): Promise<DatasetShardManifest[]> {
  await mkdir(directory, { recursive: true });
  const shards: DatasetShardManifest[] = [];
  for (let start = 0; start < rows.length; start += rowsPerShard) {
    const shardRows = rows.slice(start, start + rowsPerShard);
    const relativePath = `${filePrefix}-${String(shards.length).padStart(5, "0")}.jsonl`;
    const body = shardRows.map((row) => `${JSON.stringify(row)}\n`).join("");
    await writeFile(join(directory, relativePath), body, "utf8");
    shards.push({
      file: relativePath,
      startSeed: 0,
      endSeed: 0,
      gameCount: shardRows.length,
      sampleCount: shardRows.length,
      byteLength: Buffer.byteLength(body, "utf8"),
      sha256: createHash("sha256").update(body).digest("hex")
    });
  }
  return shards;
}

function createFixedHandBiddingMarginManifest(options: {
  options: GenerateFixedHandBiddingMarginDatasetOptions;
  samples: readonly FixedHandBiddingMarginSample[];
  shards: readonly DatasetShardManifest[];
  summary: FixedHandBiddingMarginDatasetSummary;
}): FixedHandBiddingMarginDatasetManifest {
  return {
    datasetSchemaVersion: FIXED_HAND_BIDDING_MARGIN_DATASET_SCHEMA_VERSION,
    generatorVersion: FIXED_HAND_BIDDING_MARGIN_DATASET_GENERATOR_VERSION,
    format: DATASET_FORMAT,
    sampleType: FIXED_HAND_BIDDING_MARGIN_DATASET_SAMPLE_TYPE,
    sampleSchemaVersion: FIXED_HAND_BIDDING_MARGIN_SAMPLE_SCHEMA_VERSION,
    teacher: {
      id: FIXED_HAND_BIDDING_MARGIN_TEACHER_ID,
      primaryLabel: "empiricalMarginMean",
      stdLabel: "empiricalMarginStd",
      winRateLabel: "empiricalWinRate",
      repeatsPerPair: options.options.repeats
    },
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
    fixedCondition: {
      fixed: ["candidate hand", "candidate seat index", "forced BID action"],
      varied: ["other four hands", "kitty cards", "downstream bidding", "adjutant choice", "exchange", "play"],
      note: "Candidate is forced as the initial bidding actor so fixed-hand/action pairs isolate hidden-deal variance."
    },
    opponentMix: {
      mixingRuleVersion: BIDDING_Q_COUNTERFACTUAL_OPPONENT_MIX_RULE_VERSION,
      ruleBasedWeight: 0.5,
      conservativeWeight: 0.5
    },
    pairCount: options.samples.length,
    fixedHandCount: new Set(options.samples.map((sample) => sample.fixedHandId)).size,
    rolloutCount: sum(options.samples.map((sample) => sample.rolloutCount)),
    randomSeed: options.options.randomSeed,
    sourceCommit: options.options.sourceCommit ?? null,
    playerCount: PLAYER_COUNT,
    cardCount: CARD_COUNT,
    cardIds: CARD_IDS,
    cardIdsSha256: calculateCardIdsSha256(),
    summary: options.summary,
    shardCount: options.shards.length,
    shards: options.shards
  };
}

function createNapoleonFixedMarginManifest(options: {
  options: GenerateNapoleonFixedMarginDatasetOptions;
  samples: readonly NapoleonFixedMarginSample[];
  shards: readonly DatasetShardManifest[];
  rawShards: readonly DatasetShardManifest[];
  summary: FixedHandBiddingMarginDatasetSummary;
}): NapoleonFixedMarginDatasetManifest {
  return {
    datasetSchemaVersion: NAPOLEON_FIXED_MARGIN_DATASET_SCHEMA_VERSION,
    generatorVersion: NAPOLEON_FIXED_MARGIN_DATASET_GENERATOR_VERSION,
    format: DATASET_FORMAT,
    sampleType: NAPOLEON_FIXED_MARGIN_DATASET_SAMPLE_TYPE,
    sampleSchemaVersion: NAPOLEON_FIXED_MARGIN_SAMPLE_SCHEMA_VERSION,
    teacher: {
      id: NAPOLEON_FIXED_MARGIN_TEACHER_ID,
      primaryLabel: "empiricalMarginMean",
      stdLabel: "empiricalMarginStd",
      winRateLabel: "empiricalWinRate",
      repeatsPerPair: options.options.repeats
    },
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
    fixedCondition: {
      fixed: [
        "candidate hand",
        "candidate seat index",
        "candidate role = Napoleon",
        "contract owner = candidate",
        "contract target",
        "contract suit"
      ],
      varied: ["other four hands", "kitty cards", "adjutant choice", "exchange", "play"],
      note:
        "Rollouts start from a contract-established game-core state. No BID/PASS action is applied during rollout; hidden deal reshuffle is the only repeated variation."
    },
    opponentMix: {
      mixingRuleVersion: BIDDING_Q_COUNTERFACTUAL_OPPONENT_MIX_RULE_VERSION,
      ruleBasedWeight: 1.0,
      conservativeWeight: 0.0
    },
    pairCount: options.samples.length,
    fixedHandCount: new Set(options.samples.map((sample) => sample.fixedHandId)).size,
    rolloutCount: sum(options.samples.map((sample) => sample.rolloutCount)),
    randomSeed: options.options.randomSeed,
    sourceCommit: options.options.sourceCommit ?? null,
    playerCount: PLAYER_COUNT,
    cardCount: CARD_COUNT,
    cardIds: CARD_IDS,
    cardIdsSha256: calculateCardIdsSha256(),
    summary: options.summary,
    shardCount: options.shards.length,
    shards: options.shards,
    rawRolloutShards: options.rawShards
  };
}

function napoleonFixedPairKey(fixedHandId: string, actionIndex: number): string {
  return `${fixedHandId}:${actionIndex}`;
}

function rawRolloutShardForIndex(index: number, rowsPerShard: number): string {
  return `raw-rollouts-${String(Math.floor(index / rowsPerShard)).padStart(5, "0")}.jsonl`;
}

function validateFixedHandIds(handIds: readonly string[]): void {
  if (handIds.length !== 10) {
    throw new Error("fixed hand must contain exactly 10 cards.");
  }
  if (new Set(handIds).size !== handIds.length) {
    throw new Error("fixed hand contains duplicate cards.");
  }
  for (const id of handIds) {
    cardById(id);
  }
}

function validateSeat(value: number): void {
  if (!Number.isInteger(value) || value < 0 || value >= PLAYER_COUNT) {
    throw new Error("candidateSeatIndex must be an integer seat index.");
  }
}

function validatePositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function validateUint32(name: string, value: number): void {
  if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
    throw new Error(`${name} must be a uint32 integer.`);
  }
}

function cardById(id: string): Card {
  const card = CARD_BY_ID.get(id);
  if (card === undefined) {
    throw new Error(`unknown card ${id}.`);
  }
  return card;
}

function combinations<T>(items: readonly T[], count: number): T[][] {
  if (count === 0) return [[]];
  if (items.length < count) return [];
  const result: T[][] = [];
  for (let index = 0; index <= items.length - count; index += 1) {
    for (const tail of combinations(items.slice(index + 1), count - 1)) {
      result.push([items[index], ...tail]);
    }
  }
  return result;
}

function shuffleInPlace<T>(values: T[], rng: () => number): void {
  for (let index = values.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(rng() * (index + 1));
    [values[index], values[swapIndex]] = [values[swapIndex], values[index]];
  }
}

function hashText(text: string): string {
  return createHash("sha256").update(text).digest("hex").slice(0, 12);
}

function countBy<T>(values: readonly T[], keyFn: (value: T) => string): Record<string, number> {
  const result: Record<string, number> = {};
  for (const value of values) {
    const key = keyFn(value);
    result[key] = (result[key] ?? 0) + 1;
  }
  return Object.fromEntries(
    Object.entries(result).sort(([left], [right]) => left.localeCompare(right, undefined, { numeric: true }))
  );
}

function suitCounts<T>(values: readonly T[], keyFn: (value: T) => Suit): Record<Suit, number> {
  const result = { spades: 0, hearts: 0, diamonds: 0, clubs: 0 };
  for (const value of values) {
    result[keyFn(value)] += 1;
  }
  return result;
}

function mean(values: readonly number[]): number {
  if (values.length === 0) {
    throw new Error("mean requires at least one value.");
  }
  return sum(values) / values.length;
}

function std(values: readonly number[]): number {
  if (values.length === 0) {
    throw new Error("std requires at least one value.");
  }
  const average = mean(values);
  return Math.sqrt(values.reduce((total, value) => total + (value - average) ** 2, 0) / values.length);
}

function numericSummary(values: readonly number[]): NumericSummary {
  if (values.length === 0) {
    return { count: 0, mean: null, std: null, min: null, max: null };
  }
  return {
    count: values.length,
    mean: mean(values),
    std: std(values),
    min: Math.min(...values),
    max: Math.max(...values)
  };
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function sameStringArray(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}
