import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import {
  ConservativeBiddingAgent,
  RuleBasedAgent,
  createSeededRandom,
  deriveSeed
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
import type { Card, GameAction, GameResult, GameState, PlayerId, Suit } from "@napoleon/game-core";
import {
  BIDDING_ACTION_COUNT,
  BIDDING_ENCODER_SCHEMA_VERSION,
  BIDDING_HISTORY_SUIT_ORDER,
  BIDDING_MODEL_INPUT_FEATURE_COUNT,
  BIDDING_MODEL_INPUT_SCHEMA_VERSION,
  CARD_COUNT,
  CARD_IDS,
  createBiddingModelInput,
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
import {
  createRandomFixedHands,
  stableUint32
} from "./generateFixedHandBiddingMarginDataset.js";
import type {
  FixedHandBiddingActionSpec,
  FixedHandBiddingMarginDatasetSummary,
  FixedHandBiddingMarginSample,
  FixedHandSpec,
  NumericSummary
} from "./generateFixedHandBiddingMarginDataset.js";

export const HISTORY_CONSISTENT_RAISE_MARGIN_DATASET_SAMPLE_TYPE =
  "history-consistent-raise-margin-sample" as const;
export const HISTORY_CONSISTENT_RAISE_MARGIN_SAMPLE_SCHEMA_VERSION = 1 as const;
export const HISTORY_CONSISTENT_RAISE_MARGIN_DATASET_SCHEMA_VERSION = 1 as const;
export const HISTORY_CONSISTENT_RAISE_MARGIN_DATASET_GENERATOR_VERSION = 1 as const;
export const HISTORY_CONSISTENT_RAISE_MARGIN_TEACHER_ID =
  "history-consistent-raise-contract-margin-v1" as const;

const PLAYER_COUNT = 5;
const PLAYER_IDS = ["player-0", "player-1", "player-2", "player-3", "player-4"] as const;
const TARGETS = [13, 14, 15, 16, 17, 18, 19] as const;
const SUITS = BIDDING_HISTORY_SUIT_ORDER;
const CARD_BY_ID = new Map(createDeck().map((card) => [card.id, card]));

type PublicBiddingAction = Extract<GameAction, { type: "bid" | "pass" }>;

export interface GenerateHistoryConsistentRaiseMarginDatasetOptions {
  outputDirectory: string;
  pairCount: number;
  randomSeed: number;
  fixedHandCount?: number;
  candidateSeatIndex?: number;
  preferStrongHands?: boolean;
  handPoolMultiplier?: number;
  maxDealSeedsPerHand?: number;
  maxSamplesPerFixedHand?: number;
  actionCountPerState?: number;
  gamesPerShard?: number;
  sourceCommit?: string | null;
  onProgress?: (progress: {
    completedHands: number;
    dealSeedsTried: number;
    sourceStateCount: number;
    sampleCount: number;
  }) => void;
}

export interface HistoryConsistentRaiseMarginSample
  extends Omit<
    FixedHandBiddingMarginSample,
    "sampleType" | "schemaVersion" | "finalRoleCounts"
  > {
  sampleType: typeof HISTORY_CONSISTENT_RAISE_MARGIN_DATASET_SAMPLE_TYPE;
  schemaVersion: typeof HISTORY_CONSISTENT_RAISE_MARGIN_SAMPLE_SCHEMA_VERSION;
  dealSeed: number;
  sourceStateKey: string;
  sourceBiddingStep: number;
  biddingHistorySummary: readonly PublicBiddingHistoryRow[];
  currentHighestBid: {
    playerId: PlayerId;
    targetPointCards: number;
    suit: Suit;
  };
  currentBidder: PlayerId;
  consecutivePassCount: number;
  evaluatedRaiseAction: {
    actionIndex: number;
    targetPointCards: number;
    suit: Suit;
  };
  hiddenDealChecksum: string;
  rawRolloutShard: string;
  invariantChecks: {
    candidateHandFixed: true;
    deckConservation: true;
    candidateTurnWithCurrentBid: true;
    evaluatedActionLegalInSourceState: true;
    hiddenDealMatchesSourceState: true;
    candidateRoleNapoleon: true;
    contractOwnerCandidate: true;
    targetMatches: true;
    suitMatches: true;
    downstreamBiddingActionCount: 0;
  };
  finalRoleCounts: Record<"napoleon" | "napoleon-adjutant", number>;
}

export interface HistoryConsistentRaiseRawRollout {
  sampleType: "history-consistent-raise-margin-raw-rollout";
  schemaVersion: 1;
  fixedHandId: string;
  dealSeed: number;
  sourceStateKey: string;
  candidateSeatIndex: number;
  candidatePlayerId: PlayerId;
  forcedActionIndex: number;
  forcedTargetPointCards: number;
  forcedSuit: Suit;
  hiddenDealChecksum: string;
  biddingHistorySummary: readonly PublicBiddingHistoryRow[];
  currentHighestBid: {
    playerId: PlayerId;
    targetPointCards: number;
    suit: Suit;
  };
  consecutivePassCount: number;
  napoleonSidePointCards: number;
  coalitionSidePointCards: number;
  contractMargin: number;
  contractSuccess: boolean;
  candidateRelativeReward: number;
  resultType: GameResult["resultType"];
  finalRole: "napoleon" | "napoleon-adjutant";
  invariantChecks: HistoryConsistentRaiseMarginSample["invariantChecks"];
}

export interface HistoryConsistentRaiseMarginDatasetManifest {
  datasetSchemaVersion: typeof HISTORY_CONSISTENT_RAISE_MARGIN_DATASET_SCHEMA_VERSION;
  generatorVersion: typeof HISTORY_CONSISTENT_RAISE_MARGIN_DATASET_GENERATOR_VERSION;
  format: typeof DATASET_FORMAT;
  sampleType: typeof HISTORY_CONSISTENT_RAISE_MARGIN_DATASET_SAMPLE_TYPE;
  sampleSchemaVersion: typeof HISTORY_CONSISTENT_RAISE_MARGIN_SAMPLE_SCHEMA_VERSION;
  teacher: {
    id: typeof HISTORY_CONSISTENT_RAISE_MARGIN_TEACHER_ID;
    primaryLabel: "empiricalMarginMean";
    stdLabel: "empiricalMarginStd";
    winRateLabel: "empiricalWinRate";
    repeatsPerPair: 1;
    note: string;
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
    candidateBiddingPolicy: "same-seat-policy-mix-during-source-bidding";
  };
  pairCount: number;
  fixedHandCount: number;
  uniqueDealSeedCount: number;
  uniqueRaiseStateCount: number;
  rolloutCount: number;
  randomSeed: number;
  sourceCommit: string | null;
  playerCount: 5;
  cardCount: typeof CARD_COUNT;
  cardIds: readonly string[];
  cardIdsSha256: string;
  summary: FixedHandBiddingMarginDatasetSummary & {
    uniqueDealSeedCount: number;
    uniqueRaiseStateCount: number;
    currentTargetCounts: Record<string, number>;
    currentSuitCounts: Record<Suit, number>;
  };
  shardCount: number;
  shards: readonly DatasetShardManifest[];
  rawRolloutShards: readonly DatasetShardManifest[];
}

interface PublicBiddingHistoryRow {
  step: number;
  playerId: PlayerId;
  action: PublicBiddingAction;
}

interface SourceRaiseState {
  spec: FixedHandSpec;
  dealSeed: number;
  state: GameState;
  sourceStateKey: string;
  sourceBiddingStep: number;
  candidatePlayerId: PlayerId;
  modelInput: readonly number[];
  legalBidMask: readonly number[];
  biddingHistorySummary: readonly PublicBiddingHistoryRow[];
  hiddenDealChecksum: string;
  actions: readonly FixedHandBiddingActionSpec[];
}

interface RaiseRolloutRow {
  sample: HistoryConsistentRaiseMarginSample;
  raw: HistoryConsistentRaiseRawRollout;
}

export async function generateHistoryConsistentRaiseMarginDataset(
  options: GenerateHistoryConsistentRaiseMarginDatasetOptions
): Promise<{
  outputDirectory: string;
  manifest: HistoryConsistentRaiseMarginDatasetManifest;
  samples: readonly HistoryConsistentRaiseMarginSample[];
  rawRollouts: readonly HistoryConsistentRaiseRawRollout[];
}> {
  validateGenerateHistoryConsistentRaiseMarginDatasetOptions(options);
  const outputDirectory = resolve(options.outputDirectory);
  const tempDirectory = await mkdtemp(join(dirname(outputDirectory), `.${basename(outputDirectory)}.tmp-`));
  try {
    const samples: HistoryConsistentRaiseMarginSample[] = [];
    const rawRollouts: HistoryConsistentRaiseRawRollout[] = [];
    let dealSeedsTried = 0;
    let sourceStateCount = 0;
    const fixedHandCount = options.fixedHandCount ?? Math.max(20, Math.ceil(options.pairCount / 4));
    const maxSamplesPerFixedHand =
      options.maxSamplesPerFixedHand ?? Math.max(1, Math.ceil(options.pairCount / fixedHandCount));
    const fixedHands = createRaiseFixedHands({
      fixedHandCount,
      randomSeed: options.randomSeed,
      candidateSeatIndex: options.candidateSeatIndex,
      preferStrongHands: options.preferStrongHands ?? false,
      handPoolMultiplier: options.handPoolMultiplier ?? 20
    });
    for (const spec of fixedHands) {
      let samplesForHand = 0;
      for (let dealIndex = 0; dealIndex < (options.maxDealSeedsPerHand ?? 80); dealIndex += 1) {
        if (samples.length >= options.pairCount || samplesForHand >= maxSamplesPerFixedHand) {
          break;
        }
        const dealSeed = stableUint32([
          options.randomSeed,
          "history-consistent-raise",
          spec.fixedHandId,
          dealIndex
        ].join(":"));
        dealSeedsTried += 1;
        const sourceStates = await collectRaiseStatesForDeal({
          spec,
          dealSeed,
          actionCountPerState: options.actionCountPerState ?? 4
        });
        sourceStateCount += sourceStates.length;
        for (const source of sourceStates) {
          for (const action of source.actions) {
            if (samples.length >= options.pairCount || samplesForHand >= maxSamplesPerFixedHand) {
              break;
            }
            const row = await runHistoryConsistentRaiseRollout({ source, action });
            samples.push({ ...row.sample, rawRolloutShard: "" });
            rawRollouts.push(row.raw);
            samplesForHand += 1;
          }
        }
      }
      options.onProgress?.({
        completedHands: fixedHands.indexOf(spec) + 1,
        dealSeedsTried,
        sourceStateCount,
        sampleCount: samples.length
      });
      if (samples.length >= options.pairCount) {
        break;
      }
    }
    if (samples.length < options.pairCount) {
      throw new Error(
        `only collected ${samples.length} raise samples; increase fixedHandCount or maxDealSeedsPerHand.`
      );
    }
    const rawShards = await writeJsonlShards(
      tempDirectory,
      rawRollouts,
      options.gamesPerShard ?? 1000,
      "raw-rollouts"
    );
    const rowsPerShard = options.gamesPerShard ?? 1000;
    const samplesWithRawShard = samples.map((sample, index) => ({
      ...sample,
      rawRolloutShard: rawRolloutShardForIndex(index, rowsPerShard)
    }));
    const shards = await writeJsonlShards(
      tempDirectory,
      samplesWithRawShard,
      options.gamesPerShard ?? 1000,
      "shard"
    );
    const summary = summarizeHistoryConsistentRaiseSamples(samplesWithRawShard);
    const manifest = createHistoryConsistentRaiseManifest({
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

export function validateGenerateHistoryConsistentRaiseMarginDatasetOptions(
  options: GenerateHistoryConsistentRaiseMarginDatasetOptions
): void {
  if (!options.outputDirectory) {
    throw new Error("outputDirectory is required.");
  }
  validatePositiveInteger("pairCount", options.pairCount);
  validateUint32("randomSeed", options.randomSeed);
  if (options.fixedHandCount !== undefined) validatePositiveInteger("fixedHandCount", options.fixedHandCount);
  if (options.candidateSeatIndex !== undefined) validateSeat(options.candidateSeatIndex);
  if (options.handPoolMultiplier !== undefined) {
    validatePositiveInteger("handPoolMultiplier", options.handPoolMultiplier);
  }
  if (options.maxDealSeedsPerHand !== undefined) {
    validatePositiveInteger("maxDealSeedsPerHand", options.maxDealSeedsPerHand);
  }
  if (options.maxSamplesPerFixedHand !== undefined) {
    validatePositiveInteger("maxSamplesPerFixedHand", options.maxSamplesPerFixedHand);
  }
  if (options.actionCountPerState !== undefined) {
    validatePositiveInteger("actionCountPerState", options.actionCountPerState);
  }
  if (options.gamesPerShard !== undefined) validatePositiveInteger("gamesPerShard", options.gamesPerShard);
}

export function createHistoryConsistentRaiseInitialState(options: {
  dealSeed: number;
  candidateSeatIndex: number;
  handIds: readonly string[];
}): GameState {
  validateSeat(options.candidateSeatIndex);
  validateFixedHandIds(options.handIds);
  const state = createInitialGame({
    playerIds: PLAYER_IDS as unknown as readonly PlayerId[],
    rng: createSeededRandom(deriveSeed(options.dealSeed, "true-initial-placeholder"))
  });
  const fixed = options.handIds.map((id) => cardById(id));
  const fixedSet = new Set(options.handIds);
  const remaining = createDeck().filter((card) => !fixedSet.has(card.id));
  shuffleInPlace(
    remaining,
    createSeededRandom(deriveSeed(options.dealSeed, "history-consistent-remaining-deck"))
  );
  let offset = 0;
  const players = state.players.map((player, playerIndex) => {
    if (playerIndex === options.candidateSeatIndex) {
      return { ...player, hand: fixed };
    }
    const hand = remaining.slice(offset, offset + 10);
    offset += 10;
    return { ...player, hand };
  });
  return {
    ...state,
    players,
    unusedCards: remaining.slice(offset, offset + 3)
  };
}

export function summarizeHistoryConsistentRaiseSamples(
  samples: readonly HistoryConsistentRaiseMarginSample[]
): HistoryConsistentRaiseMarginDatasetManifest["summary"] {
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
    splitHintCounts: countBy(samples, (sample) => sample.splitHint ?? "none"),
    uniqueDealSeedCount: new Set(samples.map((sample) => sample.dealSeed)).size,
    uniqueRaiseStateCount: new Set(samples.map((sample) => sample.sourceStateKey)).size,
    currentTargetCounts: countBy(samples, (sample) => String(sample.currentHighestBid.targetPointCards)),
    currentSuitCounts: suitCounts(samples, (sample) => sample.currentHighestBid.suit)
  };
}

async function collectRaiseStatesForDeal(options: {
  spec: FixedHandSpec;
  dealSeed: number;
  actionCountPerState: number;
}): Promise<readonly SourceRaiseState[]> {
  const candidatePlayerId = PLAYER_IDS[options.spec.candidateSeatIndex];
  let state = createHistoryConsistentRaiseInitialState({
    dealSeed: options.dealSeed,
    candidateSeatIndex: options.spec.candidateSeatIndex,
    handIds: options.spec.handIds
  });
  assertDeckConservation(state);
  const publicActionHistory: PublicBiddingHistoryRow[] = [];
  const agents = new Map<PlayerId, Agent>(PLAYER_IDS.map((playerId, playerIndex) => [
    playerId,
    createSourceBiddingAgent({
      sourceSeed: options.dealSeed,
      candidateSeatIndex: options.spec.candidateSeatIndex,
      playerIndex,
      rngSeed: deriveSeed(options.dealSeed, `source-agent:${playerIndex}`)
    })
  ]));
  const result: SourceRaiseState[] = [];
  let step = 0;
  while (state.phase === "bidding" && !state.isGameOver) {
    if (step > 200) {
      throw new Error("source bidding decision limit exceeded.");
    }
    const playerId = state.currentPlayerId;
    const observation = createObservation(state, playerId, publicActionHistory);
    if (
      playerId === candidatePlayerId &&
      state.bidding?.highestBid !== null &&
      state.bidding?.highestBid !== undefined
    ) {
      const encoded = encodeBiddingObservation(
        observation,
        observation.view.players.map((player) => player.id)
      );
      const { modelInput, legalBidMask } = createBiddingModelInput(encoded);
      const actions = selectRaiseActions({
        legalBidMask,
        actionCount: options.actionCountPerState,
        strongestSuit: options.spec.strongestSuit,
        currentHighestBid: state.bidding.highestBid
      });
      if (actions.length > 0) {
        result.push({
          spec: options.spec,
          dealSeed: options.dealSeed,
          state,
          sourceStateKey: sourceStateKey({
            spec: options.spec,
            dealSeed: options.dealSeed,
            step,
            biddingHistorySummary: publicActionHistory,
            hiddenDealChecksum: hiddenDealChecksum(state)
          }),
          sourceBiddingStep: step,
          candidatePlayerId,
          modelInput: Array.from(modelInput),
          legalBidMask: Array.from(legalBidMask),
          biddingHistorySummary: publicActionHistory.map((row) => ({ ...row })),
          hiddenDealChecksum: hiddenDealChecksum(state),
          actions
        });
      }
    }
    const agent = agents.get(playerId);
    if (agent === undefined) throw new Error(`missing source agent for ${playerId}.`);
    const action = await agent.selectAction(observation);
    if (!observation.legalActions.some((legal) => actionsEqual(legal, action))) {
      throw new Error(`illegal source bidding action selected: ${JSON.stringify(action)}`);
    }
    step += 1;
    if (action.type === "bid" || action.type === "pass") {
      publicActionHistory.push({ step, playerId, action });
    }
    state = applyAction(clearLatestEvent(state), action);
  }
  return result;
}

async function runHistoryConsistentRaiseRollout(options: {
  source: SourceRaiseState;
  action: FixedHandBiddingActionSpec;
}): Promise<RaiseRolloutRow> {
  const { source, action } = options;
  if (source.state.bidding?.highestBid === null || source.state.bidding?.highestBid === undefined) {
    throw new Error("raise rollout source must have a current highest bid.");
  }
  if (Number(source.legalBidMask[action.actionIndex]) !== 1) {
    throw new Error("evaluated raise action is not legal in source state.");
  }
  const beforeChecksum = hiddenDealChecksum(source.state);
  if (beforeChecksum !== source.hiddenDealChecksum) {
    throw new Error("source hidden deal checksum changed before teacher rollout.");
  }
  let state = createContractEstablishedState(source.state, {
    napoleonPlayerId: source.candidatePlayerId,
    trumpSuit: action.suit,
    targetPointCards: action.targetPointCards
  });
  assertContractEstablishedState({
    state,
    candidatePlayerId: source.candidatePlayerId,
    targetPointCards: action.targetPointCards,
    suit: action.suit,
    handIds: source.spec.handIds
  });
  const agents = new Map<PlayerId, Agent>(PLAYER_IDS.map((playerId, playerIndex) => [
    playerId,
    new RuleBasedAgent(createSeededRandom(deriveSeed(source.dealSeed, `raise-teacher-agent:${playerIndex}`)))
  ]));
  let downstreamBiddingActionCount = 0;
  let step = 0;
  while (!state.isGameOver) {
    if (state.isTrickComplete) {
      state = advanceToNextTrick(clearLatestEvent(state));
      continue;
    }
    if (step > 1000) {
      throw new Error("teacher rollout decision limit exceeded.");
    }
    const playerId = state.currentPlayerId;
    const observation = createObservation(state, playerId, []);
    if (state.phase === "bidding") downstreamBiddingActionCount += 1;
    const agent = agents.get(playerId);
    if (agent === undefined) throw new Error(`missing teacher agent for ${playerId}.`);
    const selected = await agent.selectAction(observation);
    if (!observation.legalActions.some((legal) => actionsEqual(legal, selected))) {
      throw new Error(`illegal teacher action selected: ${JSON.stringify(selected)}`);
    }
    step += 1;
    state = applyAction(clearLatestEvent(state), selected);
  }
  if (state.result === null || state.result.resultType !== "standard") {
    throw new Error("history-consistent raise rollout did not produce a standard result.");
  }
  if (downstreamBiddingActionCount !== 0) {
    throw new Error(`teacher rollout executed ${downstreamBiddingActionCount} downstream bidding actions.`);
  }
  assertFinalDeckConservation(state);
  const finalRole = terminalRoleForResult(state.result, source.candidatePlayerId);
  if (finalRole !== "napoleon" && finalRole !== "napoleon-adjutant") {
    throw new Error(`candidate final role is not Napoleon: ${finalRole}.`);
  }
  const contractMargin = state.result.napoleonTeamPointCards - state.result.targetPointCards;
  const contractSuccess = state.result.winner === "napoleon-team";
  const invariantChecks = {
    candidateHandFixed: true as const,
    deckConservation: true as const,
    candidateTurnWithCurrentBid: true as const,
    evaluatedActionLegalInSourceState: true as const,
    hiddenDealMatchesSourceState: true as const,
    candidateRoleNapoleon: true as const,
    contractOwnerCandidate: true as const,
    targetMatches: true as const,
    suitMatches: true as const,
    downstreamBiddingActionCount: 0 as const
  };
  const raw: HistoryConsistentRaiseRawRollout = {
    sampleType: "history-consistent-raise-margin-raw-rollout",
    schemaVersion: 1,
    fixedHandId: source.spec.fixedHandId,
    dealSeed: source.dealSeed,
    sourceStateKey: source.sourceStateKey,
    candidateSeatIndex: source.spec.candidateSeatIndex,
    candidatePlayerId: source.candidatePlayerId,
    forcedActionIndex: action.actionIndex,
    forcedTargetPointCards: action.targetPointCards,
    forcedSuit: action.suit,
    hiddenDealChecksum: source.hiddenDealChecksum,
    biddingHistorySummary: source.biddingHistorySummary,
    currentHighestBid: source.state.bidding.highestBid,
    consecutivePassCount: source.state.bidding.consecutivePassCount,
    napoleonSidePointCards: state.result.napoleonTeamPointCards,
    coalitionSidePointCards: state.result.alliancePointCards,
    contractMargin,
    contractSuccess,
    candidateRelativeReward: contractSuccess
      ? (7 * action.targetPointCards) / 4
      : (-3 * action.targetPointCards) / 4,
    resultType: state.result.resultType,
    finalRole,
    invariantChecks
  };
  const sample: HistoryConsistentRaiseMarginSample = {
    sampleType: HISTORY_CONSISTENT_RAISE_MARGIN_DATASET_SAMPLE_TYPE,
    schemaVersion: HISTORY_CONSISTENT_RAISE_MARGIN_SAMPLE_SCHEMA_VERSION,
    fixedHandId: source.spec.fixedHandId,
    handIds: source.spec.handIds,
    candidateSeatIndex: source.spec.candidateSeatIndex,
    candidatePlayerId: source.candidatePlayerId,
    sourceStateKey: source.sourceStateKey,
    sourceSeed: source.dealSeed,
    sourceBiddingStep: source.sourceBiddingStep,
    strongestSuit: source.spec.strongestSuit,
    strongestSuitScore: source.spec.strongestSuitScore,
    forcedActionIndex: action.actionIndex,
    forcedTargetPointCards: action.targetPointCards,
    forcedSuit: action.suit,
    forcedActionLabel: action.label ?? `${action.suit}${action.targetPointCards}`,
    modelInput: source.modelInput,
    legalBidMask: source.legalBidMask,
    rolloutCount: 1,
    empiricalMarginMean: contractMargin,
    empiricalMarginStd: 0,
    empiricalWinRate: contractSuccess ? 1 : 0,
    empiricalMarginMin: contractMargin,
    empiricalMarginMax: contractMargin,
    marginHistogram: { [String(contractMargin)]: 1 },
    resultTypeCounts: { [state.result.resultType]: 1 },
    finalRoleCounts: { [finalRole]: 1 } as Record<"napoleon" | "napoleon-adjutant", number>,
    sourceNnMu: null,
    sourceNnSigma: null,
    sourceNnPWin: null,
    splitHint: null,
    dealSeed: source.dealSeed,
    biddingHistorySummary: source.biddingHistorySummary,
    currentHighestBid: source.state.bidding.highestBid,
    currentBidder: source.state.currentPlayerId,
    consecutivePassCount: source.state.bidding.consecutivePassCount,
    evaluatedRaiseAction: {
      actionIndex: action.actionIndex,
      targetPointCards: action.targetPointCards,
      suit: action.suit
    },
    hiddenDealChecksum: source.hiddenDealChecksum,
    rawRolloutShard: "",
    invariantChecks
  };
  return { sample, raw };
}

class BiddingOnlyMixedAgent implements Agent {
  readonly ruleBased: RuleBasedAgent;
  readonly conservative: ConservativeBiddingAgent;
  readonly biddingPolicyType: "rule-based" | "conservative-bidding";

  constructor(options: { rngSeed: number; biddingPolicyType: "rule-based" | "conservative-bidding" }) {
    const rng = createSeededRandom(options.rngSeed);
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

function createSourceBiddingAgent(options: {
  sourceSeed: number;
  candidateSeatIndex: number;
  playerIndex: number;
  rngSeed: number;
}): Agent {
  return new BiddingOnlyMixedAgent({
    rngSeed: options.rngSeed,
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

function selectRaiseActions(options: {
  legalBidMask: readonly number[];
  actionCount: number;
  strongestSuit: Suit;
  currentHighestBid: { targetPointCards: number; suit: Suit };
}): readonly FixedHandBiddingActionSpec[] {
  const legal: FixedHandBiddingActionSpec[] = [];
  for (let index = 1; index < BIDDING_ACTION_COUNT; index += 1) {
    if (Number(options.legalBidMask[index]) !== 1) continue;
    const decoded = decodeBiddingQActionIndex(index);
    if (decoded.type !== "bid") continue;
    const targetPointCards = decoded.targetPointCards;
    const suit = decoded.suit;
    if (targetPointCards === undefined || suit === undefined) continue;
    legal.push({
      actionIndex: index,
      targetPointCards,
      suit,
      label: `${suit}${targetPointCards}`
    });
  }
  const byIndex = new Map(legal.map((action) => [action.actionIndex, action]));
  const perSuitLowest = SUITS
    .map((suit) => legal.find((action) => action.suit === suit))
    .filter((action): action is FixedHandBiddingActionSpec => action !== undefined);
  const preferred = [
    ...perSuitLowest,
    ...legal.filter((action) => action.targetPointCards === options.currentHighestBid.targetPointCards),
    ...legal.filter((action) => action.targetPointCards === options.currentHighestBid.targetPointCards + 1),
    ...legal.filter((action) => action.suit === options.strongestSuit),
    ...legal
  ];
  const selected = new Map<number, FixedHandBiddingActionSpec>();
  for (const action of preferred) {
    if (byIndex.has(action.actionIndex)) selected.set(action.actionIndex, action);
    if (selected.size >= options.actionCount) break;
  }
  return [...selected.values()];
}

function createRaiseFixedHands(options: {
  fixedHandCount: number;
  randomSeed: number;
  candidateSeatIndex?: number;
  preferStrongHands: boolean;
  handPoolMultiplier: number;
}): readonly FixedHandSpec[] {
  const pool = createRandomFixedHands({
    handCount: options.preferStrongHands
      ? options.fixedHandCount * options.handPoolMultiplier
      : options.fixedHandCount,
    actionCountPerHand: 1,
    randomSeed: options.randomSeed,
    candidateSeatIndex: options.candidateSeatIndex ?? 0
  });
  const selected = options.preferStrongHands
    ? [...pool].sort((left, right) => right.strongestSuitScore - left.strongestSuitScore).slice(0, options.fixedHandCount)
    : pool;
  return selected.map((spec, index) => ({
    ...spec,
    fixedHandId: `raise:${options.preferStrongHands ? "strong" : "random"}:${index}:${spec.fixedHandId}`,
    actions: []
  }));
}

function createObservation(
  state: GameState,
  playerId: PlayerId,
  publicActionHistory: readonly PublicBiddingHistoryRow[]
): PlayerObservation {
  return {
    playerId,
    view: createPlayerView(state, playerId),
    legalActions: getAutomatedLegalActions(state, playerId),
    publicActionHistory: publicActionHistory.map(({ step, playerId: actorId, action }) => ({
      step,
      playerId: actorId,
      phase: "bidding",
      action
    }))
  };
}

function getAutomatedLegalActions(state: GameState, playerId: PlayerId): GameAction[] {
  if (state.phase !== "exchanging") return [...getLegalActions(state, playerId)];
  const view = createPlayerView(state, playerId);
  const discardCount = view.exchangeRequirement?.discardCount;
  const self = view.players.find((player) => player.id === playerId);
  if (discardCount === undefined || self?.hand === undefined) return [];
  return combinations(self.hand, discardCount).map((cards) => ({
    type: "discard-cards",
    playerId,
    cardIds: cards.map((card) => card.id)
  }));
}

function assertDeckConservation(state: GameState): void {
  const ids = [
    ...state.players.flatMap((player) => player.hand.map((card) => card.id)),
    ...state.unusedCards.map((card) => card.id)
  ];
  if (ids.length !== CARD_COUNT || new Set(ids).size !== CARD_COUNT) {
    throw new Error("source state does not conserve the deck.");
  }
}

function assertFinalDeckConservation(state: GameState): void {
  const ids = [
    ...state.players.flatMap((player) => player.hand.map((card) => card.id)),
    ...state.unusedCards.map((card) => card.id),
    ...state.excludedCards.map((card) => card.id),
    ...state.awardedPointCards.flatMap((award) => award.cards.map((card) => card.id)),
    ...state.completedTricks.flatMap((trick) => trick.cards.map((played) => played.card.id))
  ];
  if (ids.length !== CARD_COUNT || new Set(ids).size !== CARD_COUNT) {
    throw new Error("terminal state does not conserve the deck.");
  }
}

function assertContractEstablishedState(options: {
  state: GameState;
  candidatePlayerId: PlayerId;
  targetPointCards: number;
  suit: Suit;
  handIds: readonly string[];
}): void {
  if (options.state.phase !== "choosing-adjutant" || options.state.bidding !== null) {
    throw new Error("teacher rollout must start after contract establishment.");
  }
  if (options.state.contract?.napoleonPlayerId !== options.candidatePlayerId) {
    throw new Error("contract owner is not candidate.");
  }
  if (
    options.state.contract.targetPointCards !== options.targetPointCards ||
    options.state.contract.trumpSuit !== options.suit
  ) {
    throw new Error("contract target/suit does not match evaluated raise.");
  }
  if (!sameStringArray(handIdsForPlayer(options.state, options.candidatePlayerId), [...options.handIds].sort())) {
    throw new Error("candidate hand changed before teacher rollout.");
  }
}

function hiddenDealChecksum(state: GameState): string {
  const payload = {
    players: state.players.map((player) => ({
      id: player.id,
      handIds: player.hand.map((card) => card.id).sort()
    })),
    unusedCardIds: state.unusedCards.map((card) => card.id).sort()
  };
  return createHash("sha256").update(JSON.stringify(payload)).digest("hex");
}

function sourceStateKey(options: {
  spec: FixedHandSpec;
  dealSeed: number;
  step: number;
  biddingHistorySummary: readonly PublicBiddingHistoryRow[];
  hiddenDealChecksum: string;
}): string {
  return createHash("sha256")
    .update(JSON.stringify({
      fixedHandId: options.spec.fixedHandId,
      dealSeed: options.dealSeed,
      step: options.step,
      history: options.biddingHistorySummary,
      hiddenDealChecksum: options.hiddenDealChecksum
    }))
    .digest("hex")
    .slice(0, 20);
}

function createHistoryConsistentRaiseManifest(options: {
  options: GenerateHistoryConsistentRaiseMarginDatasetOptions;
  samples: readonly HistoryConsistentRaiseMarginSample[];
  shards: readonly DatasetShardManifest[];
  rawShards: readonly DatasetShardManifest[];
  summary: HistoryConsistentRaiseMarginDatasetManifest["summary"];
}): HistoryConsistentRaiseMarginDatasetManifest {
  return {
    datasetSchemaVersion: HISTORY_CONSISTENT_RAISE_MARGIN_DATASET_SCHEMA_VERSION,
    generatorVersion: HISTORY_CONSISTENT_RAISE_MARGIN_DATASET_GENERATOR_VERSION,
    format: DATASET_FORMAT,
    sampleType: HISTORY_CONSISTENT_RAISE_MARGIN_DATASET_SAMPLE_TYPE,
    sampleSchemaVersion: HISTORY_CONSISTENT_RAISE_MARGIN_SAMPLE_SCHEMA_VERSION,
    teacher: {
      id: HISTORY_CONSISTENT_RAISE_MARGIN_TEACHER_ID,
      primaryLabel: "empiricalMarginMean",
      stdLabel: "empiricalMarginStd",
      winRateLabel: "empiricalWinRate",
      repeatsPerPair: 1,
      note: "Each label uses the same hidden deal that generated the visible bidding history; no post-history hidden-hand reshuffle is performed."
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
        "dealSeed hidden deal",
        "visible bidding history generated from the same deal",
        "candidate role = Napoleon during teacher",
        "contract owner = candidate during teacher",
        "contract target/suit = evaluated raise"
      ],
      varied: ["fixedHandId", "dealSeed", "source raise state", "evaluated legal raise action"],
      note: "Source bidding starts from the true initial bidding state and runs RuleBased/Conservative 50:50 until candidate turns with a current bid are collected."
    },
    opponentMix: {
      mixingRuleVersion: BIDDING_Q_COUNTERFACTUAL_OPPONENT_MIX_RULE_VERSION,
      ruleBasedWeight: 0.5,
      conservativeWeight: 0.5,
      candidateBiddingPolicy: "same-seat-policy-mix-during-source-bidding"
    },
    pairCount: options.samples.length,
    fixedHandCount: new Set(options.samples.map((sample) => sample.fixedHandId)).size,
    uniqueDealSeedCount: options.summary.uniqueDealSeedCount,
    uniqueRaiseStateCount: options.summary.uniqueRaiseStateCount,
    rolloutCount: options.samples.length,
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

function rawRolloutShardForIndex(index: number, rowsPerShard: number): string {
  return `raw-rollouts-${String(Math.floor(index / rowsPerShard)).padStart(5, "0")}.jsonl`;
}

function handIdsForPlayer(state: GameState, playerId: PlayerId): string[] {
  const player = state.players.find((candidate) => candidate.id === playerId);
  if (player === undefined) throw new Error(`missing player ${playerId}.`);
  return player.hand.map((card) => card.id).sort();
}

function terminalRoleForResult(result: GameResult, actingPlayerId: PlayerId): string {
  if (result.resultType !== "standard") return "all-pass-other";
  if (result.napoleonPlayerId === actingPlayerId && result.adjutantPlayerId === actingPlayerId) {
    return "napoleon-adjutant";
  }
  if (result.napoleonPlayerId === actingPlayerId) return "napoleon";
  if (result.adjutantPlayerId === actingPlayerId) return "adjutant";
  return "citizen";
}

function actionsEqual(left: GameAction, right: GameAction): boolean {
  if (left.type !== right.type || left.playerId !== right.playerId) return false;
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

function validateFixedHandIds(handIds: readonly string[]): void {
  if (handIds.length !== 10) throw new Error("fixed hand must contain exactly 10 cards.");
  if (new Set(handIds).size !== handIds.length) throw new Error("fixed hand contains duplicate cards.");
  for (const id of handIds) cardById(id);
}

function validateSeat(value: number): void {
  if (!Number.isInteger(value) || value < 0 || value >= PLAYER_COUNT) {
    throw new Error("candidateSeatIndex must be an integer seat index.");
  }
}

function validatePositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer.`);
}

function validateUint32(name: string, value: number): void {
  if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
    throw new Error(`${name} must be a uint32 integer.`);
  }
}

function cardById(id: string): Card {
  const card = CARD_BY_ID.get(id);
  if (card === undefined) throw new Error(`unknown card ${id}.`);
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

function sameStringArray(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function countBy<T>(values: readonly T[], keyFn: (value: T) => string): Record<string, number> {
  const result: Record<string, number> = {};
  for (const value of values) {
    const key = keyFn(value);
    result[key] = (result[key] ?? 0) + 1;
  }
  return Object.fromEntries(Object.entries(result).sort(([left], [right]) => left.localeCompare(right, undefined, { numeric: true })));
}

function suitCounts<T>(values: readonly T[], keyFn: (value: T) => Suit): Record<Suit, number> {
  const result = { spades: 0, hearts: 0, diamonds: 0, clubs: 0 };
  for (const value of values) result[keyFn(value)] += 1;
  return result;
}

function numericSummary(values: readonly number[]): NumericSummary {
  if (values.length === 0) {
    return { count: 0, mean: null, std: null, min: null, max: null };
  }
  const average = sum(values) / values.length;
  return {
    count: values.length,
    mean: average,
    std: Math.sqrt(values.reduce((total, value) => total + (value - average) ** 2, 0) / values.length),
    min: Math.min(...values),
    max: Math.max(...values)
  };
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}
