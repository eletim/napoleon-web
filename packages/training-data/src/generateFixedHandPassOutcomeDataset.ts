import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { ConservativeBiddingAgent, RuleBasedAgent, createSeededRandom, deriveSeed } from "@napoleon/ai";
import type { Agent, PlayerObservation } from "@napoleon/ai";
import {
  advanceToNextTrick,
  applyAction,
  clearLatestEvent,
  createPlayerView,
  getLegalActions
} from "@napoleon/game-core";
import type { GameAction, GameResult, PlayerId, Suit } from "@napoleon/game-core";
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
  BIDDING_Q_COUNTERFACTUAL_OPPONENT_MIX_RULE_VERSION
} from "./generateBiddingQCounterfactualDataset.js";
import {
  createFixedHandInitialState,
  createRandomFixedHands,
  stableUint32
} from "./generateFixedHandBiddingMarginDataset.js";
import type { FixedHandSpec, NumericSummary } from "./generateFixedHandBiddingMarginDataset.js";

export const FIXED_HAND_PASS_OUTCOME_DATASET_SAMPLE_TYPE =
  "fixed-hand-pass-outcome-sample" as const;
export const FIXED_HAND_PASS_OUTCOME_SAMPLE_SCHEMA_VERSION = 1 as const;
export const FIXED_HAND_PASS_OUTCOME_DATASET_SCHEMA_VERSION = 1 as const;
export const FIXED_HAND_PASS_OUTCOME_DATASET_GENERATOR_VERSION = 1 as const;
export const FIXED_HAND_PASS_OUTCOME_TEACHER_ID =
  "fixed-hand-pass-hidden-deal-role-probability-role-margin-v1" as const;

const PLAYER_IDS = ["player-0", "player-1", "player-2", "player-3", "player-4"] as const;
const PLAYER_COUNT = 5;

export interface GenerateFixedHandPassOutcomeDatasetOptions {
  outputDirectory: string;
  handCount: number;
  repeats: number;
  randomSeed: number;
  candidateSeatIndex?: number;
  gamesPerShard?: number;
  reservedHands?: readonly FixedHandSpec[];
  reserveHandsForFinal?: boolean;
  sourceCommit?: string | null;
  onProgress?: (progress: {
    totalHands: number;
    completedHands: number;
    totalRollouts: number;
    completedRollouts: number;
    sampleCount: number;
  }) => void;
}

export interface RoleConditionedMarginSummary {
  count: number;
  empiricalMarginMean: number | null;
  empiricalMarginStd: number | null;
  empiricalWinRate: number | null;
  empiricalTargetMean: number | null;
  marginMin: number | null;
  marginMax: number | null;
}

export interface FixedHandPassOutcomeSample {
  sampleType: typeof FIXED_HAND_PASS_OUTCOME_DATASET_SAMPLE_TYPE;
  schemaVersion: typeof FIXED_HAND_PASS_OUTCOME_SAMPLE_SCHEMA_VERSION;
  fixedHandId: string;
  handIds: readonly string[];
  candidateSeatIndex: number;
  candidatePlayerId: PlayerId;
  sourceStateKey: string | null;
  sourceSeed: number | null;
  sourceBiddingStep: number | null;
  strongestSuit: Suit;
  strongestSuitScore: number;
  modelInput: readonly number[];
  legalBidMask: readonly number[];
  rolloutCount: number;
  nCitizen: number;
  nAdjutant: number;
  nNoContract: number;
  nNapoleonAfterPass: number;
  pCitizenEmpirical: number;
  pAdjutantEmpirical: number;
  pNoContractEmpirical: number;
  qTeacher: number | null;
  qTeacherDenominator: number;
  citizenMargin: RoleConditionedMarginSummary;
  adjutantMargin: RoleConditionedMarginSummary;
  resultTypeCounts: Record<string, number>;
  finalRoleCounts: Record<string, number>;
  splitHint: "final-diagnostic" | null;
}

export interface FixedHandPassOutcomeDatasetManifest {
  datasetSchemaVersion: typeof FIXED_HAND_PASS_OUTCOME_DATASET_SCHEMA_VERSION;
  generatorVersion: typeof FIXED_HAND_PASS_OUTCOME_DATASET_GENERATOR_VERSION;
  format: typeof DATASET_FORMAT;
  sampleType: typeof FIXED_HAND_PASS_OUTCOME_DATASET_SAMPLE_TYPE;
  sampleSchemaVersion: typeof FIXED_HAND_PASS_OUTCOME_SAMPLE_SCHEMA_VERSION;
  teacher: {
    id: typeof FIXED_HAND_PASS_OUTCOME_TEACHER_ID;
    qLabel: "qTeacher";
    citizenMeanLabel: "citizenMargin.empiricalMarginMean";
    adjutantMeanLabel: "adjutantMargin.empiricalMarginMean";
    repeatsPerHand: number;
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
  handCount: number;
  rolloutCount: number;
  randomSeed: number;
  sourceCommit: string | null;
  playerCount: 5;
  cardCount: typeof CARD_COUNT;
  cardIds: readonly string[];
  cardIdsSha256: string;
  summary: FixedHandPassOutcomeDatasetSummary;
  shardCount: number;
  shards: readonly DatasetShardManifest[];
}

export interface FixedHandPassOutcomeDatasetSummary {
  handCount: number;
  rolloutCount: number;
  repeatCountMin: number;
  repeatCountMax: number;
  qTeacher: NumericSummary;
  pNoContract: NumericSummary;
  citizenRoleCount: NumericSummary;
  adjutantRoleCount: NumericSummary;
  citizenMarginMean: NumericSummary;
  adjutantMarginMean: NumericSummary;
  strongestSuitCounts: Record<Suit, number>;
  splitHintCounts: Record<string, number>;
}

interface PassRolloutRow {
  contractMargin: number | null;
  contractSuccess: boolean | null;
  finalDeclaredTarget: number | null;
  resultType: GameResult["resultType"];
  finalRole: string;
  modelInput: readonly number[];
  legalBidMask: readonly number[];
}

type ForcedActionWithMetadata = GameAction & {
  __forced?: true;
  __modelInput?: readonly number[];
  __legalBidMask?: readonly number[];
};
type PublicBiddingAction = Extract<GameAction, { type: "bid" | "pass" }>;

export async function generateFixedHandPassOutcomeDataset(
  options: GenerateFixedHandPassOutcomeDatasetOptions
): Promise<{
  outputDirectory: string;
  manifest: FixedHandPassOutcomeDatasetManifest;
  samples: readonly FixedHandPassOutcomeSample[];
}> {
  validateGenerateFixedHandPassOutcomeDatasetOptions(options);
  const outputDirectory = resolve(options.outputDirectory);
  const tempDirectory = await mkdtemp(
    join(dirname(outputDirectory), `.${basename(outputDirectory)}.tmp-`)
  );
  try {
    const specs = createFixedHandPassPlan(options);
    const samples: FixedHandPassOutcomeSample[] = [];
    let completedRollouts = 0;
    for (const [index, spec] of specs.entries()) {
      const rows: PassRolloutRow[] = [];
      for (let repeatIndex = 0; repeatIndex < options.repeats; repeatIndex += 1) {
        const reshuffleSeed = stableUint32([
          options.randomSeed,
          "pass",
          spec.fixedHandId,
          repeatIndex
        ].join(":"));
        rows.push(await runFixedHandPassRollout({ spec, reshuffleSeed }));
        completedRollouts += 1;
      }
      samples.push(aggregateFixedHandPassRollouts({
        spec,
        rows,
        splitHint: options.reserveHandsForFinal && spec.reason?.includes("issue409")
          ? "final-diagnostic"
          : null
      }));
      options.onProgress?.({
        totalHands: specs.length,
        completedHands: index + 1,
        totalRollouts: specs.length * options.repeats,
        completedRollouts,
        sampleCount: samples.length
      });
    }
    const shards = await writeSampleShards(tempDirectory, samples, options.gamesPerShard ?? 1000);
    const summary = summarizeFixedHandPassOutcomeSamples(samples);
    const manifest = createManifest({ options, samples, shards, summary });
    await writeFile(join(tempDirectory, "manifest.json"), serializeManifest(manifest), "utf8");
    await writeFile(join(tempDirectory, "summary.json"), serializeManifest(summary), "utf8");
    await rename(tempDirectory, outputDirectory);
    return { outputDirectory, manifest, samples };
  } catch (error) {
    await rm(tempDirectory, { recursive: true, force: true }).catch(() => undefined);
    throw error;
  }
}

export function createFixedHandPassPlan(
  options: GenerateFixedHandPassOutcomeDatasetOptions
): readonly FixedHandSpec[] {
  const reserved = [...(options.reservedHands ?? [])];
  const randomCount = Math.max(0, options.handCount - reserved.length);
  const randomHands = createRandomFixedHands({
    handCount: randomCount,
    actionCountPerHand: 1,
    randomSeed: options.randomSeed,
    candidateSeatIndex: options.candidateSeatIndex ?? 0
  });
  return [...reserved, ...randomHands].slice(0, options.handCount);
}

export function aggregateFixedHandPassRollouts(options: {
  spec: FixedHandSpec;
  rows: readonly PassRolloutRow[];
  splitHint: "final-diagnostic" | null;
}): FixedHandPassOutcomeSample {
  if (options.rows.length === 0) {
    throw new Error("cannot aggregate an empty PASS rollout group.");
  }
  const first = options.rows[0];
  const roleCounts = countBy(options.rows, (row) => row.finalRole);
  const citizenRows = options.rows.filter((row) => row.finalRole === "citizen");
  const adjutantRows = options.rows.filter((row) => row.finalRole === "adjutant");
  const nCitizen = roleCounts.citizen ?? 0;
  const nAdjutant = roleCounts.adjutant ?? 0;
  const denominator = nCitizen + nAdjutant;
  return {
    sampleType: FIXED_HAND_PASS_OUTCOME_DATASET_SAMPLE_TYPE,
    schemaVersion: FIXED_HAND_PASS_OUTCOME_SAMPLE_SCHEMA_VERSION,
    fixedHandId: options.spec.fixedHandId,
    handIds: options.spec.handIds,
    candidateSeatIndex: options.spec.candidateSeatIndex,
    candidatePlayerId: PLAYER_IDS[options.spec.candidateSeatIndex],
    sourceStateKey: options.spec.sourceStateKey ?? null,
    sourceSeed: options.spec.sourceSeed ?? null,
    sourceBiddingStep: options.spec.sourceBiddingStep ?? null,
    strongestSuit: options.spec.strongestSuit,
    strongestSuitScore: options.spec.strongestSuitScore,
    modelInput: first.modelInput,
    legalBidMask: first.legalBidMask,
    rolloutCount: options.rows.length,
    nCitizen,
    nAdjutant,
    nNoContract: (roleCounts["no-contract"] ?? 0) + (roleCounts["all-pass-starter"] ?? 0) + (roleCounts["all-pass-other"] ?? 0),
    nNapoleonAfterPass: (roleCounts.napoleon ?? 0) + (roleCounts["napoleon-adjutant"] ?? 0),
    pCitizenEmpirical: nCitizen / options.rows.length,
    pAdjutantEmpirical: nAdjutant / options.rows.length,
    pNoContractEmpirical: ((roleCounts["no-contract"] ?? 0) + (roleCounts["all-pass-starter"] ?? 0) + (roleCounts["all-pass-other"] ?? 0)) / options.rows.length,
    qTeacher: denominator === 0 ? null : nAdjutant / denominator,
    qTeacherDenominator: denominator,
    citizenMargin: summarizeRoleMargin(citizenRows),
    adjutantMargin: summarizeRoleMargin(adjutantRows),
    resultTypeCounts: countBy(options.rows, (row) => row.resultType),
    finalRoleCounts: roleCounts,
    splitHint: options.splitHint
  };
}

export function summarizeFixedHandPassOutcomeSamples(
  samples: readonly FixedHandPassOutcomeSample[]
): FixedHandPassOutcomeDatasetSummary {
  const repeats = samples.map((sample) => sample.rolloutCount);
  return {
    handCount: samples.length,
    rolloutCount: sum(repeats),
    repeatCountMin: repeats.length === 0 ? 0 : Math.min(...repeats),
    repeatCountMax: repeats.length === 0 ? 0 : Math.max(...repeats),
    qTeacher: numericSummary(samples.flatMap((sample) => sample.qTeacher === null ? [] : [sample.qTeacher])),
    pNoContract: numericSummary(samples.map((sample) => sample.pNoContractEmpirical)),
    citizenRoleCount: numericSummary(samples.map((sample) => sample.citizenMargin.count)),
    adjutantRoleCount: numericSummary(samples.map((sample) => sample.adjutantMargin.count)),
    citizenMarginMean: numericSummary(samples.flatMap((sample) => sample.citizenMargin.empiricalMarginMean === null ? [] : [sample.citizenMargin.empiricalMarginMean])),
    adjutantMarginMean: numericSummary(samples.flatMap((sample) => sample.adjutantMargin.empiricalMarginMean === null ? [] : [sample.adjutantMargin.empiricalMarginMean])),
    strongestSuitCounts: suitCounts(samples, (sample) => sample.strongestSuit),
    splitHintCounts: countBy(samples, (sample) => sample.splitHint ?? "none")
  };
}

export function validateGenerateFixedHandPassOutcomeDatasetOptions(
  options: GenerateFixedHandPassOutcomeDatasetOptions
): void {
  if (!options.outputDirectory) throw new Error("outputDirectory is required.");
  validatePositiveInteger("handCount", options.handCount);
  validatePositiveInteger("repeats", options.repeats);
  validateUint32("randomSeed", options.randomSeed);
  if (options.candidateSeatIndex !== undefined) {
    validateSeat(options.candidateSeatIndex);
  }
  if (options.gamesPerShard !== undefined) {
    validatePositiveInteger("gamesPerShard", options.gamesPerShard);
  }
}

async function runFixedHandPassRollout(options: {
  spec: FixedHandSpec;
  reshuffleSeed: number;
}): Promise<PassRolloutRow> {
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
      ? new ForcedPassCandidateAgent({ rngSeed: options.reshuffleSeed })
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
    if (step > 1000) throw new Error("decision limit exceeded.");
    const playerId = state.currentPlayerId;
    const observation = createObservation(state, playerId, publicActionHistory);
    const agent = agents.get(playerId);
    if (agent === undefined) throw new Error(`missing agent for ${playerId}.`);
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
      action.type === "pass" &&
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
    throw new Error("fixed-hand PASS rollout did not produce a forced terminal result.");
  }
  return {
    contractMargin: state.result.resultType === "standard"
      ? state.result.napoleonTeamPointCards - state.result.targetPointCards
      : null,
    contractSuccess: state.result.resultType === "standard"
      ? state.result.winner === "napoleon-team"
      : null,
    finalDeclaredTarget: state.result.resultType === "standard" ? state.result.targetPointCards : null,
    resultType: state.result.resultType,
    finalRole: terminalRoleForResult(state.result, candidatePlayerId),
    modelInput: forcedModelInput,
    legalBidMask: forcedLegalBidMask
  };
}

class ForcedPassCandidateAgent implements Agent {
  readonly ruleBased: RuleBasedAgent;
  forced = false;

  constructor(options: { rngSeed: number }) {
    this.ruleBased = new RuleBasedAgent(createSeededRandom(deriveSeed(options.rngSeed, "forced-pass")));
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    if (observation.view.phase === "bidding" && !this.forced) {
      const encoded = encodeBiddingObservation(
        observation,
        observation.view.players.map((player) => player.id)
      );
      const { modelInput, legalBidMask } = createBiddingModelInput(encoded);
      if (Number(legalBidMask[0]) !== 1) {
        throw new Error("forced PASS is illegal in fixed-hand context.");
      }
      const decoded = decodeBiddingAction(0, observation.playerId);
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
  state: ReturnType<typeof createFixedHandInitialState>,
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
  state: ReturnType<typeof createFixedHandInitialState>,
  playerId: PlayerId
): GameAction[] {
  if (state.phase !== "exchanging") {
    return [...getLegalActions(state, playerId)];
  }
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

function summarizeRoleMargin(rows: readonly PassRolloutRow[]): RoleConditionedMarginSummary {
  const margins = rows.flatMap((row) => row.contractMargin === null ? [] : [row.contractMargin]);
  const wins = rows.flatMap((row) => row.contractSuccess === null ? [] : [row.contractSuccess ? 1 : 0]);
  const targets = rows.flatMap((row) => row.finalDeclaredTarget === null ? [] : [row.finalDeclaredTarget]);
  return {
    count: rows.length,
    empiricalMarginMean: margins.length === 0 ? null : mean(margins),
    empiricalMarginStd: margins.length === 0 ? null : std(margins),
    empiricalWinRate: wins.length === 0 ? null : mean(wins),
    empiricalTargetMean: targets.length === 0 ? null : mean(targets),
    marginMin: margins.length === 0 ? null : Math.min(...margins),
    marginMax: margins.length === 0 ? null : Math.max(...margins)
  };
}

async function writeSampleShards(
  directory: string,
  samples: readonly FixedHandPassOutcomeSample[],
  gamesPerShard: number
): Promise<DatasetShardManifest[]> {
  await mkdir(directory, { recursive: true });
  const shards: DatasetShardManifest[] = [];
  for (let start = 0; start < samples.length; start += gamesPerShard) {
    const shardSamples = samples.slice(start, start + gamesPerShard);
    const file = `shard-${String(shards.length).padStart(5, "0")}.jsonl`;
    const body = shardSamples.map((sample) => `${JSON.stringify(sample)}\n`).join("");
    await writeFile(join(directory, file), body, "utf8");
    shards.push({
      file,
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

function createManifest(options: {
  options: GenerateFixedHandPassOutcomeDatasetOptions;
  samples: readonly FixedHandPassOutcomeSample[];
  shards: readonly DatasetShardManifest[];
  summary: FixedHandPassOutcomeDatasetSummary;
}): FixedHandPassOutcomeDatasetManifest {
  return {
    datasetSchemaVersion: FIXED_HAND_PASS_OUTCOME_DATASET_SCHEMA_VERSION,
    generatorVersion: FIXED_HAND_PASS_OUTCOME_DATASET_GENERATOR_VERSION,
    format: DATASET_FORMAT,
    sampleType: FIXED_HAND_PASS_OUTCOME_DATASET_SAMPLE_TYPE,
    sampleSchemaVersion: FIXED_HAND_PASS_OUTCOME_SAMPLE_SCHEMA_VERSION,
    teacher: {
      id: FIXED_HAND_PASS_OUTCOME_TEACHER_ID,
      qLabel: "qTeacher",
      citizenMeanLabel: "citizenMargin.empiricalMarginMean",
      adjutantMeanLabel: "adjutantMargin.empiricalMarginMean",
      repeatsPerHand: options.options.repeats
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
      suitOrder: BIDDING_HISTORY_SUIT_ORDER
    },
    fixedCondition: {
      fixed: ["candidate hand", "candidate seat index", "forced PASS"],
      varied: ["other four hands", "kitty cards", "downstream bidding", "adjutant choice", "exchange", "play"],
      note: "Candidate is forced to PASS as initial bidding actor; roles are observed, not forced."
    },
    opponentMix: {
      mixingRuleVersion: BIDDING_Q_COUNTERFACTUAL_OPPONENT_MIX_RULE_VERSION,
      ruleBasedWeight: 0.5,
      conservativeWeight: 0.5
    },
    handCount: options.samples.length,
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

function terminalRoleForResult(result: GameResult, actingPlayerId: PlayerId): string {
  if (result.resultType === "all-pass") return "no-contract";
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

function findMatchingLegalAction(observation: PlayerObservation, selectedAction: GameAction): GameAction {
  const action = observation.legalActions.find((legal) => actionsEqual(legal, selectedAction));
  if (action === undefined) {
    throw new Error(`selected action outside legal actions ${JSON.stringify(selectedAction)}`);
  }
  return action;
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

function validateSeat(value: number): void {
  if (!Number.isInteger(value) || value < 0 || value >= PLAYER_COUNT) {
    throw new Error("candidateSeatIndex must be an integer seat index.");
  }
}

function validatePositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${name} must be positive.`);
}

function validateUint32(name: string, value: number): void {
  if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
    throw new Error(`${name} must be a uint32 integer.`);
  }
}

function numericSummary(values: readonly number[]): NumericSummary {
  if (values.length === 0) return { count: 0, mean: null, std: null, min: null, max: null };
  return {
    count: values.length,
    mean: mean(values),
    std: std(values),
    min: Math.min(...values),
    max: Math.max(...values)
  };
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

function mean(values: readonly number[]): number {
  return sum(values) / values.length;
}

function std(values: readonly number[]): number {
  const average = mean(values);
  return Math.sqrt(values.reduce((total, value) => total + (value - average) ** 2, 0) / values.length);
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function sameStringArray(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}
