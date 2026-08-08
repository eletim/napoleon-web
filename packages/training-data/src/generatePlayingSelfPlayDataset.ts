import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import type { Agent, AutomatedGameRecord, DecisionRecord, PlayerObservation } from "@napoleon/ai";
import type { GameAction, PlayerId, WinningTeam } from "@napoleon/game-core";
import {
  CARD_COUNT,
  CARD_IDS,
  MODEL_INPUT_SCHEMA_VERSION,
  PLAYER_COUNT,
  PLAYING_ENCODER_SCHEMA_VERSION,
  createPlayingModelInput,
  createRelativePlayerOrder,
  encodeBiddingHistory,
  encodeBiddingHistoryFromPublicActions,
  encodePlayingObservation,
  getCardId,
  getCardIndex,
  validateEncodedPlayingObservation
} from "@napoleon/ai-observation";
import {
  DATASET_FORMAT,
  MAX_SHARD_COUNT,
  RULE_BASED_AGENT_VERSION,
  UINT32_MAX
} from "./schema.js";
import type { DatasetGenerationProgress, DatasetShardManifest } from "./types.js";
import { createJsonlShardWriter, type JsonlShardWriter } from "./shardWriter.js";
import { calculateCardIdsSha256, serializeManifest } from "./serialization.js";

export const PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE = "playing-self-play-sample" as const;
export const PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION = 1 as const;
export const PLAYING_SELF_PLAY_DATASET_SCHEMA_VERSION = 3 as const;
export const PLAYING_SELF_PLAY_DATASET_GENERATOR_VERSION = 1 as const;
export const PLAYING_SELF_PLAY_REWARD_TYPE = "terminal-team-win" as const;
export const PLAYING_SELF_PLAY_REWARD_VERSION = 1 as const;
export const PLAYING_SELF_PLAY_SAMPLING_ALGORITHM = "masked-categorical" as const;
export const DEFAULT_PLAYING_SELF_PLAY_TEMPERATURE = 1.0 as const;

export type PlayingSelfPlayRole = "napoleon" | "adjutant" | "alliance";

export interface PlayingSelfPlayOutcome {
  winner: WinningTeam;
  napoleonPlayerId: PlayerId;
  actingPlayerTeam: WinningTeam;
  actingPlayerRole: PlayingSelfPlayRole;
}

export interface PlayingSelfPlaySample {
  sampleType: typeof PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE;
  schemaVersion: typeof PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION;
  seed: number;
  step: number;
  actingPlayerId: PlayerId;
  relativePlayerIds: readonly PlayerId[];
  observation: ReturnType<typeof encodePlayingObservation>;
  selectedCardIndex: number;
  behaviorLogProbability: number;
  terminalReward: 1 | -1;
  outcome: PlayingSelfPlayOutcome;
}

export interface PlayingSelfPlayPolicy {
  metadata: unknown;
  predictLogits: (modelInput: Float32Array | readonly number[]) => Promise<Float32Array>;
}

export interface PlayingSelfPlayPolicyArtifactOptions {
  onnxPath: string;
  metadataPath: string;
  artifactId?: string;
}

export interface PlayingSelfPlaySampledAction {
  selectedCardIndex: number;
  logProbability: number;
}

export interface GeneratePlayingSelfPlayDatasetOptions {
  outputDirectory: string;
  playingPolicy: PlayingSelfPlayPolicy;
  playingPolicyArtifact: PlayingSelfPlayPolicyArtifactOptions;
  startSeed: number;
  gameCount: number;
  gamesPerShard: number;
  temperature?: number;
  onProgress?: (progress: DatasetGenerationProgress) => void;
  maxDecisionSteps?: number;
}

export interface PlayingSelfPlayDatasetManifest {
  datasetSchemaVersion: typeof PLAYING_SELF_PLAY_DATASET_SCHEMA_VERSION;
  generatorVersion: typeof PLAYING_SELF_PLAY_DATASET_GENERATOR_VERSION;
  format: typeof DATASET_FORMAT;
  sampleType: typeof PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE;
  sampleSchemaVersion: typeof PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION;
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
  shards: readonly DatasetShardManifest[];
  playingEncoderSchemaVersion: typeof PLAYING_ENCODER_SCHEMA_VERSION;
  playingModelInputSchemaVersion: typeof MODEL_INPUT_SCHEMA_VERSION;
  behaviorPolicy: {
    type: "playing-onnx";
    artifactId: string;
    onnxFileName: string;
    metadataFileName: string;
    onnxSha256: string;
    metadataSha256: string;
    metadata: unknown;
  };
  samplingAlgorithm: typeof PLAYING_SELF_PLAY_SAMPLING_ALGORITHM;
  temperature: number;
  reward: {
    type: typeof PLAYING_SELF_PLAY_REWARD_TYPE;
    version: typeof PLAYING_SELF_PLAY_REWARD_VERSION;
  };
  nonPlayingAgent: {
    type: "rule-based";
    version: typeof RULE_BASED_AGENT_VERSION;
  };
}

export interface GeneratePlayingSelfPlayDatasetResult {
  outputDirectory: string;
  manifest: PlayingSelfPlayDatasetManifest;
}

export async function generatePlayingSelfPlayDataset(
  options: GeneratePlayingSelfPlayDatasetOptions
): Promise<GeneratePlayingSelfPlayDatasetResult> {
  const temperature = options.temperature ?? DEFAULT_PLAYING_SELF_PLAY_TEMPERATURE;
  validatePlayingSelfPlayGenerationOptions({ ...options, temperature });

  const outputDirectory = resolve(options.outputDirectory);
  await ensureOutputDoesNotExist(outputDirectory);
  await mkdir(dirname(outputDirectory), { recursive: true });

  const artifact = {
    artifactId: options.playingPolicyArtifact.artifactId ?? basename(options.playingPolicyArtifact.metadataPath),
    onnxFileName: basename(options.playingPolicyArtifact.onnxPath),
    metadataFileName: basename(options.playingPolicyArtifact.metadataPath),
    onnxSha256: await sha256File(options.playingPolicyArtifact.onnxPath),
    metadataSha256: await sha256File(options.playingPolicyArtifact.metadataPath),
    metadata: options.playingPolicy.metadata
  };
  const tempDirectory = await mkdtemp(
    join(dirname(outputDirectory), `.${basenameForTemp(outputDirectory)}.tmp-`)
  );
  let activeShard: JsonlShardWriter<PlayingSelfPlaySample> | null = null;

  try {
    const shards: DatasetShardManifest[] = [];
    let totalSampleCount = 0;
    let shardGameCount = 0;

    for (let gameOffset = 0; gameOffset < options.gameCount; gameOffset += 1) {
      const seed = options.startSeed + gameOffset;

      if (activeShard === null) {
        activeShard = createJsonlShardWriter(
          tempDirectory,
          shards.length,
          seed,
          serializePlayingSelfPlaySample
        );
        shardGameCount = 0;
      }

      const record = await runPlayingSelfPlayGame(seed, options.playingPolicy, temperature, options.maxDecisionSteps);
      const samples = await createPlayingSelfPlaySamples(record, options.playingPolicy, temperature);

      for (const sample of samples) {
        validatePlayingSelfPlaySample(sample, seed);
        await activeShard.writeSample(sample);
      }

      totalSampleCount += samples.length;
      shardGameCount += 1;

      const shardIsComplete =
        shardGameCount === options.gamesPerShard || gameOffset === options.gameCount - 1;

      if (shardIsComplete) {
        const completedShard = await activeShard.close(seed, shardGameCount);
        shards.push(completedShard);
        activeShard = null;
      }

      options.onProgress?.({
        completedGames: gameOffset + 1,
        totalGames: options.gameCount,
        sampleCount: totalSampleCount,
        completedShards: shards.length,
        currentSeed: seed
      });
    }

    const manifest = createPlayingSelfPlayManifest({
      options,
      temperature,
      sampleCount: totalSampleCount,
      shards,
      artifact
    });

    validatePlayingSelfPlayDatasetManifest(manifest);
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

export async function createPlayingSelfPlaySamples(
  record: AutomatedGameRecord,
  policy: PlayingSelfPlayPolicy,
  temperature: number = DEFAULT_PLAYING_SELF_PLAY_TEMPERATURE
): Promise<readonly PlayingSelfPlaySample[]> {
  validateTemperature(temperature);

  const samples: PlayingSelfPlaySample[] = [];

  for (const decision of record.decisions) {
    if (decision.phase !== "playing") {
      continue;
    }

    samples.push(await createPlayingSelfPlaySample(record, decision, policy, temperature));
  }

  return samples;
}

export function serializePlayingSelfPlaySample(sample: PlayingSelfPlaySample): string {
  return `${JSON.stringify(sample)}\n`;
}

export function calculatePlayingSelfPlayLogProbability(options: {
  logits: Float32Array | readonly number[];
  legalPlayMask: ArrayLike<number | boolean>;
  selectedCardIndex: number;
  temperature?: number;
}): number {
  const distribution = createMaskedCategoricalDistribution(
    options.logits,
    options.legalPlayMask,
    options.temperature ?? DEFAULT_PLAYING_SELF_PLAY_TEMPERATURE
  );
  const index = distribution.legalCardIndices.indexOf(options.selectedCardIndex);

  if (index === -1) {
    throw new Error(`selectedCardIndex ${options.selectedCardIndex} is not legal under legalPlayMask.`);
  }

  return distribution.logProbabilities[index];
}

export function validatePlayingSelfPlaySample(
  sample: PlayingSelfPlaySample,
  expectedSeed: number
): void {
  validateJsonSafeValue("sample", sample);

  if (sample.sampleType !== PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE) {
    throw new Error(`Sample sampleType must be ${PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE}.`);
  }
  if (sample.schemaVersion !== PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION) {
    throw new Error(`Unexpected self-play sample schemaVersion: ${sample.schemaVersion}`);
  }
  if (sample.seed !== expectedSeed) {
    throw new Error(`Sample seed must match current seed: ${sample.seed} !== ${expectedSeed}`);
  }
  if (sample.actingPlayerId !== sample.relativePlayerIds[0]) {
    throw new Error("actingPlayerId must match relativePlayerIds[0].");
  }
  if (!sameStringArray(sample.relativePlayerIds, sample.observation.relativePlayerIds)) {
    throw new Error("sample.relativePlayerIds must match observation.relativePlayerIds.");
  }

  validateEncodedPlayingObservation(sample.observation);
  validateCardIndex("selectedCardIndex", sample.selectedCardIndex);

  if (sample.observation.legalPlayMask[sample.selectedCardIndex] !== 1) {
    throw new Error("selectedCardIndex must be legal in observation.legalPlayMask.");
  }
  if (!Number.isFinite(sample.behaviorLogProbability) || sample.behaviorLogProbability > 1e-12) {
    throw new Error("behaviorLogProbability must be finite and <= 0.");
  }
  if (sample.terminalReward !== 1 && sample.terminalReward !== -1) {
    throw new Error("terminalReward must be +1 or -1.");
  }
  if (sample.outcome.actingPlayerTeam === sample.outcome.winner && sample.terminalReward !== 1) {
    throw new Error("Winning team samples must have terminalReward +1.");
  }
  if (sample.outcome.actingPlayerTeam !== sample.outcome.winner && sample.terminalReward !== -1) {
    throw new Error("Losing team samples must have terminalReward -1.");
  }
}

export function validatePlayingSelfPlayDatasetManifest(
  manifest: PlayingSelfPlayDatasetManifest
): void {
  if (manifest.datasetSchemaVersion !== PLAYING_SELF_PLAY_DATASET_SCHEMA_VERSION) {
    throw new Error("Self-play manifest datasetSchemaVersion mismatch.");
  }
  if (manifest.generatorVersion !== PLAYING_SELF_PLAY_DATASET_GENERATOR_VERSION) {
    throw new Error("Self-play manifest generatorVersion mismatch.");
  }
  if (
    manifest.format !== DATASET_FORMAT ||
    manifest.sampleType !== PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE ||
    manifest.sampleSchemaVersion !== PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION
  ) {
    throw new Error("Self-play manifest format or sample type mismatch.");
  }
  validateUint32("Manifest startSeed", manifest.startSeed);
  validateUint32("Manifest endSeed", manifest.endSeed);
  validatePositiveInteger("Manifest gameCount", manifest.gameCount);
  validatePositiveInteger("Manifest sampleCount", manifest.sampleCount);
  validatePositiveInteger("Manifest gamesPerShard", manifest.gamesPerShard);
  validatePositiveInteger("Manifest shardCount", manifest.shardCount);

  if (manifest.endSeed !== manifest.startSeed + manifest.gameCount - 1) {
    throw new Error("Self-play manifest seed range mismatch.");
  }
  if (manifest.shardCount !== Math.ceil(manifest.gameCount / manifest.gamesPerShard)) {
    throw new Error("Self-play manifest shardCount mismatch.");
  }
  if (manifest.shardCount !== manifest.shards.length) {
    throw new Error("Self-play manifest shardCount must match shards length.");
  }
  if (manifest.sampleCount !== sum(manifest.shards.map((shard) => shard.sampleCount))) {
    throw new Error("Self-play manifest sampleCount must equal shard sample counts.");
  }
  if (manifest.playerCount !== PLAYER_COUNT || manifest.cardCount !== CARD_COUNT) {
    throw new Error("Self-play manifest fixed dimensions mismatch.");
  }
  if (!sameStringArray(manifest.cardIds, CARD_IDS)) {
    throw new Error("Self-play manifest cardIds mismatch.");
  }
  if (manifest.cardIdsSha256 !== calculateCardIdsSha256()) {
    throw new Error("Self-play manifest cardIdsSha256 mismatch.");
  }
  if (
    manifest.playingEncoderSchemaVersion !== PLAYING_ENCODER_SCHEMA_VERSION ||
    manifest.playingModelInputSchemaVersion !== MODEL_INPUT_SCHEMA_VERSION
  ) {
    throw new Error("Self-play manifest playing schema metadata mismatch.");
  }
  if (
    manifest.behaviorPolicy.type !== "playing-onnx" ||
    !sha256Pattern.test(manifest.behaviorPolicy.onnxSha256) ||
    !sha256Pattern.test(manifest.behaviorPolicy.metadataSha256)
  ) {
    throw new Error("Self-play manifest behavior policy metadata mismatch.");
  }
  if (manifest.samplingAlgorithm !== PLAYING_SELF_PLAY_SAMPLING_ALGORITHM) {
    throw new Error("Self-play manifest samplingAlgorithm mismatch.");
  }
  validateTemperature(manifest.temperature);
  if (
    manifest.reward.type !== PLAYING_SELF_PLAY_REWARD_TYPE ||
    manifest.reward.version !== PLAYING_SELF_PLAY_REWARD_VERSION
  ) {
    throw new Error("Self-play manifest reward metadata mismatch.");
  }
  if (
    manifest.nonPlayingAgent.type !== "rule-based" ||
    manifest.nonPlayingAgent.version !== RULE_BASED_AGENT_VERSION
  ) {
    throw new Error("Self-play manifest non-playing agent metadata mismatch.");
  }

  validateShards(manifest);
}

async function runPlayingSelfPlayGame(
  seed: number,
  policy: PlayingSelfPlayPolicy,
  temperature: number,
  maxDecisionSteps: number | undefined
): Promise<AutomatedGameRecord> {
  return runAutomatedGame({
    seed,
    maxDecisionSteps,
    createAgent: ({ rng }) =>
      new PlayingSelfPlayAgent({
        policy,
        rng,
        temperature
      })
  });
}

class PlayingSelfPlayAgent implements Agent {
  private readonly ruleBasedAgent: RuleBasedAgent;

	  constructor(
	    private readonly options: {
	      policy: PlayingSelfPlayPolicy;
	      rng: () => number;
	      temperature: number;
	    }
  ) {
    this.ruleBasedAgent = new RuleBasedAgent(options.rng);
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    if (observation.view.phase !== "playing") {
      return this.ruleBasedAgent.selectAction(observation);
    }

    const { modelInput, legalPlayMask } = createPlayingSelfPlayPolicyInput(observation);
    const logits = await this.options.policy.predictLogits(modelInput);
    const selection = samplePlayingSelfPlayAction({
      logits,
      legalPlayMask,
      rng: this.options.rng,
      temperature: this.options.temperature
    });
    const selectedCardId = getCardId(selection.selectedCardIndex);
    const selectedAction = observation.legalActions.find(
      (action) => action.type === "play-card" && action.cardId === selectedCardId
    );

    if (selectedAction === undefined) {
      throw new Error(
        `Self-play ONNX policy selected card index ${selection.selectedCardIndex} (${selectedCardId}) outside legal actions.`
      );
    }

    return selectedAction;
  }
}

function createPlayingSelfPlayPolicyInput(observation: PlayerObservation): {
  modelInput: Float32Array;
  legalPlayMask: readonly number[];
} {
  if (observation.view.phase !== "playing") {
    throw new Error(
      `createPlayingSelfPlayPolicyInput requires a playing observation, got ${observation.view.phase}.`
    );
  }

  if (observation.publicActionHistory === undefined) {
    throw new Error("Playing self-play policy input requires publicActionHistory.");
  }

  const absolutePlayerIds = observation.view.players.map((player) => player.id);
  const relativePlayerIds = createRelativePlayerOrder(absolutePlayerIds, observation.playerId);
  const biddingHistory = encodeBiddingHistoryFromPublicActions(
    observation.publicActionHistory,
    relativePlayerIds
  );
  const encodedObservation = encodePlayingObservation(
    observation,
    absolutePlayerIds,
    biddingHistory
  );

  return createPlayingModelInput(encodedObservation);
}

function samplePlayingSelfPlayAction(options: {
  logits: Float32Array | readonly number[];
  legalPlayMask: ArrayLike<number | boolean>;
  rng: () => number;
  temperature?: number;
}): PlayingSelfPlaySampledAction {
  const distribution = createMaskedCategoricalDistribution(
    options.logits,
    options.legalPlayMask,
    options.temperature ?? DEFAULT_PLAYING_SELF_PLAY_TEMPERATURE
  );

  if (distribution.legalCardIndices.length === 1) {
    return {
      selectedCardIndex: distribution.legalCardIndices[0],
      logProbability: 0
    };
  }

  const randomValue = options.rng();
  if (!Number.isFinite(randomValue) || randomValue < 0 || randomValue >= 1) {
    throw new Error("rng must return a finite value in [0, 1).");
  }

  let cumulativeProbability = 0;

  for (let index = 0; index < distribution.legalCardIndices.length; index += 1) {
    cumulativeProbability += distribution.probabilities[index];

    if (randomValue < cumulativeProbability) {
      return {
        selectedCardIndex: distribution.legalCardIndices[index],
        logProbability: distribution.logProbabilities[index]
      };
    }
  }

  const lastIndex = distribution.legalCardIndices.length - 1;

  return {
    selectedCardIndex: distribution.legalCardIndices[lastIndex],
    logProbability: distribution.logProbabilities[lastIndex]
  };
}

function createMaskedCategoricalDistribution(
  logits: Float32Array | readonly number[],
  legalPlayMask: ArrayLike<number | boolean>,
  temperature: number
): {
  legalCardIndices: readonly number[];
  probabilities: readonly number[];
  logProbabilities: readonly number[];
} {
  if (logits.length !== CARD_COUNT) {
    throw new Error(`logits must contain ${CARD_COUNT} values, got ${logits.length}.`);
  }

  validateLegalPlayMask(legalPlayMask);
  validateTemperature(temperature);

  const legalCardIndices: number[] = [];
  const scaledLogits: number[] = [];

  for (let index = 0; index < CARD_COUNT; index += 1) {
    const logit = Number(logits[index]);

    if (!Number.isFinite(logit)) {
      throw new Error(`logits[${index}] must be finite.`);
    }

    if (isLegalMaskValue(legalPlayMask[index])) {
      legalCardIndices.push(index);
      scaledLogits.push(logit / temperature);
    }
  }

  if (legalCardIndices.length === 1) {
    return {
      legalCardIndices,
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

  for (let index = 0; index < probabilities.length; index += 1) {
    if (
      !Number.isFinite(probabilities[index]) ||
      probabilities[index] < 0 ||
      !Number.isFinite(logProbabilities[index]) ||
      logProbabilities[index] > 1e-12
    ) {
      throw new Error("masked categorical distribution contains an invalid probability.");
    }
  }

  return {
    legalCardIndices,
    probabilities,
    logProbabilities
  };
}

async function createPlayingSelfPlaySample(
  record: AutomatedGameRecord,
  decision: DecisionRecord,
  policy: PlayingSelfPlayPolicy,
  temperature: number
): Promise<PlayingSelfPlaySample> {
  if (decision.action.type !== "play-card") {
    throw new Error(`Playing self-play decision action must be play-card, got ${decision.action.type}.`);
  }

  const relativePlayerIds = createRelativePlayerOrder(record.playerIds, decision.playerId);
  const biddingHistory = encodeBiddingHistory(record, decision, relativePlayerIds);
  const observation = encodePlayingObservation(
    decision.observation,
    record.playerIds,
    biddingHistory
  );
  const selectedCardIndex = getCardIndex(decision.action.cardId);
  const { modelInput, legalPlayMask } = createPlayingModelInput(observation);
  const logits = await policy.predictLogits(modelInput);
  const behaviorLogProbability = calculatePlayingSelfPlayLogProbability({
    logits,
    legalPlayMask,
    selectedCardIndex,
    temperature
  });
  const outcome = createOutcome(record, decision.playerId);

  return {
    sampleType: PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE,
    schemaVersion: PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION,
    seed: record.seed,
    step: decision.step,
    actingPlayerId: decision.playerId,
    relativePlayerIds: observation.relativePlayerIds,
    observation,
    selectedCardIndex,
    behaviorLogProbability,
    terminalReward: outcome.actingPlayerTeam === outcome.winner ? 1 : -1,
    outcome
  };
}

function createOutcome(record: AutomatedGameRecord, actingPlayerId: PlayerId): PlayingSelfPlayOutcome {
  const actingPlayerTeam = getPlayerTeam(actingPlayerId, record.result);

  return {
    winner: record.result.winner,
    napoleonPlayerId: record.result.napoleonPlayerId,
    actingPlayerTeam,
    actingPlayerRole: getPlayerRole(actingPlayerId, record.result)
  };
}

function getPlayerTeam(playerId: PlayerId, result: AutomatedGameRecord["result"]): WinningTeam {
  if (
    playerId === result.napoleonPlayerId ||
    (result.adjutantPlayerId !== null && playerId === result.adjutantPlayerId)
  ) {
    return "napoleon-team";
  }

  return "alliance";
}

function getPlayerRole(playerId: PlayerId, result: AutomatedGameRecord["result"]): PlayingSelfPlayRole {
  if (playerId === result.napoleonPlayerId) {
    return "napoleon";
  }
  if (result.adjutantPlayerId !== null && playerId === result.adjutantPlayerId) {
    return "adjutant";
  }

  return "alliance";
}

function createPlayingSelfPlayManifest(input: {
  options: GeneratePlayingSelfPlayDatasetOptions;
  temperature: number;
  sampleCount: number;
  shards: readonly DatasetShardManifest[];
  artifact: {
    artifactId: string;
    onnxFileName: string;
    metadataFileName: string;
    onnxSha256: string;
    metadataSha256: string;
    metadata: unknown;
  };
}): PlayingSelfPlayDatasetManifest {
  return {
    datasetSchemaVersion: PLAYING_SELF_PLAY_DATASET_SCHEMA_VERSION,
    generatorVersion: PLAYING_SELF_PLAY_DATASET_GENERATOR_VERSION,
    format: DATASET_FORMAT,
    sampleType: PLAYING_SELF_PLAY_DATASET_SAMPLE_TYPE,
    sampleSchemaVersion: PLAYING_SELF_PLAY_SAMPLE_SCHEMA_VERSION,
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
    shards: input.shards,
    playingEncoderSchemaVersion: PLAYING_ENCODER_SCHEMA_VERSION,
    playingModelInputSchemaVersion: MODEL_INPUT_SCHEMA_VERSION,
    behaviorPolicy: {
      type: "playing-onnx",
      ...input.artifact
    },
    samplingAlgorithm: PLAYING_SELF_PLAY_SAMPLING_ALGORITHM,
    temperature: input.temperature,
    reward: {
      type: PLAYING_SELF_PLAY_REWARD_TYPE,
      version: PLAYING_SELF_PLAY_REWARD_VERSION
    },
    nonPlayingAgent: {
      type: "rule-based",
      version: RULE_BASED_AGENT_VERSION
    }
  };
}

function validatePlayingSelfPlayGenerationOptions(
  options: GeneratePlayingSelfPlayDatasetOptions & { temperature: number }
): void {
  validateUint32("startSeed", options.startSeed);
  validatePositiveInteger("gameCount", options.gameCount);
  validatePositiveInteger("gamesPerShard", options.gamesPerShard);
  validateTemperature(options.temperature);

  if (options.outputDirectory.length === 0) {
    throw new Error("outputDirectory must be a non-empty path.");
  }

  const endSeed = options.startSeed + options.gameCount - 1;
  if (!Number.isSafeInteger(endSeed) || endSeed > UINT32_MAX) {
    throw new Error(`Seed range exceeds uint32: ${options.startSeed}..${endSeed}`);
  }

  const expectedShardCount = Math.ceil(options.gameCount / options.gamesPerShard);
  if (expectedShardCount > MAX_SHARD_COUNT) {
    throw new Error(
      `Dataset would require ${expectedShardCount} shards, exceeding the maximum ${MAX_SHARD_COUNT}.`
    );
  }
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

function validateShards(manifest: PlayingSelfPlayDatasetManifest): void {
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
    if (shard.startSeed !== expectedStartSeed) {
      throw new Error(`Shard ${shard.file} has a seed gap or overlap.`);
    }
    if (shard.gameCount !== shard.endSeed - shard.startSeed + 1) {
      throw new Error(`Shard ${shard.file} gameCount must match seed range.`);
    }

    const isLastShard = index === manifest.shards.length - 1;
    if (!isLastShard && shard.gameCount !== manifest.gamesPerShard) {
      throw new Error(`Shard ${shard.file} gameCount must match gamesPerShard.`);
    }
    if (isLastShard && shard.gameCount > manifest.gamesPerShard) {
      throw new Error(`Final shard ${shard.file} gameCount must not exceed gamesPerShard.`);
    }

    seenFiles.add(shard.file);
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
  validatePositiveInteger(`Shard ${shard.file} sampleCount`, shard.sampleCount);
  validatePositiveInteger(`Shard ${shard.file} byteLength`, shard.byteLength);

  if (shard.endSeed < shard.startSeed) {
    throw new Error(`Shard ${shard.file} endSeed must be greater than or equal to startSeed.`);
  }
  if (!sha256Pattern.test(shard.sha256)) {
    throw new Error(`Shard ${shard.file} sha256 must be lowercase hex.`);
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

function validateCardIndex(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value >= CARD_COUNT) {
    throw new Error(`${name} must be an integer between 0 and ${CARD_COUNT - 1}.`);
  }
}

function validateTemperature(temperature: number): void {
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new Error("temperature must be finite and greater than 0.");
  }
}

function validateLegalPlayMask(mask: ArrayLike<number | boolean>): void {
  if (mask.length !== CARD_COUNT) {
    throw new Error(`legalPlayMask must contain ${CARD_COUNT} values, got ${mask.length}.`);
  }

  let legalCount = 0;
  for (let index = 0; index < CARD_COUNT; index += 1) {
    const value = mask[index];
    if (value !== 0 && value !== 1 && value !== false && value !== true) {
      throw new Error(`legalPlayMask[${index}] must be 0/1 or boolean.`);
    }
    if (isLegalMaskValue(value)) {
      legalCount += 1;
    }
  }

  if (legalCount === 0) {
    throw new Error("legalPlayMask must contain at least one legal card.");
  }
}

function isLegalMaskValue(value: number | boolean): boolean {
  return value === 1 || value === true;
}

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

async function sha256File(path: string): Promise<string> {
  return createHash("sha256").update(await readFile(path)).digest("hex");
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

function sameStringArray(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) {
    return false;
  }

  return left.every((value, index) => value === right[index]);
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

const sha256Pattern = /^[0-9a-f]{64}$/;
