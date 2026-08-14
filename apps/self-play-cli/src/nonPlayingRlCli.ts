import { writeFile } from "node:fs/promises";
import {
  generateNonPlayingAdjutantRlDataset,
  generateNonPlayingBiddingRlDataset,
  generateNonPlayingExchangeRlDataset,
  UINT32_MAX
} from "@napoleon/training-data";
import {
  loadNonPlayingPolicyOnnxModel,
  loadPolicyOnnxModel,
  runFullPolicyVsRuleBasedEvaluation
} from "@napoleon/policy-onnx";
import {
  optionalValue,
  parseOptionMap,
  parsePositiveInteger,
  parsePositiveNumber,
  parseUnsignedInteger,
  requireValue
} from "./cliArgs.js";
import { createProgressReporter } from "./formatProgress.js";

type NonPlayingPhase = "bidding" | "adjutant" | "exchange";

interface ParsedRolloutArgs {
  phase: NonPlayingPhase;
  policyOnnx: string;
  policyMetadata: string;
  playingOnnx: string;
  playingMetadata: string;
  output: string;
  startSeed: number;
  games: number;
  gamesPerShard: number;
  temperature: number;
  inferenceDevice: "cpu" | "auto" | "cuda";
  inferenceMaxBatchSize: number;
  artifactId: string | undefined;
  playingArtifactId: string | undefined;
  progressPrefix: string;
}

interface ParsedEvaluationArgs {
  playingOnnx: string;
  playingMetadata: string;
  biddingOnnx: string;
  biddingMetadata: string;
  adjutantOnnx: string;
  adjutantMetadata: string;
  exchangeOnnx: string;
  exchangeMetadata: string;
  output: string;
  startSeed: number;
  games: number;
  inferenceDevice: "cpu" | "auto" | "cuda";
  inferenceMaxBatchSize: number;
  progressPrefix: string;
}

const rolloutOptionNames = new Set([
  "--phase",
  "--policy-onnx",
  "--policy-metadata",
  "--playing-onnx",
  "--playing-metadata",
  "--output",
  "--start-seed",
  "--games",
  "--games-per-shard",
  "--temperature",
  "--inference-device",
  "--inference-max-batch-size",
  "--artifact-id",
  "--playing-artifact-id",
  "--progress-prefix"
]);

const evaluationOptionNames = new Set([
  "--playing-onnx",
  "--playing-metadata",
  "--bidding-onnx",
  "--bidding-metadata",
  "--adjutant-onnx",
  "--adjutant-metadata",
  "--exchange-onnx",
  "--exchange-metadata",
  "--output",
  "--start-seed",
  "--games",
  "--inference-device",
  "--inference-max-batch-size",
  "--progress-prefix"
]);

export async function runNonPlayingRlCli(
  command: "non-playing-rollout" | "full-policy-evaluate",
  argv: readonly string[],
  io: {
    stdout: { write: (chunk: string) => void };
    stderr: { write: (chunk: string) => void };
  }
): Promise<number> {
  try {
    if (command === "non-playing-rollout") {
      await runNonPlayingRollout(parseRolloutArgs(argv), io);
      return 0;
    }

    await runFullPolicyEvaluation(parseEvaluationArgs(argv), io);
    return 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    io.stderr.write(`${message}\n`);
    return 1;
  }
}

function parseRolloutArgs(argv: readonly string[]): ParsedRolloutArgs {
  const values = parseOptionMap(argv, rolloutOptionNames);
  const startSeed = parseUnsignedInteger("--start-seed", requireValue(values, "--start-seed"));
  const games = parsePositiveInteger("--games", requireValue(values, "--games"));
  validateSeedRange(startSeed, games);

  return {
    phase: parsePhase(requireValue(values, "--phase")),
    policyOnnx: requireValue(values, "--policy-onnx"),
    policyMetadata: requireValue(values, "--policy-metadata"),
    playingOnnx: requireValue(values, "--playing-onnx"),
    playingMetadata: requireValue(values, "--playing-metadata"),
    output: requireValue(values, "--output"),
    startSeed,
    games,
    gamesPerShard: parsePositiveInteger(
      "--games-per-shard",
      optionalValue(values, "--games-per-shard") ?? String(games)
    ),
    temperature: parsePositiveNumber("--temperature", optionalValue(values, "--temperature") ?? "1"),
    inferenceDevice: parseInferenceDevice(optionalValue(values, "--inference-device") ?? "cpu"),
    inferenceMaxBatchSize: parsePositiveInteger(
      "--inference-max-batch-size",
      optionalValue(values, "--inference-max-batch-size") ?? "256"
    ),
    artifactId: optionalValue(values, "--artifact-id"),
    playingArtifactId: optionalValue(values, "--playing-artifact-id"),
    progressPrefix: optionalValue(values, "--progress-prefix") ?? ""
  };
}

function parseEvaluationArgs(argv: readonly string[]): ParsedEvaluationArgs {
  const values = parseOptionMap(argv, evaluationOptionNames);
  const startSeed = parseUnsignedInteger("--start-seed", requireValue(values, "--start-seed"));
  const games = parsePositiveInteger("--games", requireValue(values, "--games"));
  validateSeedRange(startSeed, games);

  return {
    playingOnnx: requireValue(values, "--playing-onnx"),
    playingMetadata: requireValue(values, "--playing-metadata"),
    biddingOnnx: requireValue(values, "--bidding-onnx"),
    biddingMetadata: requireValue(values, "--bidding-metadata"),
    adjutantOnnx: requireValue(values, "--adjutant-onnx"),
    adjutantMetadata: requireValue(values, "--adjutant-metadata"),
    exchangeOnnx: requireValue(values, "--exchange-onnx"),
    exchangeMetadata: requireValue(values, "--exchange-metadata"),
    output: requireValue(values, "--output"),
    startSeed,
    games,
    inferenceDevice: parseInferenceDevice(optionalValue(values, "--inference-device") ?? "cpu"),
    inferenceMaxBatchSize: parsePositiveInteger(
      "--inference-max-batch-size",
      optionalValue(values, "--inference-max-batch-size") ?? "256"
    ),
    progressPrefix: optionalValue(values, "--progress-prefix") ?? ""
  };
}

async function runNonPlayingRollout(
  args: ParsedRolloutArgs,
  io: {
    stdout: { write: (chunk: string) => void };
    stderr: { write: (chunk: string) => void };
  }
): Promise<void> {
  const policy = await loadNonPlayingPolicyOnnxModel({
    onnxPath: args.policyOnnx,
    metadataPath: args.policyMetadata,
    inferenceDevice: args.inferenceDevice,
    inferenceMaxBatchSize: args.inferenceMaxBatchSize
  });
  const playingPolicy = await loadPolicyOnnxModel({
    onnxPath: args.playingOnnx,
    metadataPath: args.playingMetadata,
    inferenceDevice: args.inferenceDevice,
    inferenceMaxBatchSize: args.inferenceMaxBatchSize
  });
  const common = {
    outputDirectory: args.output,
    playingPolicy,
    playingPolicyArtifact: {
      onnxPath: args.playingOnnx,
      metadataPath: args.playingMetadata,
      artifactId: args.playingArtifactId
    },
    startSeed: args.startSeed,
    gameCount: args.games,
    gamesPerShard: args.gamesPerShard,
    temperature: args.temperature,
    onProgress: createProgressReporter(args.games, (text) => {
      io.stderr.write(`${args.progressPrefix}${text}`);
    })
  };
  const result =
    args.phase === "bidding"
      ? await generateNonPlayingBiddingRlDataset({
          ...common,
          biddingPolicy: policy,
          biddingPolicyArtifact: policyArtifact(args)
        })
      : args.phase === "adjutant"
        ? await generateNonPlayingAdjutantRlDataset({
            ...common,
            adjutantPolicy: policy,
            adjutantPolicyArtifact: policyArtifact(args)
          })
        : await generateNonPlayingExchangeRlDataset({
            ...common,
            exchangePolicy: policy,
            exchangePolicyArtifact: policyArtifact(args)
          });

  io.stdout.write(`${JSON.stringify({
    phase: args.phase,
    outputDirectory: result.outputDirectory,
    gameCount: result.manifest.gameCount,
    sampleCount: result.manifest.sampleCount,
    shardCount: result.manifest.shardCount,
    startSeed: result.manifest.startSeed,
    endSeed: result.manifest.endSeed,
    sampleType: result.manifest.sampleType,
    phaseScope: result.manifest.phaseScope,
    requestedInferenceDevice: result.manifest.behaviorPolicy.requestedInferenceDevice,
    resolvedInferenceDevice: result.manifest.behaviorPolicy.resolvedInferenceDevice,
    executionProvider: result.manifest.behaviorPolicy.executionProvider,
    behaviorOnnxSha256: result.manifest.behaviorPolicy.onnxSha256,
    behaviorMetadataSha256: result.manifest.behaviorPolicy.metadataSha256
  })}\n`);
}

async function runFullPolicyEvaluation(
  args: ParsedEvaluationArgs,
  io: {
    stdout: { write: (chunk: string) => void };
    stderr: { write: (chunk: string) => void };
  }
): Promise<void> {
  const playingPolicy = await loadPolicyOnnxModel({
    onnxPath: args.playingOnnx,
    metadataPath: args.playingMetadata,
    inferenceDevice: args.inferenceDevice,
    inferenceMaxBatchSize: args.inferenceMaxBatchSize
  });
  const biddingPolicy = await loadNonPlayingPolicyOnnxModel({
    onnxPath: args.biddingOnnx,
    metadataPath: args.biddingMetadata,
    inferenceDevice: args.inferenceDevice,
    inferenceMaxBatchSize: args.inferenceMaxBatchSize
  });
  const adjutantPolicy = await loadNonPlayingPolicyOnnxModel({
    onnxPath: args.adjutantOnnx,
    metadataPath: args.adjutantMetadata,
    inferenceDevice: args.inferenceDevice,
    inferenceMaxBatchSize: args.inferenceMaxBatchSize
  });
  const exchangePolicy = await loadNonPlayingPolicyOnnxModel({
    onnxPath: args.exchangeOnnx,
    metadataPath: args.exchangeMetadata,
    inferenceDevice: args.inferenceDevice,
    inferenceMaxBatchSize: args.inferenceMaxBatchSize
  });
  const result = await runFullPolicyVsRuleBasedEvaluation({
    playingPolicy,
    biddingPolicy,
    adjutantPolicy,
    exchangePolicy,
    startSeed: args.startSeed,
    gameCount: args.games
  });

  await writeFile(args.output, `${JSON.stringify(result, null, 2)}\n`, "utf8");

  const counts = result.diagnostics.policyAgentDecisionCounts;
  const completedGames = result.run.completedCount;
  const scheduledGames = result.run.gameCount * result.run.rotationOffsets.length;
  io.stderr.write(`${args.progressPrefix}completed ${completedGames}/${scheduledGames}\n`);
  io.stdout.write(`${JSON.stringify({
    output: args.output,
    startSeed: result.configuration.startSeed,
    endSeed: result.configuration.endSeed,
    gameCount: result.configuration.gameCount,
    scheduledGames,
    completedGames,
    failedGames: result.run.failedCount,
    policyAgentDecisionCounts: counts,
    fallbackCount: counts.ruleBasedFallbackDecisionCount,
    illegalActionCount: result.comparison.illegalActionCount,
    adjutantSelection: result.diagnostics.adjutantSelection
  })}\n`);
}

function policyArtifact(args: ParsedRolloutArgs) {
  return {
    onnxPath: args.policyOnnx,
    metadataPath: args.policyMetadata,
    artifactId: args.artifactId
  };
}

function parsePhase(value: string): NonPlayingPhase {
  if (value === "bidding" || value === "adjutant" || value === "exchange") {
    return value;
  }
  throw new Error("--phase must be one of bidding, adjutant, exchange.");
}

function parseInferenceDevice(value: string): "cpu" | "auto" | "cuda" {
  if (value === "cpu" || value === "auto" || value === "cuda") {
    return value;
  }
  throw new Error("--inference-device must be one of cpu, auto, cuda.");
}

function validateSeedRange(startSeed: number, games: number): void {
  const endSeed = startSeed + games - 1;
  if (startSeed > UINT32_MAX || !Number.isSafeInteger(endSeed) || endSeed > UINT32_MAX) {
    throw new Error(`Seed range exceeds uint32: ${startSeed}..${endSeed}`);
  }
}
