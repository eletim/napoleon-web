#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import {
  generateBiddingQCounterfactualDataset
} from "../packages/training-data/dist/index.js";
import {
  loadNonPlayingPolicyOnnxModel,
  loadPolicyOnnxModel
} from "../packages/policy-onnx/dist/index.js";

const args = parseArgs(process.argv.slice(2));

async function main() {
  const outputDirectory = requireArg("--output");
  const biddingOnnx = requireArg("--bidding-onnx");
  const biddingMetadata = requireArg("--bidding-metadata");
  const playingOnnx = args.get("--playing-onnx") ??
    "benchmarks/playing-policies/ppo-separated-v1000/policy.onnx";
  const playingMetadata = args.get("--playing-metadata") ??
    "benchmarks/playing-policies/ppo-separated-v1000/policy.json";
  const fixedAdjutantOnnx = args.get("--adjutant-onnx");
  const fixedAdjutantMetadata = args.get("--adjutant-metadata");
  const fixedExchangeOnnx = args.get("--exchange-onnx");
  const fixedExchangeMetadata = args.get("--exchange-metadata");
  const inferenceDevice = optionalChoice("--inference-device", ["cpu", "auto", "cuda"], "cpu");
  const randomSeed = optionalInt("--random-seed", 368);
  const logicalSeedCount = optionalInt("--logical-seeds", 100);
  const startSeed = optionalInt("--start-seed", randomSeed);
  const maxSourceStates = optionalInt("--max-source-states", logicalSeedCount * 5);
  const repeats = optionalInt("--repeats", 1);
  const gamesPerShard = optionalInt("--games-per-shard", 100);
  const randomLegalBidCount = optionalInt("--random-legal-bid-count", 2);

  if ((fixedAdjutantOnnx === undefined) !== (fixedAdjutantMetadata === undefined)) {
    throw new Error("--adjutant-onnx and --adjutant-metadata must be specified together.");
  }
  if ((fixedExchangeOnnx === undefined) !== (fixedExchangeMetadata === undefined)) {
    throw new Error("--exchange-onnx and --exchange-metadata must be specified together.");
  }

  const biddingPolicy = await loadNonPlayingPolicyOnnxModel({
    onnxPath: biddingOnnx,
    metadataPath: biddingMetadata,
    inferenceDevice,
    inferenceMaxBatchSize: 256
  });
  const playingPolicy = await loadPolicyOnnxModel({
    onnxPath: playingOnnx,
    metadataPath: playingMetadata,
    inferenceDevice,
    inferenceMaxBatchSize: 256
  });
  const fixedAdjutantPolicy = fixedAdjutantOnnx === undefined
    ? undefined
    : await loadNonPlayingPolicyOnnxModel({
        onnxPath: fixedAdjutantOnnx,
        metadataPath: fixedAdjutantMetadata,
        inferenceDevice,
        inferenceMaxBatchSize: 256
      });
  const fixedExchangePolicy = fixedExchangeOnnx === undefined
    ? undefined
    : await loadNonPlayingPolicyOnnxModel({
        onnxPath: fixedExchangeOnnx,
        metadataPath: fixedExchangeMetadata,
        inferenceDevice,
        inferenceMaxBatchSize: 256
      });

  const result = await generateBiddingQCounterfactualDataset({
    outputDirectory,
    biddingPolicy,
    biddingPolicyArtifact: {
      onnxPath: biddingOnnx,
      metadataPath: biddingMetadata,
      artifactId: args.get("--bidding-artifact-id") ?? "candidate-bidding"
    },
    playingPolicy,
    playingPolicyArtifact: {
      onnxPath: playingOnnx,
      metadataPath: playingMetadata,
      artifactId: args.get("--playing-artifact-id") ?? "ppo-separated-v1000"
    },
    fixedAdjutantPolicy,
    fixedAdjutantPolicyArtifact: fixedAdjutantOnnx === undefined ? undefined : {
      onnxPath: fixedAdjutantOnnx,
      metadataPath: fixedAdjutantMetadata,
      artifactId: args.get("--adjutant-artifact-id") ?? "fixed-adjutant"
    },
    fixedExchangePolicy,
    fixedExchangePolicyArtifact: fixedExchangeOnnx === undefined ? undefined : {
      onnxPath: fixedExchangeOnnx,
      metadataPath: fixedExchangeMetadata,
      artifactId: args.get("--exchange-artifact-id") ?? "fixed-exchange"
    },
    startSeed,
    logicalSeedCount,
    maxSourceStates,
    repeats,
    gamesPerShard,
    randomSeed,
    randomLegalBidCount,
    inferenceDevice,
    sourceCommit: args.get("--source-commit") ?? sourceCommit()
  });

  console.log(JSON.stringify({
    outputDirectory: result.outputDirectory,
    sampleCount: result.manifest.sampleCount,
    sourceStates: result.manifest.sourceStates,
    forcedStateActionPairs: result.manifest.forcedStateActionPairs,
    summary: result.summary
  }, null, 2));
}

function parseArgs(argv) {
  const parsed = new Map();
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--") {
      continue;
    }
    if (!arg.startsWith("--")) {
      throw new Error(`Unexpected positional argument: ${arg}`);
    }
    const equals = arg.indexOf("=");
    if (equals !== -1) {
      parsed.set(arg.slice(0, equals), arg.slice(equals + 1));
      continue;
    }
    const next = argv[index + 1];
    if (next === undefined || next.startsWith("--")) {
      parsed.set(arg, "true");
      continue;
    }
    parsed.set(arg, next);
    index += 1;
  }
  return parsed;
}

function requireArg(name) {
  const value = args.get(name);
  if (value === undefined || value.length === 0) {
    throw new Error(`Missing required argument ${name}.`);
  }
  return value;
}

function optionalInt(name, fallback) {
  const raw = args.get(name);
  if (raw === undefined) {
    return fallback;
  }
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${name} must be a non-negative integer.`);
  }
  return value;
}

function optionalChoice(name, choices, fallback) {
  const value = args.get(name) ?? fallback;
  if (!choices.includes(value)) {
    throw new Error(`${name} must be one of ${choices.join(", ")}.`);
  }
  return value;
}

function sourceCommit() {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  } catch (error) {
    const stdout = typeof error === "object" && error !== null && "stdout" in error
      ? error.stdout
      : undefined;
    if (typeof stdout === "string" && stdout.trim().length > 0) {
      return stdout.trim();
    }
    if (Buffer.isBuffer(stdout) && stdout.length > 0) {
      return stdout.toString("utf8").trim();
    }
    return null;
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
