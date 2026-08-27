#!/usr/bin/env node
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import {
  PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID,
  loadBiddingMarginOnnxModel,
  loadRepoManagedPlayingPolicyBenchmark,
  runIssue429T1BiddingRuntimeEvaluation
} from "../packages/policy-onnx/dist/index.js";

const args = parseArgs(process.argv.slice(2));
const games = numberArg(args.games, 1000, "--games");
const startSeed = numberArg(args.startSeed, 429000000, "--start-seed");
const output = args.output ? resolve(args.output) : null;
const inferenceDevice = args.inferenceDevice ?? "cpu";
const models = arrayArg(args.model).map(parseModel);

if (models.length === 0) {
  usage("at least one --model label=/path/to/artifact-dir is required");
}

const playing = await loadRepoManagedPlayingPolicyBenchmark(
  PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID,
  { inferenceDevice }
);
if (playing.critic === undefined) {
  throw new Error("PPO separated benchmark artifact must include a critic for PASS Citizen EV.");
}

const candidates = [];
let baseline = null;
for (const model of models) {
  const margin = await loadBiddingMarginOnnxModel({
    onnxPath: `${model.directory}/margin.onnx`,
    metadataPath: `${model.directory}/margin.json`,
    inferenceDevice
  });
  const result = await runIssue429T1BiddingRuntimeEvaluation({
    startSeed,
    gameCount: games,
    playingPolicy: playing.policy,
    critic: playing.critic,
    t1MarginModel: margin,
    progress: args.progress === "true"
      ? (message) => console.error(`[issue431:${model.label}] ${message}`)
      : undefined
  });
  if (baseline === null) {
    baseline = result.candidates[0];
  }
  candidates.push({
    label: model.label,
    artifactDirectory: model.directory,
    summary: result.candidates[1],
    pairedComparisonAgainstCurrent: result.pairedComparison
  });
}

const result = {
  schemaVersion: 1,
  configuration: {
    startSeed,
    endSeed: startSeed + games - 1,
    gameCount: games,
    inferenceDevice,
    currentRuntimeBaseline: baseline
  },
  candidates
};

const json = `${JSON.stringify(result, null, 2)}\n`;
if (output === null) {
  process.stdout.write(json);
} else {
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, json, "utf8");
  console.log(JSON.stringify({
    output,
    games,
    startSeed,
    candidates: result.candidates.map((candidate) => ({
      label: candidate.label,
      completed: candidate.summary.games.completed,
      crashed: candidate.summary.games.crashed,
      openingBidRate: candidate.summary.bidding.openingBidRate,
      raiseRate: candidate.summary.bidding.raiseRate,
      contractSuccessRate: candidate.summary.contracts.contractSuccessRate,
      finalCandidateNapoleonRate: candidate.summary.contracts.finalCandidateNapoleonRate,
      candidateRelativeReward: candidate.summary.rewards.candidateRelativeReward.mean,
      fallbackCount: candidate.summary.safety.fallbackCount,
      inferenceFailureCount: candidate.summary.safety.modelInferenceFailureCount
    }))
  }, null, 2));
}

function parseArgs(values) {
  const result = {};
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!value.startsWith("--")) usage(`unexpected argument ${value}`);
    const key = value.slice(2).replace(/-([a-z])/g, (_match, letter) => letter.toUpperCase());
    const next = values[index + 1];
    if (next === undefined || next.startsWith("--")) {
      result[key] = "true";
    } else if (result[key] === undefined) {
      result[key] = next;
      index += 1;
    } else {
      result[key] = Array.isArray(result[key]) ? [...result[key], next] : [result[key], next];
      index += 1;
    }
  }
  return result;
}

function arrayArg(value) {
  if (value === undefined) return [];
  return Array.isArray(value) ? value : [value];
}

function parseModel(value) {
  const [label, directory] = value.split("=", 2);
  if (!label || !directory) {
    usage("--model must be label=/path/to/artifact-dir");
  }
  return { label, directory: resolve(directory) };
}

function numberArg(value, fallback, label) {
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    usage(`${label} must be a positive integer`);
  }
  return parsed;
}

function usage(message) {
  console.error(message);
  console.error(
    "usage: node scripts/evaluate-issue431-raise-scale-runtime.mjs " +
    "--games 1000 --start-seed 429000000 --model 5k=/path/to/artifact " +
    "[--model 10k=/path/to/artifact] [--output /tmp/issue431-runtime.json] [--progress]"
  );
  process.exit(2);
}
