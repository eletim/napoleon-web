#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import {
  generateHistoryConsistentRaiseMarginDataset
} from "../packages/training-data/dist/index.js";

const args = parseArgs(process.argv.slice(2));

if (!args.output) {
  usage("missing --output");
}

const pairCount = numberArg(args.pairs, 1000, "--pairs");
const randomSeed = numberArg(args.seed, 427000, "--seed");
const fixedHandCount = optionalNumberArg(args.fixedHands, "--fixed-hands");
const preferStrongHands = args.preferStrongHands === "true";
const handPoolMultiplier = numberArg(args.handPoolMultiplier, 20, "--hand-pool-multiplier");
const maxDealSeedsPerHand = numberArg(args.maxDealSeedsPerHand, 80, "--max-deal-seeds-per-hand");
const maxSamplesPerFixedHand = optionalNumberArg(args.maxSamplesPerFixedHand, "--max-samples-per-fixed-hand");
const actionCountPerState = numberArg(args.actionsPerState, 4, "--actions-per-state");
const gamesPerShard = numberArg(args.gamesPerShard, 1000, "--games-per-shard");
const candidateSeatIndex = optionalNumberArg(args.candidateSeatIndex, "--candidate-seat-index");
const sourceCommit = args.sourceCommit ?? gitCommitOrNull();

const result = await generateHistoryConsistentRaiseMarginDataset({
  outputDirectory: args.output,
  pairCount,
  randomSeed,
  fixedHandCount,
  preferStrongHands,
  handPoolMultiplier,
  maxDealSeedsPerHand,
  maxSamplesPerFixedHand,
  actionCountPerState,
  gamesPerShard,
  candidateSeatIndex,
  sourceCommit,
  onProgress: (progress) => {
    if (
      progress.sampleCount >= pairCount ||
      progress.completedHands % 10 === 0
    ) {
      console.error(
        `hands ${progress.completedHands}; dealSeeds ${progress.dealSeedsTried}; ` +
        `raiseStates ${progress.sourceStateCount}; samples ${progress.sampleCount}/${pairCount}`
      );
    }
  }
});

console.log(JSON.stringify({
  outputDirectory: result.outputDirectory,
  pairCount: result.manifest.pairCount,
  fixedHandCount: result.manifest.fixedHandCount,
  uniqueDealSeedCount: result.manifest.uniqueDealSeedCount,
  uniqueRaiseStateCount: result.manifest.uniqueRaiseStateCount,
  rolloutCount: result.manifest.rolloutCount,
  summary: result.manifest.summary
}, null, 2));

function parseArgs(values) {
  const result = {};
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value === "--") continue;
    if (!value.startsWith("--")) usage(`unexpected argument ${value}`);
    const key = value.slice(2).replace(/-([a-z])/g, (_match, letter) => letter.toUpperCase());
    const next = values[index + 1];
    if (next === undefined || next.startsWith("--")) {
      result[key] = "true";
    } else {
      result[key] = next;
      index += 1;
    }
  }
  return result;
}

function numberArg(value, fallback, name) {
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) usage(`${name} must be a positive integer`);
  return parsed;
}

function optionalNumberArg(value, name) {
  if (value === undefined) return undefined;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) usage(`${name} must be a non-negative integer`);
  return parsed;
}

function gitCommitOrNull() {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

function usage(message) {
  console.error(`error: ${message}`);
  console.error(
    "usage: node scripts/generate-history-consistent-raise-margin-dataset.mjs " +
    "--output /tmp/raise-ds --pairs 1000 [--fixed-hands 250] " +
    "[--prefer-strong-hands] [--hand-pool-multiplier 20] " +
    "[--max-deal-seeds-per-hand 80] [--max-samples-per-fixed-hand 4] [--actions-per-state 4]"
  );
  process.exit(1);
}
