#!/usr/bin/env node
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import {
  ISSUE427_T1_BIDDING_MARGIN_POLICY_ID,
  PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID,
  loadRepoManagedBiddingMarginPolicyBenchmark,
  loadRepoManagedPlayingPolicyBenchmark,
  runIssue429T1BiddingRuntimeEvaluation
} from "../packages/policy-onnx/dist/index.js";

const args = parseArgs(process.argv.slice(2));
const games = numberArg(args.games, 1000, "--games");
const startSeed = numberArg(args.startSeed, 429000000, "--start-seed");
const output = args.output ? resolve(args.output) : null;
const inferenceDevice = args.inferenceDevice ?? "cpu";

const playing = await loadRepoManagedPlayingPolicyBenchmark(
  PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID,
  { inferenceDevice }
);
if (playing.critic === undefined) {
  throw new Error("PPO separated benchmark artifact must include a critic for PASS Citizen EV.");
}
const t1 = await loadRepoManagedBiddingMarginPolicyBenchmark(
  ISSUE427_T1_BIDDING_MARGIN_POLICY_ID,
  { inferenceDevice }
);
const result = await runIssue429T1BiddingRuntimeEvaluation({
  startSeed,
  gameCount: games,
  playingPolicy: playing.policy,
  critic: playing.critic,
  t1MarginModel: t1.model,
  progress: args.progress === "true" ? (message) => console.error(`[issue429] ${message}`) : undefined
});

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
      completed: candidate.games.completed,
      crashed: candidate.games.crashed,
      openingBidRate: candidate.bidding.openingBidRate,
      raiseRate: candidate.bidding.raiseRate,
      contractSuccessRate: candidate.contracts.contractSuccessRate,
      candidateRelativeReward: candidate.rewards.candidateRelativeReward.mean,
      fallbackCount: candidate.safety.fallbackCount,
      inferenceFailureCount: candidate.safety.modelInferenceFailureCount
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
    } else {
      result[key] = next;
      index += 1;
    }
  }
  return result;
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
  console.error("usage: node scripts/evaluate-issue429-t1-runtime.mjs --games 1000 --start-seed 429000000 --output /tmp/issue429.json [--progress]");
  process.exit(2);
}
