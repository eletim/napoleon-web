#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { RuleBasedAgent } from "../packages/ai/dist/index.js";
import {
  generateExchangeCounterfactualDataset
} from "../packages/training-data/dist/index.js";
import {
  CriticEvBiddingAgent,
  FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID,
  PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID,
  PolicyOnnxAgent,
  T1NapoleonEvBiddingAgent,
  loadRepoManagedBiddingMarginPolicyBenchmark,
  loadRepoManagedPlayingPolicyBenchmark
} from "../packages/policy-onnx/dist/index.js";

const args = parseArgs(process.argv.slice(2));
const states = positiveIntegerArg(args.states, 5, "--states");
const startSeed = nonNegativeIntegerArg(args.startSeed, 434000000, "--start-seed");
const statesPerShard = positiveIntegerArg(args.statesPerShard, 10, "--states-per-shard");
const maxDealAttempts = positiveIntegerArg(args.maxDealAttempts, Math.max(states * 25, states), "--max-deal-attempts");
const outputDirectory = resolve(args.output ?? `/tmp/napoleon-exchange-counterfactual-${states}states`);
const inferenceDevice = args.inferenceDevice ?? "cpu";

const playing = await loadRepoManagedPlayingPolicyBenchmark(
  PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID,
  { inferenceDevice }
);
if (playing.critic === undefined) {
  throw new Error("Repo-managed playing benchmark must include a critic for T1 PASS EV bidding.");
}
const biddingMargin = await loadRepoManagedBiddingMarginPolicyBenchmark(
  FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID,
  { inferenceDevice }
);

const result = await generateExchangeCounterfactualDataset({
  outputDirectory,
  sourceStateCount: states,
  startSeed,
  statesPerShard,
  maxDealAttempts,
  sourceCommit: currentGitCommit(),
  biddingPolicy: {
    id: FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID,
    description: "Frozen raise v1 T1 Napoleon EV bidding runtime policy",
    provenance: biddingMargin.artifact
  },
  adjutantPolicy: {
    id: "rule-based-adjutant-v1",
    description: "RuleBasedAgent adjutant selection"
  },
  playingPolicy: {
    id: PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID,
    description: "Repo-managed fixed playing policy rollout",
    provenance: playing.artifact
  },
  playingPolicyDeterministic: true,
  createBiddingAgent: ({ rng }) => new T1NapoleonEvBiddingAgent({
    marginModel: biddingMargin.model,
    passEvAgent: new CriticEvBiddingAgent({
      critic: playing.critic,
      delegateAgent: new RuleBasedAgent(rng)
    }),
    delegateAgent: new RuleBasedAgent(rng),
    fallbackOnInferenceError: false
  }),
  createAdjutantAgent: ({ rng }) => new RuleBasedAgent(rng),
  createPlayingAgent: ({ rng }) => new PolicyOnnxAgent({
    policy: playing.policy,
    rng
  }),
  onProgress: args.progress === "true"
    ? (progress) => {
      console.error([
        `[exchange-cf] states=${progress.completedSourceStates}/${progress.requestedSourceStates}`,
        `samples=${progress.sampleCount}`,
        `rollouts=${progress.rolloutCount}`,
        `seed=${progress.currentSeed}`,
        `attempts=${progress.dealAttempts}`,
        `shards=${progress.completedShards}`
      ].join(" "));
    }
    : undefined
});

console.log(JSON.stringify({
  outputDirectory: result.outputDirectory,
  states: result.manifest.sourceStateCount,
  samples: result.manifest.sampleCount,
  rollouts: result.manifest.rolloutCount,
  candidatesPerState: result.manifest.summary.candidateCountPerState,
  invariantFailures: result.manifest.summary.invariantFailureCount,
  ruleBasedMatchRate: result.manifest.summary.ruleBasedMatchRate,
  ruleBasedRelativeRewardRegret: result.manifest.summary.ruleBasedRelativeRewardRegret,
  marginSpread: result.manifest.summary.marginSpread,
  bestBuriedPointCardCount: result.manifest.summary.bestBuriedPointCardCount,
  ruleBasedBuriedPointCardCount: result.manifest.summary.ruleBasedBuriedPointCardCount
}, null, 2));

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

function positiveIntegerArg(value, fallback, label) {
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    usage(`${label} must be a positive integer`);
  }
  return parsed;
}

function nonNegativeIntegerArg(value, fallback, label) {
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0) {
    usage(`${label} must be a non-negative integer`);
  }
  return parsed;
}

function currentGitCommit() {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], {
      encoding: "utf8"
    }).trim();
  } catch (error) {
    if (typeof error?.stdout === "string" && error.stdout.trim() !== "") {
      return error.stdout.trim();
    }
    return null;
  }
}

function usage(message) {
  console.error(message);
  console.error("usage: node scripts/generate-exchange-counterfactual-dataset.mjs --states 50 --start-seed 434000000 --output /tmp/issue434-exchange-cf --progress");
  process.exit(2);
}
