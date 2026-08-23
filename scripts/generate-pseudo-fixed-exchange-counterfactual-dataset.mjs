#!/usr/bin/env node
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import {
  ConservativeBiddingAgent,
  RuleBasedAgent
} from "../packages/ai/dist/index.js";
import {
  generatePseudoFixedExchangeCounterfactualDataset
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

const OPPONENT_POLICY_IDS = [
  FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID,
  "strong-rule-based-bidding-v1",
  "conservative-bidding-v1"
];

const args = parseArgs(process.argv.slice(2));
const groups = positiveIntegerArg(args.groups, 50, "--groups");
const repeats = positiveIntegerArg(args.repeats, 10, "--repeats");
const startSeed = nonNegativeIntegerArg(args.startSeed, 440000000, "--start-seed");
const statesPerShard = positiveIntegerArg(args.statesPerShard, 10, "--states-per-shard");
const maxDealAttempts = positiveIntegerArg(
  args.maxDealAttempts,
  Math.max(groups * repeats * 60, groups * repeats),
  "--max-deal-attempts"
);
const maxDealAttemptsPerAccepted = positiveIntegerArg(
  args.maxDealAttemptsPerAccepted,
  60,
  "--max-deal-attempts-per-accepted"
);
const outputDirectory = resolve(
  args.output ?? `/tmp/napoleon-pseudo-fixed-exchange-${groups}x${repeats}`
);
const inferenceDevice = args.inferenceDevice ?? "cpu";
const candidatePlayerIndex = nonNegativeIntegerArg(
  args.candidatePlayerIndex,
  0,
  "--candidate-player-index"
);

class CandidateNoPassFrozenBiddingAgent {
  constructor(frozenAgent) {
    this.frozenAgent = frozenAgent;
  }

  async selectAction(observation) {
    if (observation.view.phase !== "bidding") {
      return this.frozenAgent.selectAction(observation);
    }
    const bidEvaluations = await this.frozenAgent.evaluateLegalBidCandidates(observation);
    if (bidEvaluations.length > 0) {
      return bidEvaluations[0].action;
    }
    return this.frozenAgent.selectAction(observation);
  }
}

function selectOpponentPolicyId(seed, playerIndex) {
  const digest = createHash("sha256")
    .update(`issue440-opponent-policy:${seed}:${playerIndex}`)
    .digest();
  return OPPONENT_POLICY_IDS[digest[0] % OPPONENT_POLICY_IDS.length];
}

const playing = await loadRepoManagedPlayingPolicyBenchmark(
  PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID,
  { inferenceDevice }
);
if (playing.critic === undefined) {
  throw new Error("Repo-managed playing benchmark must include a critic for Frozen bidding.");
}
const biddingMargin = await loadRepoManagedBiddingMarginPolicyBenchmark(
  FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID,
  { inferenceDevice }
);

function createFrozenBiddingAgent(rng) {
  return new T1NapoleonEvBiddingAgent({
    marginModel: biddingMargin.model,
    passEvAgent: new CriticEvBiddingAgent({
      critic: playing.critic,
      delegateAgent: new RuleBasedAgent(rng)
    }),
    delegateAgent: new RuleBasedAgent(rng),
    fallbackOnInferenceError: false
  });
}

const result = await generatePseudoFixedExchangeCounterfactualDataset({
  outputDirectory,
  fixedThirteenGroupCount: groups,
  acceptedDealsPerFixedThirteenGroup: repeats,
  startSeed,
  statesPerShard,
  maxDealAttempts,
  maxDealAttemptsPerAccepted,
  diagnosticOnly: args.diagnosticOnly === "true",
  candidatePlayerIndex,
  sourceCommit: currentGitCommit(),
  biddingPolicy: {
    id: "pseudo-fixed-candidate-frozen-no-pass-opponent-1-1-1-v1",
    description: "Candidate uses frozen-raise-v1 with PASS forbidden while legal BID exists; opponents use seeded Frozen/strong RuleBased/Conservative 1:1:1.",
    provenance: {
      candidateBasePolicy: biddingMargin.artifact,
      opponentMixRule: "per-seat-seeded-frozen-strong-conservative-1-1-1-v1"
    }
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
  createBiddingAgent: (context) => {
    if (context.playerIndex === candidatePlayerIndex) {
      return new CandidateNoPassFrozenBiddingAgent(createFrozenBiddingAgent(context.rng));
    }
    switch (selectOpponentPolicyId(context.seed, context.playerIndex)) {
      case FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID:
        return createFrozenBiddingAgent(context.rng);
      case "strong-rule-based-bidding-v1":
        return new RuleBasedAgent(context.rng);
      case "conservative-bidding-v1":
        return new ConservativeBiddingAgent(context.rng);
      default:
        throw new Error("unreachable opponent policy");
    }
  },
  getBiddingPolicyId: (context) =>
    context.playerIndex === candidatePlayerIndex
      ? `${FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID}-candidate-no-pass`
      : selectOpponentPolicyId(context.seed, context.playerIndex),
  createAdjutantAgent: ({ rng }) => new RuleBasedAgent(rng),
  createPlayingAgent: ({ rng }) => new PolicyOnnxAgent({
    policy: playing.policy,
    rng
  }),
  onProgress: args.progress === "true"
    ? (progress) => {
      console.error([
        `[pseudo-fixed-exchange] states=${progress.completedSourceStates}/${progress.requestedSourceStates}`,
        `groups=${progress.completedFixedThirteenGroups ?? 0}/${progress.requestedFixedThirteenGroups ?? groups}`,
        `samples=${progress.sampleCount}`,
        `seed=${progress.currentSeed}`,
        `attempts=${progress.dealAttempts}`,
        `shards=${progress.completedShards}`
      ].join(" "));
    }
    : undefined
});

console.log(JSON.stringify({
  outputDirectory: result.outputDirectory,
  groups,
  repeats,
  diagnosticOnly: args.diagnosticOnly === "true",
  states: result.manifest.sourceStateCount,
  samples: result.manifest.sampleCount,
  invariantFailures: result.manifest.summary.invariantFailureCount,
  pseudoFixedThirteen: result.manifest.summary.pseudoFixedThirteen,
  ruleBasedMarginRegret: result.manifest.summary.ruleBasedMarginRegret,
  ruleBasedRelativeRewardRegret: result.manifest.summary.ruleBasedRelativeRewardRegret
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
  console.error("usage: node scripts/generate-pseudo-fixed-exchange-counterfactual-dataset.mjs --groups 50 --repeats 10 --start-seed 440000000 --output /tmp/issue440-pseudo-fixed --progress");
  process.exit(2);
}
