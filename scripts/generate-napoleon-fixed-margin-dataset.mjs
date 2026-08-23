#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import {
  createRandomFixedHands,
  generateNapoleonFixedMarginDataset
} from "../packages/training-data/dist/index.js";

const args = parseArgs(process.argv.slice(2));

if (!args.output) {
  usage("missing --output");
}

const pairCount = numberArg(args.pairs, 1000, "--pairs");
const repeats = numberArg(args.repeats, 50, "--repeats");
const randomSeed = numberArg(args.seed, 423000, "--seed");
const actionCountPerHand = numberArg(args.actionsPerHand, 4, "--actions-per-hand");
const decisionContextMode = stringArg(
  args.decisionContextMode ?? args.contextMode,
  "opening",
  "--decision-context-mode",
  ["opening", "raise", "mixed"]
);
const gamesPerShard = numberArg(args.gamesPerShard, 1000, "--games-per-shard");
const reservedHands = args.reservedHands
  ? await loadReservedHands(args.reservedHands, Number(args.reservedLimit ?? 20))
  : [];
const sourceCommit = args.sourceCommit ?? gitCommitOrNull();

const result = await generateNapoleonFixedMarginDataset({
  outputDirectory: args.output,
  pairCount,
  repeats,
  randomSeed,
  actionCountPerHand,
  decisionContextMode,
  gamesPerShard,
  reservedHands,
  reserveHandsForFinal: reservedHands.length > 0,
  sourceCommit,
  onProgress: (progress) => {
    if (progress.completedPairs === progress.totalPairs || progress.completedPairs % 25 === 0) {
      console.error(
        `pairs ${progress.completedPairs}/${progress.totalPairs}; rollouts ${progress.completedRollouts}/${progress.totalRollouts}`
      );
    }
  }
});

console.log(JSON.stringify({
  outputDirectory: result.outputDirectory,
  pairCount: result.manifest.pairCount,
  fixedHandCount: result.manifest.fixedHandCount,
  rolloutCount: result.manifest.rolloutCount,
  repeats,
  rawRolloutShards: result.manifest.rawRolloutShards.length,
  summary: result.manifest.summary
}, null, 2));

async function loadReservedHands(path, limit) {
  const raw = JSON.parse(await readFile(path, "utf8"));
  const selections = Array.isArray(raw.selections) ? raw.selections.slice(0, limit) : [];
  return selections.map((selection, index) => {
    const handIds = selection.handIds;
    const generated = createRandomFixedHands({
      handCount: 1,
      actionCountPerHand: 4,
      randomSeed: 423000 + index,
      candidateSeatIndex: selection.candidateSeatIndex ?? 0,
      decisionContextMode
    })[0];
    const actions = Array.isArray(selection.actions) && selection.actions.length > 0
      ? selection.actions
          .filter((action) => action.actionIndex > 0)
          .map((action) => ({
            actionIndex: action.actionIndex,
            targetPointCards: action.target ?? action.targetPointCards,
            suit: action.suit,
            decisionContext: action.decisionContext,
            currentBidTargetPointCards: action.currentBidTargetPointCards,
            currentBidSuit: action.currentBidSuit,
            currentBidderSeatIndex: action.currentBidderSeatIndex,
            consecutivePassCount: action.consecutivePassCount,
            biddingStep: action.biddingStep,
            label: action.label,
            sourceNnMu: action.sourceNn?.mu ?? null,
            sourceNnSigma: action.sourceNn?.sigma ?? null,
            sourceNnPWin: action.sourceNn?.gaussianWinProbability ?? null
          }))
      : generated.actions;
    return {
      fixedHandId: selection.fixedHandId ?? `${index}:${hashText(handIds.join(","))}`,
      handIds,
      candidateSeatIndex: selection.candidateSeatIndex ?? 0,
      sourceStateKey: selection.sourceStateKey ?? null,
      sourceSeed: selection.sourceSeed ?? null,
      sourceBiddingStep: selection.biddingStep ?? null,
      strongestSuit: selection.strongestSuit ?? generated.strongestSuit,
      strongestSuitScore: selection.strongestSuitScore ?? generated.strongestSuitScore,
      reason: `issue423-${selection.reason ?? "reserved"}`,
      actions
    };
  });
}

function parseArgs(values) {
  const result = {};
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value === "--") {
      continue;
    }
    if (!value.startsWith("--")) {
      usage(`unexpected argument ${value}`);
    }
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
  if (value === undefined) {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    usage(`${name} must be a positive integer`);
  }
  return parsed;
}

function stringArg(value, fallback, name, allowed) {
  const selected = value ?? fallback;
  if (!allowed.includes(selected)) {
    usage(`${name} must be one of ${allowed.join(", ")}`);
  }
  return selected;
}

function gitCommitOrNull() {
  try {
    return execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

function hashText(text) {
  let hash = 0;
  for (let index = 0; index < text.length; index++) {
    hash = ((hash << 5) - hash + text.charCodeAt(index)) | 0;
  }
  return Math.abs(hash).toString(16).padStart(8, "0");
}

function usage(message) {
  console.error(`error: ${message}`);
  console.error(
    "usage: node scripts/generate-napoleon-fixed-margin-dataset.mjs --output /tmp/ds --pairs 1000 --repeats 50 [--decision-context-mode opening|raise|mixed] [--reserved-hands /tmp/selected-hands.json]"
  );
  process.exit(1);
}
