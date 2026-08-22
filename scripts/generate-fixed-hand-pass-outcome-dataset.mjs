#!/usr/bin/env node
import { readFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import {
  createRandomFixedHands,
  generateFixedHandPassOutcomeDataset
} from "../packages/training-data/dist/index.js";

const args = parseArgs(process.argv.slice(2));
if (!args.output) usage("missing --output");

const handCount = numberArg(args.hands, 1000, "--hands");
const repeats = numberArg(args.repeats, 50, "--repeats");
const randomSeed = numberArg(args.seed, 414000, "--seed");
const gamesPerShard = numberArg(args.gamesPerShard, 1000, "--games-per-shard");
const reservedHands = args.reservedHands
  ? await loadReservedHands(args.reservedHands, Number(args.reservedLimit ?? 20))
  : [];

const result = await generateFixedHandPassOutcomeDataset({
  outputDirectory: args.output,
  handCount,
  repeats,
  randomSeed,
  gamesPerShard,
  reservedHands,
  reserveHandsForFinal: reservedHands.length > 0,
  sourceCommit: args.sourceCommit ?? gitCommitOrNull(),
  onProgress: (progress) => {
    if (progress.completedHands === progress.totalHands || progress.completedHands % 25 === 0) {
      console.error(`hands ${progress.completedHands}/${progress.totalHands}; rollouts ${progress.completedRollouts}/${progress.totalRollouts}`);
    }
  }
});

console.log(JSON.stringify({
  outputDirectory: result.outputDirectory,
  handCount: result.manifest.handCount,
  rolloutCount: result.manifest.rolloutCount,
  repeats,
  summary: result.manifest.summary
}, null, 2));

async function loadReservedHands(path, limit) {
  const raw = JSON.parse(await readFile(path, "utf8"));
  const selections = Array.isArray(raw.selections) ? raw.selections.slice(0, limit) : [];
  return selections.map((selection, index) => {
    const generated = createRandomFixedHands({
      handCount: 1,
      actionCountPerHand: 1,
      randomSeed: 414000 + index,
      candidateSeatIndex: selection.candidateSeatIndex ?? 0
    })[0];
    const handIds = selection.handIds;
    return {
      fixedHandId: `${index}:${hashText(handIds.join(","))}`,
      handIds,
      candidateSeatIndex: selection.candidateSeatIndex ?? 0,
      sourceStateKey: selection.sourceStateKey ?? null,
      sourceSeed: selection.sourceSeed ?? null,
      sourceBiddingStep: selection.biddingStep ?? null,
      strongestSuit: selection.strongestSuit ?? generated.strongestSuit,
      strongestSuitScore: selection.strongestSuitScore ?? generated.strongestSuitScore,
      reason: `issue409-${selection.reason ?? "reserved"}`,
      actions: generated.actions
    };
  });
}

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
  if (!Number.isInteger(parsed) || parsed <= 0) usage(`${name} must be positive`);
  return parsed;
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
  for (let index = 0; index < text.length; index += 1) {
    hash = ((hash << 5) - hash + text.charCodeAt(index)) | 0;
  }
  return Math.abs(hash).toString(16).padStart(8, "0");
}

function usage(message) {
  console.error(`error: ${message}`);
  console.error("usage: node scripts/generate-fixed-hand-pass-outcome-dataset.mjs --output /tmp/ds --hands 1000 --repeats 50");
  process.exit(1);
}
