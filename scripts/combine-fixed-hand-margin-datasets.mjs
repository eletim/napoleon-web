#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rename, rm, writeFile } from "node:fs/promises";
import { basename, dirname, join, relative, resolve, sep } from "node:path";

const args = parseArgs(process.argv.slice(2));
if (!args.output || !args.inputs) {
  usage("missing --output or --inputs");
}

const output = resolve(args.output);
const inputs = args.inputs.split(",").map((value) => resolve(value.trim())).filter(Boolean);
const rowsPerShard = numberArg(args.rowsPerShard, 5000, "--rows-per-shard");
if (inputs.length < 2) usage("--inputs must contain at least two comma-separated dataset directories");
for (const input of inputs) {
  if (pathsOverlap(output, input)) {
    usage(`--output must not overlap an input dataset: ${output} vs ${input}`);
  }
}

const rows = [];
const inputManifests = [];
for (const input of inputs) {
  const manifest = JSON.parse(await readFile(join(input, "manifest.json"), "utf8"));
  inputManifests.push({
    path: input,
    sampleType: manifest.sampleType,
    pairCount: manifest.pairCount,
    rolloutCount: manifest.rolloutCount,
    summary: manifest.summary ?? null
  });
  for (const shard of manifest.shards ?? []) {
    const text = await readFile(join(input, shard.file), "utf8");
    for (const line of text.split(/\n/)) {
      if (line.trim()) rows.push(JSON.parse(line));
    }
  }
}

const outputParent = dirname(output);
await mkdir(outputParent, { recursive: true });
const staging = await mkdtemp(join(outputParent, `.${basename(output)}.`));
let replacedOutput = false;
const shards = [];
try {
  for (let start = 0; start < rows.length; start += rowsPerShard) {
    const shardRows = rows.slice(start, start + rowsPerShard);
    const file = `shard-${String(shards.length).padStart(5, "0")}.jsonl`;
    const body = shardRows.map((row) => `${JSON.stringify(row)}\n`).join("");
    await writeFile(join(staging, file), body, "utf8");
    shards.push({
      file,
      startSeed: 0,
      endSeed: 0,
      gameCount: shardRows.length,
      sampleCount: shardRows.length,
      byteLength: Buffer.byteLength(body, "utf8"),
      sha256: createHash("sha256").update(body).digest("hex")
    });
  }

  const manifest = {
    datasetSchemaVersion: 1,
    generatorVersion: 1,
    format: "jsonl-shards-v1",
    sampleType: "mixed-fixed-hand-margin-sample",
    sampleSchemaVersion: 1,
    teacher: {
      id: "mixed-opening-history-consistent-raise-margin-v1",
      primaryLabel: "empiricalMarginMean",
      stdLabel: "empiricalMarginStd",
      winRateLabel: "empiricalWinRate"
    },
    pairCount: rows.length,
    fixedHandCount: new Set(rows.map((row) => row.fixedHandId)).size,
    uniqueRaiseStateCount: new Set(
      rows.filter((row) => row.sourceStateKey).map((row) => row.sourceStateKey)
    ).size,
    rolloutCount: rows.reduce((sum, row) => sum + Number(row.rolloutCount ?? 0), 0),
    inputs: inputManifests,
    shardCount: shards.length,
    shards
  };
  await writeFile(join(staging, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  await writeFile(join(staging, "summary.json"), `${JSON.stringify({
    pairCount: manifest.pairCount,
    fixedHandCount: manifest.fixedHandCount,
    uniqueRaiseStateCount: manifest.uniqueRaiseStateCount,
    rolloutCount: manifest.rolloutCount,
    inputCount: inputs.length
  }, null, 2)}\n`, "utf8");

  await rm(output, { recursive: true, force: true });
  await rename(staging, output);
  replacedOutput = true;

  console.log(JSON.stringify({
    output,
    pairCount: manifest.pairCount,
    fixedHandCount: manifest.fixedHandCount,
    uniqueRaiseStateCount: manifest.uniqueRaiseStateCount,
    rolloutCount: manifest.rolloutCount
  }, null, 2));
} finally {
  if (!replacedOutput) {
    await rm(staging, { recursive: true, force: true });
  }
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

function numberArg(value, fallback, name) {
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) usage(`${name} must be a positive integer`);
  return parsed;
}

function pathsOverlap(left, right) {
  return left === right || isSubpath(left, right) || isSubpath(right, left);
}

function isSubpath(parent, child) {
  const value = relative(parent, child);
  return value !== "" && !value.startsWith("..") && !value.startsWith(sep);
}

function usage(message) {
  console.error(`error: ${message}`);
  console.error("usage: node scripts/combine-fixed-hand-margin-datasets.mjs --output /tmp/mixed --inputs /tmp/opening,/tmp/raise");
  process.exit(1);
}
