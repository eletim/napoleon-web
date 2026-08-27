import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { createDeck } from "@napoleon/game-core";
import {
  FIXED_HAND_PASS_OUTCOME_DATASET_SAMPLE_TYPE,
  aggregateFixedHandPassRollouts,
  createFixedHandInitialState,
  generateFixedHandPassOutcomeDataset,
  stableUint32
} from "../src/index.js";
import type { FixedHandSpec } from "../src/index.js";

describe("generateFixedHandPassOutcomeDataset", () => {
  it("keeps the fixed candidate hand and deterministic reshuffle invariant", () => {
    const handIds = createDeck().slice(0, 10).map((card) => card.id);
    const first = createFixedHandInitialState({ seed: 414, candidateSeatIndex: 1, handIds });
    const second = createFixedHandInitialState({ seed: 414, candidateSeatIndex: 1, handIds });
    expect(first.players[1].hand.map((card) => card.id)).toEqual(handIds);
    expect(second.players.map((player) => player.hand.map((card) => card.id)))
      .toEqual(first.players.map((player) => player.hand.map((card) => card.id)));
    expect(stableUint32("pass-414")).toBe(stableUint32("pass-414"));
  });

  it("aggregates empirical role frequencies and role-conditioned margins", () => {
    const sample = aggregateFixedHandPassRollouts({
      spec: fixedSpec(),
      splitHint: null,
      rows: [
        row("citizen", -2, false),
        row("citizen", 1, true),
        row("adjutant", 3, true),
        row("no-contract", null, null)
      ]
    });
    expect(sample.sampleType).toBe(FIXED_HAND_PASS_OUTCOME_DATASET_SAMPLE_TYPE);
    expect(sample.nCitizen).toBe(2);
    expect(sample.nAdjutant).toBe(1);
    expect(sample.nNoContract).toBe(1);
    expect(sample.qTeacher).toBeCloseTo(1 / 3);
    expect(sample.citizenMargin.empiricalMarginMean).toBeCloseTo(-0.5);
    expect(sample.citizenMargin.empiricalMarginStd).toBeCloseTo(1.5);
    expect(sample.citizenMargin.empiricalTargetMean).toBe(13);
    expect(sample.adjutantMargin.empiricalWinRate).toBe(1);
  });

  it("generates deterministic grouped PASS samples", async () => {
    await withTempDir(async (directory) => {
      const spec = fixedSpec();
      const first = await generateFixedHandPassOutcomeDataset({
        outputDirectory: join(directory, "first"),
        handCount: 1,
        repeats: 2,
        randomSeed: 414,
        reservedHands: [spec],
        gamesPerShard: 10
      });
      const second = await generateFixedHandPassOutcomeDataset({
        outputDirectory: join(directory, "second"),
        handCount: 1,
        repeats: 2,
        randomSeed: 414,
        reservedHands: [spec],
        gamesPerShard: 10
      });
      expect(second.samples).toEqual(first.samples);
      expect(first.samples).toHaveLength(1);
      expect(first.samples[0].rolloutCount).toBe(2);
      const manifest = JSON.parse(
        await readFile(join(directory, "first", "manifest.json"), "utf8")
      ) as { teacher: { qLabel: string }; handCount: number };
      expect(manifest.teacher.qLabel).toBe("qTeacher");
      expect(manifest.handCount).toBe(1);
    });
  });
});

function fixedSpec(): FixedHandSpec {
  return {
    fixedHandId: "fixture-pass-hand",
    handIds: createDeck().slice(0, 10).map((card) => card.id),
    candidateSeatIndex: 0,
    sourceStateKey: "fixture",
    sourceSeed: 1,
    sourceBiddingStep: 1,
    strongestSuit: "spades",
    strongestSuitScore: 10,
    reason: "fixture",
    actions: []
  };
}

function row(finalRole: string, contractMargin: number | null, contractSuccess: boolean | null) {
  return {
    contractMargin,
    contractSuccess,
    finalDeclaredTarget: contractMargin === null ? null : 13,
    resultType: contractMargin === null ? "all-pass" as const : "standard" as const,
    finalRole,
    modelInput: [0, 1, 2],
    legalBidMask: [1, 1]
  };
}

async function withTempDir(callback: (directory: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), "napoleon-pass-outcome-test-"));
  try {
    await callback(directory);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}
