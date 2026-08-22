import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { createDeck } from "@napoleon/game-core";
import {
  FIXED_HAND_BIDDING_MARGIN_DATASET_SAMPLE_TYPE,
  aggregateFixedHandBidRollouts,
  assertDeckConservation,
  createFixedHandInitialState,
  createRandomFixedHands,
  generateFixedHandBiddingMarginDataset,
  stableUint32,
  summarizeFixedHandBiddingMarginSamples
} from "../src/index.js";
import type { FixedHandBiddingActionSpec, FixedHandSpec } from "../src/index.js";

describe("generateFixedHandBiddingMarginDataset", () => {
  it("keeps candidate hand fixed while preserving a legal full deck", () => {
    const handIds = createDeck().slice(0, 10).map((card) => card.id);
    const state = createFixedHandInitialState({ seed: 411, candidateSeatIndex: 2, handIds });
    expect(state.players[2].hand.map((card) => card.id)).toEqual(handIds);
    assertDeckConservation(state);
  });

  it("deterministically reshuffles non-candidate cards by seed", () => {
    const handIds = createDeck().slice(4, 14).map((card) => card.id);
    const first = createFixedHandInitialState({ seed: 100, candidateSeatIndex: 0, handIds });
    const second = createFixedHandInitialState({ seed: 100, candidateSeatIndex: 0, handIds });
    const third = createFixedHandInitialState({ seed: 101, candidateSeatIndex: 0, handIds });
    expect(snapshotHands(second)).toEqual(snapshotHands(first));
    expect(snapshotHands(third)).not.toEqual(snapshotHands(first));
    expect(stableUint32("issue-411")).toBe(stableUint32("issue-411"));
  });

  it("aggregates N rollouts into empirical mean/std/win-rate labels", () => {
    const spec = fixedSpec();
    const action = spec.actions[0];
    const sample = aggregateFixedHandBidRollouts({
      spec,
      action,
      splitHint: null,
      rows: [
        rollout(-2, false),
        rollout(1, true),
        rollout(4, true)
      ]
    });
    expect(sample.sampleType).toBe(FIXED_HAND_BIDDING_MARGIN_DATASET_SAMPLE_TYPE);
    expect(sample.rolloutCount).toBe(3);
    expect(sample.empiricalMarginMean).toBe(1);
    expect(sample.empiricalMarginStd).toBeCloseTo(Math.sqrt(6));
    expect(sample.empiricalWinRate).toBeCloseTo(2 / 3);
    expect(sample.marginHistogram).toEqual({ "-2": 1, "1": 1, "4": 1 });
  });

  it("keeps N-repeat grouping at one sample per fixed hand/action pair", async () => {
    await withTempDir(async (directory) => {
      const spec = fixedSpec();
      const result = await generateFixedHandBiddingMarginDataset({
        outputDirectory: join(directory, "dataset"),
        pairCount: 1,
        repeats: 2,
        randomSeed: 411,
        reservedHands: [spec],
        gamesPerShard: 10
      });
      expect(result.samples).toHaveLength(1);
      expect(result.samples[0].fixedHandId).toBe(spec.fixedHandId);
      expect(result.samples[0].forcedActionIndex).toBe(spec.actions[0].actionIndex);
      expect(result.samples[0].rolloutCount).toBe(2);
      expect(result.manifest.rolloutCount).toBe(2);
      const manifest = JSON.parse(
        await readFile(join(directory, "dataset", "manifest.json"), "utf8")
      ) as { teacher: { primaryLabel: string }; shards: Array<{ file: string }> };
      expect(manifest.teacher.primaryLabel).toBe("empiricalMarginMean");
      expect(manifest.shards[0].file).toBe("shard-00000.jsonl");
    });
  });

  it("produces deterministic smoke datasets for the same seed", async () => {
    await withTempDir(async (directory) => {
      const first = await generateFixedHandBiddingMarginDataset({
        outputDirectory: join(directory, "first"),
        pairCount: 2,
        repeats: 1,
        randomSeed: 900,
        actionCountPerHand: 2
      });
      const second = await generateFixedHandBiddingMarginDataset({
        outputDirectory: join(directory, "second"),
        pairCount: 2,
        repeats: 1,
        randomSeed: 900,
        actionCountPerHand: 2
      });
      expect(second.samples).toEqual(first.samples);
      expect(summarizeFixedHandBiddingMarginSamples(first.samples).pairCount).toBe(2);
      expect(createRandomFixedHands({
        handCount: 2,
        actionCountPerHand: 2,
        randomSeed: 900,
        candidateSeatIndex: 0
      })).toEqual(createRandomFixedHands({
        handCount: 2,
        actionCountPerHand: 2,
        randomSeed: 900,
        candidateSeatIndex: 0
      }));
    });
  });
});

function fixedSpec(): FixedHandSpec {
  const handIds = createDeck().slice(0, 10).map((card) => card.id);
  const action: FixedHandBiddingActionSpec = {
    actionIndex: 1,
    targetPointCards: 13,
    suit: "spades",
    label: "spades13"
  };
  return {
    fixedHandId: "fixture-hand",
    handIds,
    candidateSeatIndex: 0,
    sourceStateKey: "fixture-state",
    sourceSeed: 1,
    sourceBiddingStep: 1,
    strongestSuit: "spades",
    strongestSuitScore: 10,
    reason: "issue409-fixture",
    actions: [action]
  };
}

function rollout(contractMargin: number, contractSuccess: boolean) {
  return {
    fixedHandId: "fixture-hand",
    contractMargin,
    contractSuccess,
    resultType: "standard" as const,
    finalRole: "napoleon",
    modelInput: [0, 1, 2],
    legalBidMask: [0, 1]
  };
}

function snapshotHands(state: ReturnType<typeof createFixedHandInitialState>): string[][] {
  return state.players.map((player) => player.hand.map((card) => card.id));
}

async function withTempDir(callback: (directory: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), "napoleon-fixed-hand-margin-test-"));
  try {
    await callback(directory);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}
