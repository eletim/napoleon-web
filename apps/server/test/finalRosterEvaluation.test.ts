import { describe, expect, it } from "vitest";
import {
  createRosterSchedule,
  describeCompositionTrend,
  describeFinalAssessment,
  describeRosterTrend,
  validateStartSeed
} from "../src/finalRosterEvaluation.js";

describe("final fixed-roster evaluation schedule", () => {
  it("creates six 1,000-game rosters with exactly balanced seat combinations", () => {
    const schedule = createRosterSchedule(462_600_000);
    expect(schedule).toHaveLength(6_000);
    expect(schedule[0].seed).toBe(462_600_000);
    expect(schedule.at(-1)?.seed).toBe(462_605_999);

    for (let aiCount = 0; aiCount <= 5; aiCount += 1) {
      const roster = schedule.filter((entry) => entry.aiCount === aiCount);
      expect(roster).toHaveLength(1_000);
      const counts = new Map<string, number>();
      for (const entry of roster) {
        expect(entry.seatPolicies.filter((policy) => policy === "AI")).toHaveLength(aiCount);
        const key = entry.aiSeats.join(",");
        counts.set(key, (counts.get(key) ?? 0) + 1);
      }
      const expectedCombinationCount = [1, 5, 10, 10, 5, 1][aiCount];
      expect(counts.size).toBe(expectedCombinationCount);
      expect(new Set(counts.values())).toEqual(new Set([1_000 / expectedCombinationCount]));
    }
  });

  it("is deterministic for a fixed seed and changes only seeds for a different start", () => {
    const first = createRosterSchedule(100);
    const second = createRosterSchedule(100);
    const shifted = createRosterSchedule(200);
    expect(first).toEqual(second);
    expect(shifted.map((entry) => entry.seatPolicies)).toEqual(
      first.map((entry) => entry.seatPolicies)
    );
    expect(shifted[0].seed).not.toBe(first[0].seed);
  });

  it("describes the observed roster trend instead of assuming monotonic results", () => {
    expect(describeRosterTrend([0.30, 0.35, 0.39, 0.45, 0.52, 0.62]))
      .toContain("単調に上がりました");
    expect(describeRosterTrend([0.30, 0.35, 0.34, 0.45, 0.52, 0.62]))
      .toContain("単調増加にはなりませんでした");
    expect(describeRosterTrend([0.30, 0.35, 0.35, 0.45, 0.52, 0.62]))
      .toContain("単調増加にはなりませんでした");
    expect(describeFinalAssessment([0.30, 0.35, 0.34, 0.45, 0.52, 0.62]))
      .toContain("一貫した単調関係は確認できませんでした");
  });

  it("derives composition claims from observed rates", () => {
    const ordered = {
      "napoleon-ai-adjutant-ai": { napoleonSideWinRate: 0.61 },
      "napoleon-ai-adjutant-rb": { napoleonSideWinRate: 0.50 },
      "napoleon-rb-adjutant-ai": { napoleonSideWinRate: 0.44 },
      "napoleon-rb-adjutant-rb": { napoleonSideWinRate: 0.34 }
    };
    expect(describeCompositionTrend(ordered)).toContain("AI+AIが4分類中最大");
    expect(describeCompositionTrend({
      ...ordered,
      "napoleon-rb-adjutant-ai": { napoleonSideWinRate: 0.65 }
    })).toContain("一方向の関係には揃いませんでした");
  });

  it("requires the complete 6,000-seed schedule to fit in uint32", () => {
    expect(validateStartSeed(0xffff_ffff - 5_999)).toBe(0xffff_ffff - 5_999);
    expect(() => validateStartSeed(0xffff_ffff - 5_998)).toThrow(/within 0/);
    expect(() => validateStartSeed(-1)).toThrow(/within 0/);
  });
});
