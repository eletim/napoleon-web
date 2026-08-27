import { describe, expect, it } from "vitest";
import {
  citizenStratumSummary,
  createBalancedAssignments
} from "../src/finalMixedEvaluation.js";

describe("final mixed COM-AI evaluation", () => {
  it("creates deterministic, exactly balanced assignments for every seat", () => {
    const first = createBalancedAssignments(100, 462202203);
    const second = createBalancedAssignments(100, 462202203);
    expect(first).toEqual(second);
    expect(createBalancedAssignments(100, 462202204)).not.toEqual(first);
    for (let seat = 0; seat < 5; seat += 1) {
      expect(first.filter((row) => row[seat] === "COM-AI")).toHaveLength(50);
    }
  });

  it("rejects a game count that cannot be exactly balanced", () => {
    expect(() => createBalancedAssignments(99, 1)).toThrow(/even integer/);
  });

  it("stratifies a focal Citizen by enemy composition and other Citizen AI count", () => {
    const game = {
      napoleonPolicy: "COM-AI",
      adjutantPolicy: "COM-RuleBase",
      seats: [
        { seat: 0, policy: "COM-AI", role: "Citizen", win: 1, relativeReward: 0.5 },
        { seat: 1, policy: "COM-RuleBase", role: "Citizen", win: 1, relativeReward: 0.2 },
        { seat: 2, policy: "COM-RuleBase", role: "Citizen", win: 1, relativeReward: 0.2 },
        { seat: 3, policy: "COM-AI", role: "Napoleon", win: 0, relativeReward: -1 },
        { seat: 4, policy: "COM-RuleBase", role: "Adjutant", win: 0, relativeReward: -1 }
      ]
    } as any;

    expect(citizenStratumSummary([game], {
      napoleonPolicy: "COM-AI",
      adjutantPolicy: "COM-RuleBase",
      otherCitizenAiCount: 0
    }, "COM-AI").win.n).toBe(1);
    expect(citizenStratumSummary([game], {
      napoleonPolicy: "COM-AI",
      adjutantPolicy: "COM-RuleBase",
      otherCitizenAiCount: 1
    }, "COM-RuleBase").win.n).toBe(2);
    expect(citizenStratumSummary([game], {
      napoleonPolicy: "COM-AI",
      adjutantPolicy: "COM-RuleBase",
      otherCitizenAiCount: 0
    }, "COM-RuleBase").win.n).toBe(0);
  });
});
