import { describe, expect, it } from "vitest";
import { createBalancedAssignments } from "../src/finalMixedEvaluation.js";

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
});
