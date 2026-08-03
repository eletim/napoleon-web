import { describe, expect, it } from "vitest";
import type { PublicSimulationDecision } from "@napoleon/protocol";
import {
  createSimulationFilename,
  formatHandCounts,
  formatSimulationAction,
  getSimulationPlayerRole
} from "./simulationDisplay";

describe("simulation display helpers", () => {
  it("formats simulation actions for timeline rows", () => {
    expect(
      formatSimulationAction("player-2", {
        type: "bid",
        suit: "spades",
        targetPointCards: 15
      })
    ).toBe("player-2: スペード 15枚で入札");
    expect(formatSimulationAction("player-4", { type: "pass" })).toBe("player-4: パス");
    expect(
      formatSimulationAction("player-2", {
        type: "choose-adjutant",
        cardId: "spades-J"
      })
    ).toBe("player-2: J♠を副官指定");
    expect(
      formatSimulationAction("player-1", {
        type: "play-card",
        cardId: "hearts-10"
      })
    ).toBe("player-1: 10♥を出した");
    expect(
      formatSimulationAction("player-2", {
        type: "discard-cards",
        cardIds: ["clubs-2", "diamonds-3", "joker"]
      })
    ).toBe("player-2: 3枚を捨てた");
  });

  it("formats player roles from the final result", () => {
    const result = {
      winner: "napoleon-team",
      napoleonTeamPointCards: 13,
      alliancePointCards: 7,
      targetPointCards: 13,
      napoleonPlayerId: "player-2",
      adjutantPlayerId: "player-4"
    } as const;

    expect(getSimulationPlayerRole("player-2", result)).toBe("ナポレオン");
    expect(getSimulationPlayerRole("player-4", result)).toBe("副官");
    expect(getSimulationPlayerRole("player-1", result)).toBe("連合軍");
  });

  it("formats timeline hand counts and download filenames", () => {
    const decision = {
      handCounts: {
        "player-0": 10,
        "player-1": 9
      }
    } as unknown as PublicSimulationDecision;

    expect(formatHandCounts(decision)).toBe("player-0:10 / player-1:9");
    expect(createSimulationFilename(12345)).toBe("napoleon-simulation-seed-12345.json");
  });
});
