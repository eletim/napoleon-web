import { describe, expect, it } from "vitest";
import { createRelativePlayerOrder, getRelativePlayerIndex } from "../src/index.js";

const playerIds = ["player-0", "player-1", "player-2", "player-3", "player-4"] as const;

describe("relative player indexing", () => {
  it("rotates all five seats around the acting player", () => {
    expect(createRelativePlayerOrder(playerIds, "player-0")).toEqual([
      "player-0",
      "player-1",
      "player-2",
      "player-3",
      "player-4"
    ]);
    expect(createRelativePlayerOrder(playerIds, "player-1")).toEqual([
      "player-1",
      "player-2",
      "player-3",
      "player-4",
      "player-0"
    ]);
    expect(createRelativePlayerOrder(playerIds, "player-2")).toEqual([
      "player-2",
      "player-3",
      "player-4",
      "player-0",
      "player-1"
    ]);
    expect(createRelativePlayerOrder(playerIds, "player-3")).toEqual([
      "player-3",
      "player-4",
      "player-0",
      "player-1",
      "player-2"
    ]);
    expect(createRelativePlayerOrder(playerIds, "player-4")).toEqual([
      "player-4",
      "player-0",
      "player-1",
      "player-2",
      "player-3"
    ]);
  });

  it("looks up relative player indices", () => {
    const relativePlayerIds = createRelativePlayerOrder(playerIds, "player-2");

    expect(getRelativePlayerIndex(relativePlayerIds, "player-2")).toBe(0);
    expect(getRelativePlayerIndex(relativePlayerIds, "player-0")).toBe(3);
  });

  it("rejects invalid player sets", () => {
    expect(() => createRelativePlayerOrder(playerIds.slice(0, 4), "player-0")).toThrow(
      "Expected exactly 5 player ids"
    );
    expect(() =>
      createRelativePlayerOrder(["player-0", "player-1", "player-2", "player-3", "player-3"], "player-0")
    ).toThrow("Player ids must be unique");
    expect(() => createRelativePlayerOrder(playerIds, "missing")).toThrow(
      "selfPlayerId is not included"
    );
    expect(() => getRelativePlayerIndex(playerIds, "missing")).toThrow(
      "playerId is not included"
    );
  });
});
