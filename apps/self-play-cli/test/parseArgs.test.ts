import { describe, expect, it } from "vitest";
import { parseArgs } from "../src/parseArgs.js";

describe("parseArgs", () => {
  it("parses valid arguments", () => {
    expect(parseArgs([
      "--start-seed",
      "0",
      "--games",
      "10",
      "--output",
      "./datasets/example",
      "--games-per-shard",
      "5"
    ])).toEqual({
      type: "options",
      options: {
        startSeed: 0,
        games: 10,
        output: "./datasets/example",
        gamesPerShard: 5
      }
    });
  });

  it("defaults games-per-shard to 100", () => {
    expect(parseArgs([
      "--start-seed",
      "1",
      "--games",
      "2",
      "--output",
      "./out"
    ])).toEqual({
      type: "options",
      options: {
        startSeed: 1,
        games: 2,
        output: "./out",
        gamesPerShard: 100
      }
    });
  });

  it("accepts uint32 max when generating one game", () => {
    expect(parseArgs([
      "--start-seed",
      "4294967295",
      "--games",
      "1",
      "--output",
      "./out"
    ])).toMatchObject({
      type: "options",
      options: {
        startSeed: 4294967295,
        games: 1
      }
    });
  });

  it("prints help", () => {
    expect(parseArgs(["--help"])).toMatchObject({
      type: "help"
    });
  });

  it.each([
    [["--games", "1", "--output", "./out"], "--start-seed is required"],
    [["--start-seed", "0", "--output", "./out"], "--games is required"],
    [["--start-seed", "0", "--games", "1"], "--output is required"],
    [["--start-seed", "-1", "--games", "1", "--output", "./out"], "--start-seed must be an integer"],
    [["--start-seed", "4294967296", "--games", "1", "--output", "./out"], "--start-seed must be an integer between"],
    [["--start-seed", "0", "--games", "0", "--output", "./out"], "--games must be a positive integer"],
    [["--start-seed", "0", "--games", "1.5", "--output", "./out"], "--games must be an integer"],
    [["--start-seed", "4294967290", "--games", "10", "--output", "./out"], "Seed range exceeds uint32"],
    [["--start-seed", "0", "--games", "1", "--output", "./out", "--games-per-shard", "0"], "--games-per-shard must be a positive integer"],
    [["--start-seed", "0", "--games", "1", "--output", "./out", "--parallel", "2"], "Unknown argument: --parallel"],
    [["--start-seed", "0", "--start-seed", "1", "--games", "1", "--output", "./out"], "Duplicate argument: --start-seed"],
    [["--start-seed", "0", "--games", "1", "--output"], "--output requires a value"],
    [["--start-seed", "0", "--games", "1", "--output", ""], "--output must be a non-empty path"]
  ])("rejects invalid arguments %#", (argv, message) => {
    expect(() => parseArgs(argv)).toThrow(message);
  });
});
