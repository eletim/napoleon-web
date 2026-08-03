import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import { createPlayingTrainingSamples } from "@napoleon/ai-observation";
import {
  validateGenerationOptions,
  validatePlayingTrainingSample
} from "../src/index.js";

describe("validation", () => {
  it("validates generation options and seed ranges", () => {
    expect(() => validateGenerationOptions({
      startSeed: 0,
      gameCount: 1,
      gamesPerShard: 1,
      outputDirectory: "out"
    })).not.toThrow();

    expect(() => validateGenerationOptions({
      startSeed: 4294967295,
      gameCount: 2,
      gamesPerShard: 1,
      outputDirectory: "out"
    })).toThrow("Seed range exceeds uint32");
  });

  it("rejects samples with non-json-safe values", async () => {
    const record = await runAutomatedGame({
      seed: 0,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const sample = structuredClone(createPlayingTrainingSamples(record)[0]);

    sample.observation.trickNumber = Number.NaN;

    expect(() => validatePlayingTrainingSample(sample, 0)).toThrow("NaN or Infinity");
  });
});
