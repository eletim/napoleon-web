import { describe, expect, it } from "vitest";
import { RuleBasedAgent } from "../ruleBasedAgent.js";
import { RandomAgent } from "../randomAgent.js";
import { runAutomatedGame } from "./runAutomatedGame.js";
import type { RunAutomatedGameOptions } from "./types.js";

describe("runAutomatedGame", () => {
  it("runs a five-agent game to completion", async () => {
    const record = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });

    expect(record.schemaVersion).toBe(1);
    expect(record.seed).toBe(12345);
    expect(record.playerIds).toEqual(["player-0", "player-1", "player-2", "player-3", "player-4"]);
    expect(record.decisions.length).toBeGreaterThan(0);
    expect(record.result.winner).toMatch(/^(napoleon-team|alliance)$/);
  });

  it("replays the same seed and agent setup exactly", async () => {
    const createOptions = (): RunAutomatedGameOptions => ({
      seed: 12345,
      createAgent: ({ playerIndex, rng }) =>
        playerIndex === 0 ? new RandomAgent(rng) : new RuleBasedAgent(rng)
    });

    const first = await runAutomatedGame(createOptions());
    const second = await runAutomatedGame(createOptions());

    expect(second).toEqual(first);
  });

  it("deals different initial hands for known different seeds", async () => {
    const records = await Promise.all(
      [1, 2, 3].map((seed) =>
        runAutomatedGame({
          seed,
          createAgent: ({ rng }) => new RuleBasedAgent(rng)
        })
      )
    );
    const serializedHands = records.map((record) => JSON.stringify(record.initialHands));

    expect(new Set(serializedHands).size).toBe(serializedHands.length);
  });

  it("records only actions that were legal at the decision point", async () => {
    const record = await runAutomatedGame({
      seed: 777,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });

    for (const decision of record.decisions) {
      expect(decision.legalActions).toContainEqual(decision.action);
    }
  });

  it("keeps complete-information hand labels separate from observations", async () => {
    const record = await runAutomatedGame({
      seed: 54321,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });

    for (const decision of record.decisions) {
      for (const player of decision.observation.view.players) {
        expect(decision.actualHands[player.id]?.length).toBe(player.handCount);
        expect(decision.handCounts[player.id]).toBe(player.handCount);

        if (player.id === decision.playerId) {
          expect(player.hand?.map((card) => card.id)).toEqual(decision.actualHands[player.id]);
        } else {
          expect(player).not.toHaveProperty("hand");
        }
      }
    }
  });
});
