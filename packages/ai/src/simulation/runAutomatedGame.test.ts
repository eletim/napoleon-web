import { describe, expect, it } from "vitest";
import { createDeck, isJokerCard, isStandardCard } from "@napoleon/game-core";
import { RuleBasedAgent } from "../ruleBasedAgent.js";
import { RandomAgent } from "../randomAgent.js";
import { runAutomatedGame } from "./runAutomatedGame.js";
import type { ActualCardState } from "./types.js";
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

  it("records all 53 card locations as complete-information labels", async () => {
    const record = await runAutomatedGame({
      seed: 777,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });

    expect(record.initialActualState.unusedCardIds).toHaveLength(3);
    expectCompleteCardState(record.initialActualState);

    for (const decision of record.decisions) {
      expectCompleteCardState(decision.actualState);
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

  it("keeps opponent hidden hands out of observations", async () => {
    const record = await runAutomatedGame({
      seed: 54321,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });

    for (const decision of record.decisions) {
      for (const player of decision.observation.view.players) {
        if (player.id === decision.playerId) {
          expect(player.hand?.map((card) => card.id)).toEqual(
            decision.actualState.hands[player.id]
          );
        } else {
          expect(player).not.toHaveProperty("hand");
        }
      }
    }
  });

});

function expectCompleteCardState(actualState: ActualCardState): void {
  const allCardIds = [
    ...Object.values(actualState.hands).flat(),
    ...actualState.unusedCardIds,
    ...actualState.excludedCardIds,
    ...Object.values(actualState.awardedPointCardIds).flat(),
    ...actualState.currentTrickCardIds,
    ...actualState.completedTrickCardIds
  ];
  const deck = createDeck();

  expect(allCardIds).toHaveLength(53);
  expect(new Set(allCardIds).size).toBe(53);
  expect(allCardIds.filter((cardId) => cardId === "joker")).toHaveLength(1);
  expect(allCardIds.filter((cardId) => cardId !== "joker")).toHaveLength(52);
  expect(new Set(allCardIds)).toEqual(new Set(deck.map((card) => card.id)));
  expect(deck.filter(isStandardCard)).toHaveLength(52);
  expect(deck.filter(isJokerCard)).toHaveLength(1);
}
