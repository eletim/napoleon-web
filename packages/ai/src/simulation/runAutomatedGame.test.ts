import { describe, expect, it } from "vitest";
import { createDeck, isJokerCard, isStandardCard } from "@napoleon/game-core";
import type { GameAction } from "@napoleon/game-core";
import type { Agent, PlayerObservation } from "../types.js";
import { RuleBasedAgent } from "../ruleBasedAgent.js";
import { RandomAgent } from "../randomAgent.js";
import { runAutomatedGame } from "./runAutomatedGame.js";
import type { ActualCardState } from "./types.js";
import type { RunAutomatedGameOptions } from "./types.js";

class DuplicateDiscardAgent implements Agent {
  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    if (observation.view.phase === "exchanging") {
      const self = observation.view.players.find((player) => player.id === observation.playerId);

      if (self?.hand === undefined || self.hand.length < 2) {
        throw new Error("Expected exchange hand for duplicate discard test.");
      }

      return {
        type: "discard-cards",
        playerId: observation.playerId,
        cardIds: [self.hand[0].id, self.hand[0].id, self.hand[1].id]
      };
    }

    if (observation.view.phase === "choosing-adjutant") {
      return {
        type: "choose-adjutant",
        playerId: observation.playerId,
        cardId: "spades-A"
      };
    }

    const passAction = observation.legalActions.find((action) => action.type === "pass");
    const fallback = observation.legalActions[0];

    if (passAction !== undefined) {
      return passAction;
    }

    if (fallback === undefined) {
      throw new Error("Expected legal action.");
    }

    return fallback;
  }
}

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

  it("accepts uint32 seed boundary values", async () => {
    const lower = await runAutomatedGame({
      seed: 0,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const upper = await runAutomatedGame({
      seed: 0xffffffff,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });

    expect(lower.seed).toBe(0);
    expect(upper.seed).toBe(0xffffffff);
    expect(lower.decisions.length).toBeGreaterThan(0);
    expect(upper.decisions.length).toBeGreaterThan(0);
  });

  it("rejects seeds outside the uint32 range", async () => {
    const invalidSeeds = [-1, 0x100000000, 1.5, Number.POSITIVE_INFINITY, "123", {}];

    for (const seed of invalidSeeds) {
      await expect(
        runAutomatedGame({
          seed: seed as number,
          createAgent: ({ rng }) => new RuleBasedAgent(rng)
        })
      ).rejects.toThrow("seed must be an integer between 0 and 4294967295.");
    }
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

  it("includes only prior public bidding actions in each observation", async () => {
    const record = await runAutomatedGame({
      seed: 777,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });

    record.decisions.forEach((decision) => {
      expect(decision.observation.publicActionHistory).toEqual(
        record.decisions
          .filter(
            (previous) =>
              previous.step < decision.step &&
              previous.phase === "bidding" &&
              (previous.action.type === "bid" || previous.action.type === "pass")
          )
          .map((previous) => ({
            step: previous.step,
            playerId: previous.playerId,
            phase: previous.phase,
            action: previous.action
          }))
      );
      expect(
        decision.observation.publicActionHistory?.every(
          (record) => record.action.type === "bid" || record.action.type === "pass"
        )
      ).toBe(true);
    });
  });

  it("shows buried-card events once and clears them before later decisions", async () => {
    const record = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const eventDecisionIndex = record.decisions.findIndex(
      (decision) =>
        decision.phase === "playing" &&
        decision.observation.view.latestEvent?.type === "buried-cards-resolved"
    );

    expect(eventDecisionIndex).toBeGreaterThanOrEqual(0);
    expect(record.decisions[eventDecisionIndex].observation.view.latestEvent).toMatchObject({
      type: "buried-cards-resolved"
    });

    for (const decision of record.decisions.slice(eventDecisionIndex + 1)) {
      expect(decision.observation.view.latestEvent).toBeNull();
    }

    const laterPlayingDecisions = record.decisions.filter(
      (decision) => decision.phase === "playing" && decision.trickNumber >= 2
    );

    expect(laterPlayingDecisions.length).toBeGreaterThan(0);

    for (const decision of laterPlayingDecisions) {
      expect(decision.observation.view.latestEvent).toBeNull();
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

  it("rejects duplicate discard card ids before applying the action", async () => {
    await expect(
      runAutomatedGame({
        seed: 0,
        createAgent: () => new DuplicateDiscardAgent()
      })
    ).rejects.toThrow("Automated agent selected an illegal action");
  });

  it("rejects duplicate player ids before creating agents", async () => {
    let createAgentCount = 0;

    await expect(
      runAutomatedGame({
        seed: 1,
        playerIds: ["player-0", "player-1", "player-2", "player-3", "player-3"],
        createAgent: ({ rng }) => {
          createAgentCount += 1;
          return new RuleBasedAgent(rng);
        }
      })
    ).rejects.toThrow("playerIds must be unique.");
    expect(createAgentCount).toBe(0);
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
