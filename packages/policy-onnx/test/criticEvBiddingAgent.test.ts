import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import { applyAction, createInitialGame, createPlayerView } from "@napoleon/game-core";
import type { GameState } from "@napoleon/game-core";
import {
  CriticEvBiddingAgent,
  type PolicyCriticValueModel
} from "../src/index.js";

class ConstantCritic implements PolicyCriticValueModel {
  constructor(private readonly value: number) {}

  async predictValuesBatch(
    modelInputs: readonly (Float32Array | readonly number[])[]
  ): Promise<readonly number[]> {
    return modelInputs.map(() => this.value);
  }
}

describe("CriticEvBiddingAgent", () => {
  it("evaluates every legal bid and deterministically chooses the maximum EV legal action", async () => {
    const state = createInitialGame({ rng: () => 0 });
    const observation = createObservation(state);
    const agent = new CriticEvBiddingAgent({
      critic: new ConstantCritic(0),
      delegateAgent: new RuleBasedAgent(() => 0)
    });

    const evaluations = await agent.evaluateLegalBiddingActions(observation);
    const action = await agent.selectAction(observation);

    expect(evaluations).toHaveLength(observation.legalActions.length);
    expect(action).toEqual({
      type: "bid",
      playerId: "player-0",
      suit: "clubs",
      targetPointCards: 19
    });
    expect(await agent.selectAction(observation)).toEqual(action);
  });

  it("keeps pass legal after a maximum opposing bid and evaluates it as a citizen EV", async () => {
    const state = createStateWithOpponentMaxBid();
    const observation = createObservation(state);
    const agent = new CriticEvBiddingAgent({
      critic: new ConstantCritic(1),
      delegateAgent: new RuleBasedAgent(() => 0)
    });

    const evaluations = await agent.evaluateLegalBiddingActions(observation);
    const action = await agent.selectAction(observation);

    expect(observation.legalActions).toEqual([{ type: "pass", playerId: "player-0" }]);
    expect(action).toEqual({ type: "pass", playerId: "player-0" });
    expect(evaluations[0]).toMatchObject({
      role: "alliance",
      baseWinRateEquivalent: 1,
      effectiveNapoleonWinRate: 0,
      expectedValue: 7
    });
  });

  it("can complete an automated game without illegal actions", async () => {
    const record = await runAutomatedGame({
      seed: 193,
      createAgent: () =>
        new CriticEvBiddingAgent({
          critic: new ConstantCritic(0),
          delegateAgent: new RuleBasedAgent(() => 0)
        })
    });

    expect(record.result.winner === "napoleon-team" || record.result.winner === "alliance").toBe(true);
    expect(record.decisions.length).toBeGreaterThan(0);
  });
});

function createObservation(state: GameState) {
  const playerId = state.currentPlayerId;
  const view = createPlayerView(state, playerId);
  return {
    playerId,
    view,
    legalActions: view.legalActions
  };
}

function createStateWithOpponentMaxBid(): GameState {
  let state = createInitialGame({ rng: () => 0 });
  state = applyAction(state, { type: "pass", playerId: "player-0" });
  state = applyAction(state, {
    type: "bid",
    playerId: "player-1",
    suit: "spades",
    targetPointCards: 19
  });
  state = applyAction(state, { type: "pass", playerId: "player-2" });
  state = applyAction(state, { type: "pass", playerId: "player-3" });
  state = applyAction(state, { type: "pass", playerId: "player-4" });
  return state;
}
