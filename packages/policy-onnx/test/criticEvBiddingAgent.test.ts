import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import { applyAction, createInitialGame, createPlayerView, orumaCardId } from "@napoleon/game-core";
import type { Card, GameState, PlayerState } from "@napoleon/game-core";
import {
  CriticEvBiddingAgent,
  type PolicyCriticValueModel
} from "../src/index.js";

class ConstantCritic implements PolicyCriticValueModel {
  public lastBatchSize = 0;

  constructor(private readonly value: number) {}

  async predictValuesBatch(
    modelInputs: readonly (Float32Array | readonly number[])[]
  ): Promise<readonly number[]> {
    this.lastBatchSize = modelInputs.length;
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
    const clubs13Evaluation = evaluations.find((evaluation) =>
      evaluation.action.type === "bid" &&
      evaluation.action.suit === "clubs" &&
      evaluation.action.targetPointCards === 13
    );
    expect(clubs13Evaluation).toMatchObject({
      baseWinRateEquivalent: 0.5,
      effectiveNapoleonWinRate: 0.6
    });
    expect(clubs13Evaluation?.expectedValue).toBeCloseTo(6.6, 6);
    expect(await agent.selectAction(observation)).toEqual(action);
  });

  it("evaluates non-terminal pass with no highest bid as a deferred pass without critic input", async () => {
    const state = createInitialGame({ rng: () => 0 });
    const observation = createObservation(state);
    const critic = new ConstantCritic(0);
    const agent = new CriticEvBiddingAgent({
      critic,
      delegateAgent: new RuleBasedAgent(() => 0)
    });

    const evaluations = await agent.evaluateLegalBiddingActions(observation);
    const passEvaluation = evaluations.find((evaluation) => evaluation.action.type === "pass");

    expect(critic.lastBatchSize).toBe(observation.legalActions.length - 1);
    expect(passEvaluation).toMatchObject({
      role: "deferred-pass",
      criticValue: 0,
      expectedValue: 0,
      contract: null,
      calledAdjutantCardId: null
    });
  });

  it("evaluates the terminal fifth pass with no highest bid as the all-pass payoff", async () => {
    const state = createStateBeforeTerminalAllPass();
    const observation = createObservation(state);
    const critic = new ConstantCritic(0);
    const agent = new CriticEvBiddingAgent({
      critic,
      delegateAgent: new RuleBasedAgent(() => 0)
    });

    const evaluations = await agent.evaluateLegalBiddingActions(observation);
    const passEvaluation = evaluations.find((evaluation) => evaluation.action.type === "pass");

    expect(critic.lastBatchSize).toBe(observation.legalActions.length - 1);
    expect(passEvaluation).toMatchObject({
      role: "all-pass-other",
      criticValue: 0,
      expectedValue: -1,
      contract: null,
      calledAdjutantCardId: null
    });
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

  it("evaluates pass as adjutant EV when self holds Oruma against the current opponent bid", async () => {
    const state = forcePlayerToHoldCard(createStateWithOpponentMaxBid(), "player-0", orumaCardId);
    const observation = createObservation(state);
    const agent = new CriticEvBiddingAgent({
      critic: new ConstantCritic(0.2),
      delegateAgent: new RuleBasedAgent(() => 0)
    });

    const evaluations = await agent.evaluateLegalBiddingActions(observation);

    expect(observation.legalActions).toEqual([{ type: "pass", playerId: "player-0" }]);
    expect(evaluations[0]).toMatchObject({
      role: "adjutant",
      baseWinRateEquivalent: 0.6,
      effectiveNapoleonWinRate: 0.6
    });
    expect(evaluations[0].expectedValue).toBeCloseTo(7.2, 6);
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

function createStateBeforeTerminalAllPass(): GameState {
  let state = createInitialGame({ rng: () => 0 });
  state = applyAction(state, { type: "pass", playerId: "player-0" });
  state = applyAction(state, { type: "pass", playerId: "player-1" });
  state = applyAction(state, { type: "pass", playerId: "player-2" });
  state = applyAction(state, { type: "pass", playerId: "player-3" });
  return state;
}

function forcePlayerToHoldCard(state: GameState, playerId: string, cardId: string): GameState {
  const owner = state.players.find((player) => player.hand.some((card) => card.id === cardId));
  if (owner?.id === playerId) {
    return state;
  }

  const target = state.players.find((player) => player.id === playerId);
  if (target === undefined) {
    throw new Error(`Player ${playerId} is not present in state.`);
  }

  const targetSwapCard = target.hand.find((card) => card.id !== cardId);
  const forcedCard =
    owner?.hand.find((card) => card.id === cardId) ??
    state.unusedCards.find((card) => card.id === cardId);
  if (targetSwapCard === undefined || forcedCard === undefined) {
    throw new Error("Cannot swap forced card into target hand.");
  }

  const players = state.players.map((player): PlayerState => {
    if (player.id === playerId) {
      return {
        ...player,
        hand: replaceCard(player.hand, targetSwapCard.id, forcedCard)
      };
    }
    if (owner !== undefined && player.id === owner.id) {
      return {
        ...player,
        hand: replaceCard(player.hand, forcedCard.id, targetSwapCard)
      };
    }
    return player;
  });

  const unusedCards = owner === undefined
    ? replaceCard(state.unusedCards, forcedCard.id, targetSwapCard)
    : state.unusedCards;

  return { ...state, players, unusedCards };
}

function replaceCard(hand: readonly Card[], cardId: string, replacement: Card): readonly Card[] {
  return hand.map((card) => card.id === cardId ? replacement : card);
}
