import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import { applyAction, createInitialGame, createPlayerView } from "@napoleon/game-core";
import type { GameState } from "@napoleon/game-core";
import {
  BIDDING_ACTION_COUNT,
  BIDDING_MODEL_INPUT_FEATURE_COUNT,
  CriticEvBiddingAgent,
  ISSUE427_T1_BIDDING_MARGIN_POLICY_ID,
  PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID,
  T1NapoleonEvBiddingAgent,
  createT1NapoleonEvBiddingDiagnostics,
  gaussianSuccessProbability,
  loadRepoManagedPlayingPolicyBenchmark,
  loadRepoManagedBiddingMarginPolicyBenchmark,
  napoleonRelativeEv,
  runIssue429T1BiddingRuntimeEvaluation,
  type BiddingMarginOnnxModel,
  type BiddingMarginOnnxPrediction,
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

class FakeMarginModel {
  constructor(
    private readonly means: readonly number[],
    private readonly sigma = 1
  ) {}

  async predict(): Promise<BiddingMarginOnnxPrediction> {
    return {
      mean: Float32Array.from(this.means),
      sigma: Float32Array.from(Array(BIDDING_ACTION_COUNT).fill(this.sigma)),
      logVariance: Float32Array.from(Array(BIDDING_ACTION_COUNT).fill(Math.log(this.sigma ** 2)))
    };
  }
}

class FailingMarginModel {
  async predict(): Promise<BiddingMarginOnnxPrediction> {
    throw new Error("model failed");
  }
}

describe("T1NapoleonEvBiddingAgent", () => {
  it("loads the repo-managed T1 margin ONNX artifact", async () => {
    const loaded = await loadRepoManagedBiddingMarginPolicyBenchmark(
      ISSUE427_T1_BIDDING_MARGIN_POLICY_ID,
      { inferenceDevice: "cpu" }
    );

    expect(loaded.artifact.id).toBe(ISSUE427_T1_BIDDING_MARGIN_POLICY_ID);
    expect(loaded.model.metadata.variant).toBe("M2");
    const prediction = await loaded.model.predict(new Float32Array(BIDDING_MODEL_INPUT_FEATURE_COUNT));
    expect(prediction.mean).toHaveLength(BIDDING_ACTION_COUNT);
    expect(prediction.sigma.every((value) => Number.isFinite(value) && value > 0)).toBe(true);
  });

  it("uses the Issue #429 Napoleon relative EV formula", () => {
    expect(napoleonRelativeEv(1, 16)).toBe(28);
    expect(napoleonRelativeEv(0, 16)).toBe(-12);
    expect(napoleonRelativeEv(0.3, 16)).toBeCloseTo(0, 12);
    expect(gaussianSuccessProbability(0, 1)).toBeCloseTo(0.5, 6);
  });

  it("passes opening when every legal bid is non-positive EV", async () => {
    const observation = createObservation(createInitialGame({ rng: () => 0 }));
    const diagnostics = createT1NapoleonEvBiddingDiagnostics();
    const action = await createAgent({
      means: Array(BIDDING_ACTION_COUNT).fill(-10),
      diagnostics
    }).selectAction(observation);

    expect(action).toEqual({ type: "pass", playerId: observation.playerId });
    expect(diagnostics.decisionRecords[0]).toMatchObject({
      kind: "opening",
      selectedActionType: "pass",
      passEv: 0
    });
  });

  it("selects the max-EV legal opening bid and ignores high-scoring illegal actions", async () => {
    const observation = createObservation(createInitialGame({ rng: () => 0 }));
    const means = Array(BIDDING_ACTION_COUNT).fill(-10);
    means[1] = 10;
    means[28] = 100;
    const restricted = {
      ...observation,
      legalActions: [
        { type: "pass" as const, playerId: observation.playerId },
        { type: "bid" as const, playerId: observation.playerId, suit: "spades" as const, targetPointCards: 13 }
      ]
    };

    const action = await createAgent({ means }).selectAction(restricted);

    expect(action).toEqual({
      type: "bid",
      playerId: observation.playerId,
      suit: "spades",
      targetPointCards: 13
    });
  });

  it("compares raise EV against the existing CriticEv PASS EV", async () => {
    const state = createRaiseState();
    const observation = createObservation(state);
    const passPreferredMeans = Array(BIDDING_ACTION_COUNT).fill(-10);
    const raisePreferredMeans = Array(BIDDING_ACTION_COUNT).fill(-10);
    raisePreferredMeans[28] = 10;

    await expect(createAgent({
      means: passPreferredMeans,
      criticValue: 1
    }).selectAction(observation)).resolves.toEqual({
      type: "pass",
      playerId: observation.playerId
    });

    await expect(createAgent({
      means: raisePreferredMeans,
      criticValue: 1
    }).selectAction(observation)).resolves.toEqual({
      type: "bid",
      playerId: observation.playerId,
      suit: "clubs",
      targetPointCards: 19
    });
  });

  it("is deterministic for identical model predictions", async () => {
    const observation = createObservation(createRaiseState());
    const means = Array(BIDDING_ACTION_COUNT).fill(-10);
    means[8] = 10;
    const agent = createAgent({ means, criticValue: -1 });

    await expect(agent.selectAction(observation)).resolves.toEqual(
      await agent.selectAction(observation)
    );
  });

  it("falls back to a legal RuleBased action when inference fails", async () => {
    const observation = createObservation(createInitialGame({ rng: () => 0 }));
    const diagnostics = createT1NapoleonEvBiddingDiagnostics();
    const agent = new T1NapoleonEvBiddingAgent({
      marginModel: new FailingMarginModel() as unknown as BiddingMarginOnnxModel,
      passEvAgent: new CriticEvBiddingAgent({ critic: new ConstantCritic(0) }),
      delegateAgent: new RuleBasedAgent(() => 0),
      diagnostics
    });
    const action = await agent.selectAction(observation);

    expect(observation.legalActions).toContainEqual(action);
    expect(diagnostics.inferenceFailureCount).toBe(1);
    expect(diagnostics.fallbackCount).toBe(1);
  });

  it("can complete a smoke game without illegal bids", async () => {
    const means = Array(BIDDING_ACTION_COUNT).fill(-10);
    means[1] = 10;
    const record = await runAutomatedGame({
      seed: 429,
      createAgent: () => createAgent({ means, criticValue: 0 })
    });

    expect(record.result.resultType).toBe("standard");
    expect(record.decisions.length).toBeGreaterThan(0);
  });

  it("aggregates Issue #429 runtime rewards from repo-managed artifacts", async () => {
    const playing = await loadRepoManagedPlayingPolicyBenchmark(
      PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID,
      { inferenceDevice: "cpu" }
    );
    const t1 = await loadRepoManagedBiddingMarginPolicyBenchmark(
      ISSUE427_T1_BIDDING_MARGIN_POLICY_ID,
      { inferenceDevice: "cpu" }
    );
    if (playing.critic === undefined) {
      throw new Error("fixture playing artifact must include a critic.");
    }

    const result = await runIssue429T1BiddingRuntimeEvaluation({
      startSeed: 429,
      gameCount: 2,
      playingPolicy: playing.policy,
      critic: playing.critic,
      t1MarginModel: t1.model
    });

    expect(result.candidates).toHaveLength(2);
    for (const candidate of result.candidates) {
      expect(candidate.games.completed).toBe(2);
      expect(candidate.safety.crashCount).toBe(0);
      expect(candidate.rewards.candidateRelativeReward.count).toBe(2);
      expect(candidate.rewards.meanRelativeRewardPerPlayerGame.count).toBe(10);
    }
  });
});

function createAgent(options: {
  means: readonly number[];
  sigma?: number;
  criticValue?: number;
  diagnostics?: ReturnType<typeof createT1NapoleonEvBiddingDiagnostics>;
}) {
  return new T1NapoleonEvBiddingAgent({
    marginModel: new FakeMarginModel(options.means, options.sigma ?? 1) as unknown as BiddingMarginOnnxModel,
    passEvAgent: new CriticEvBiddingAgent({
      critic: new ConstantCritic(options.criticValue ?? 0)
    }),
    delegateAgent: new RuleBasedAgent(() => 0),
    diagnostics: options.diagnostics
  });
}

function createObservation(state: GameState) {
  const playerId = state.currentPlayerId;
  const view = createPlayerView(state, playerId);
  return {
    playerId,
    view,
    legalActions: view.legalActions,
    publicActionHistory: state.bidding?.history.map((action, index) => ({
      step: index + 1,
      playerId: action.playerId,
      phase: "bidding" as const,
      action
    })) ?? []
  };
}

function createRaiseState(): GameState {
  let state = createInitialGame({ rng: () => 0 });
  state = applyAction(state, { type: "pass", playerId: "player-0" });
  state = applyAction(state, {
    type: "bid",
    playerId: "player-1",
    suit: "spades",
    targetPointCards: 13
  });
  state = applyAction(state, { type: "pass", playerId: "player-2" });
  state = applyAction(state, { type: "pass", playerId: "player-3" });
  state = applyAction(state, { type: "pass", playerId: "player-4" });
  return state;
}
