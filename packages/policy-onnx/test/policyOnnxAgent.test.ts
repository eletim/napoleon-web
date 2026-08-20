import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, it, vi } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import {
  createAdjutantTrainingSample,
  createBiddingTrainingSample,
  createExchangeTrainingSample,
  createPlayingTrainingSample,
  encodeAdjutantModelInput,
  encodeBiddingModelInput,
  encodeExchangeModelInput,
  encodePlayingModelInput,
  getCardId,
  getCardIndex
} from "@napoleon/ai-observation";
import type { GameAction } from "@napoleon/game-core";
import {
  ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
  BIDDING_ACTION_COUNT,
  BIDDING_MODEL_INPUT_FEATURE_COUNT,
  BIDDING_MODEL_INPUT_SCHEMA_VERSION,
  CARD_COUNT,
  COMPLETE_INFO_COMPACT_PLAYING_OBSERVATION_VARIANT,
  COMPLETE_INFO_PLAYING_MODEL_INPUT_FEATURE_COUNT,
  EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
  MODEL_INPUT_FEATURE_COUNT,
  MULTIPHASE_DATASET_SCHEMA_VERSION,
  NONPLAYING_ONNX_METADATA_SCHEMA_VERSION,
  ONNX_INPUT_NAME,
  ONNX_OPSET_VERSION,
  ONNX_OUTPUT_NAME,
  PolicyOnnxAgent,
  calculateCardIdsSha256,
  createPolicyOnnxAdjutantInput,
  createPolicyOnnxBiddingInput,
  createPolicyOnnxCompleteInfoPlayInput,
  createPolicyOnnxExchangeInput,
  createPolicyOnnxPlayInput,
  loadNonPlayingPolicyOnnxModel,
  loadPolicyOnnxModel
} from "../src/index.js";
import type {
  NonPlayingPolicyOnnxMetadata,
  NonPlayingPolicyOnnxModel,
  NonPlayingPolicyType,
  PolicyOnnxModel
} from "../src/index.js";
import { createConstantPolicyOnnx } from "./testOnnxFixture.js";

describe("PolicyOnnxAgent", () => {
  it("runs a complete automated game through the ONNX playing policy", async () => {
    const policy = await createIncreasingLogitPolicy();
    const record = await runAutomatedGame({
      seed: 24680,
      createAgent: ({ rng }) => new PolicyOnnxAgent({ policy, rng })
    });
    const playDecisions = record.decisions.filter((decision) => decision.phase === "playing");

    expect(record.result.winner).toMatch(/^(napoleon-team|alliance)$/);
    expect(playDecisions.length).toBeGreaterThan(0);
    expect(
      playDecisions.every((decision) =>
        decision.legalActions.every((action) => action.type === "play-card")
      )
    ).toBe(true);
    expect(countIllegalPlayActions(playDecisions)).toBe(0);
  });

  it("replays the same seed and ONNX model with the same action sequence", async () => {
    const policy = await createIncreasingLogitPolicy();
    const createRecord = () =>
      runAutomatedGame({
        seed: 13579,
        createAgent: ({ rng }) => new PolicyOnnxAgent({ policy, rng })
      });

    const first = await createRecord();
    const second = await createRecord();

    expect(second.decisions.map((decision) => decision.action)).toEqual(
      first.decisions.map((decision) => decision.action)
    );
  });

  it("uses RuleBasedAgent for non-playing phases", async () => {
    const policy = await createIncreasingLogitPolicy();
    const ruleBased = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const onnx = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new PolicyOnnxAgent({ policy, rng })
    });

    expect(nonPlayingActions(onnx)).toEqual(nonPlayingActions(ruleBased));
  });

  it("runs a complete deterministic game with ONNX policies for every phase", async () => {
    const {
      policy,
      biddingPolicy,
      adjutantPolicy,
      exchangePolicy
    } = await createFullPolicyFixture();
    const biddingSpy = vi.spyOn(biddingPolicy, "selectBidding");
    const adjutantSpy = vi.spyOn(adjutantPolicy, "selectAdjutant");
    const exchangeSpy = vi.spyOn(exchangePolicy, "selectExchange");
    const playSpy = vi.spyOn(policy, "selectLegalPlay");
    const ruleBasedSpy = vi.spyOn(RuleBasedAgent.prototype, "selectAction");
    const createRecord = () =>
      runAutomatedGame({
        seed: 112233,
        createAgent: ({ rng }) =>
          new PolicyOnnxAgent({
            policy,
            biddingPolicy,
            adjutantPolicy,
            exchangePolicy,
            rng
          })
      });

    try {
      const first = await createRecord();
      const second = await createRecord();

      expect(first.result.winner).toMatch(/^(napoleon-team|alliance)$/);
      expect(biddingSpy).toHaveBeenCalled();
      expect(adjutantSpy).toHaveBeenCalled();
      expect(exchangeSpy).toHaveBeenCalled();
      expect(playSpy).toHaveBeenCalled();
      expect(ruleBasedSpy).not.toHaveBeenCalled();
      expect(countIllegalActions(first.decisions)).toBe(0);
      expect(exchangeActionsAreLegalThreeCardDiscards(first.decisions)).toBe(true);
      expect(second.decisions.map((decision) => decision.action)).toEqual(
        first.decisions.map((decision) => decision.action)
      );
    } finally {
      ruleBasedSpy.mockRestore();
      playSpy.mockRestore();
      exchangeSpy.mockRestore();
      adjutantSpy.mockRestore();
      biddingSpy.mockRestore();
    }
  });

  it("runs sequential-card exchange policy as three ONNX selections before discarding", async () => {
    const {
      policy,
      biddingPolicy,
      adjutantPolicy
    } = await createFullPolicyFixture();
    const exchangePolicy = await createNonPlayingPolicy("exchange", {
      decisionMode: "sequential-card-v1"
    });
    const exchangeCardSpy = vi.spyOn(exchangePolicy, "selectExchangeCard");
    const exchangeTop3Spy = vi.spyOn(exchangePolicy, "selectExchange");

    try {
      const record = await runAutomatedGame({
        seed: 112233,
        createAgent: ({ rng }) =>
          new PolicyOnnxAgent({
            policy,
            biddingPolicy,
            adjutantPolicy,
            exchangePolicy,
            rng
          })
      });

      const exchangeDecisionCount = record.decisions.filter(
        (decision) => decision.phase === "exchanging"
      ).length;
      expect(exchangeDecisionCount).toBeGreaterThan(0);
      expect(exchangeCardSpy).toHaveBeenCalledTimes(exchangeDecisionCount * 3);
      expect(exchangeTop3Spy).not.toHaveBeenCalled();
      expect(countIllegalActions(record.decisions)).toBe(0);
      expect(exchangeActionsAreLegalThreeCardDiscards(record.decisions)).toBe(true);
    } finally {
      exchangeTop3Spy.mockRestore();
      exchangeCardSpy.mockRestore();
    }
  });


  it("rejects non-playing policy artifact types assigned to the wrong phase slots", async () => {
    const policy = await createIncreasingLogitPolicy();
    const exchangePolicy = await createNonPlayingPolicy("exchange");

    expect(() =>
      new PolicyOnnxAgent({
        policy,
        biddingPolicy: exchangePolicy
      })
    ).toThrow(/biddingPolicy policy type mismatch/);
  });

  it("selects the only legal play-card action for forced plays", async () => {
    const policy = await createIncreasingLogitPolicy();
    const source = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const forcedDecision = source.decisions.find(
      (decision) => decision.phase === "playing" && decision.legalActions.length === 1
    );

    if (forcedDecision === undefined) {
      throw new Error("Expected a forced play decision.");
    }

    const agent = new PolicyOnnxAgent({ policy });

    await expect(agent.selectAction(forcedDecision.observation)).resolves.toEqual(
      forcedDecision.legalActions[0]
    );
  });

  it("uses complete-info compact playing input when the policy metadata requests it", async () => {
    const source = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = source.decisions.find(
      (candidate) => candidate.phase === "playing" && candidate.action.type === "play-card"
    );

    if (decision === undefined || decision.action.type !== "play-card") {
      throw new Error("Expected a playing decision.");
    }

    const observedInputLengths: number[] = [];
    const compactPolicy = {
      metadata: {
        playingObservationVariant: COMPLETE_INFO_COMPACT_PLAYING_OBSERVATION_VARIANT
      },
      async selectLegalPlay(input) {
        observedInputLengths.push(input.modelInput.length);
        return {
          selectedCardIndex: getCardIndex(decision.action.cardId),
          logits: new Float32Array(CARD_COUNT)
        };
      }
    } as PolicyOnnxModel;
    const agent = new PolicyOnnxAgent({ policy: compactPolicy });
    const compactInput = createPolicyOnnxCompleteInfoPlayInput(decision.observation, {
      actualState: decision.actualState,
      playerIds: source.playerIds
    });

    expect(compactInput.modelInput).toHaveLength(COMPLETE_INFO_PLAYING_MODEL_INPUT_FEATURE_COUNT);
    await expect(
      agent.selectActionWithContext(decision.observation, {
        actualState: decision.actualState,
        playerIds: source.playerIds
      })
    ).resolves.toEqual(decision.action);
    expect(observedInputLengths).toEqual([COMPLETE_INFO_PLAYING_MODEL_INPUT_FEATURE_COUNT]);
  });

  it("rejects complete-info compact playing input without runtime actual state", async () => {
    const source = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = source.decisions.find((candidate) => candidate.phase === "playing");

    if (decision === undefined) {
      throw new Error("Expected a playing decision.");
    }

    const compactPolicy = {
      metadata: {
        playingObservationVariant: COMPLETE_INFO_COMPACT_PLAYING_OBSERVATION_VARIANT
      },
      async selectLegalPlay() {
        throw new Error("selectLegalPlay should not run without complete-info context.");
      }
    } as unknown as PolicyOnnxModel;
    const agent = new PolicyOnnxAgent({ policy: compactPolicy });

    await expect(agent.selectAction(decision.observation)).rejects.toThrow(
      "requires runtime actual state"
    );
  });

  it("builds the same live model input as the training sample pipeline for every phase", async () => {
    const source = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const biddingDecision = requiredDecision(source, "bidding");
    const adjutantDecision = requiredDecision(source, "choosing-adjutant");
    const exchangeDecision = requiredDecision(source, "exchanging");
    const playingDecisions = source.playerIds.map((playerId) => {
      const decision = source.decisions.find(
        (candidate) =>
          candidate.playerId === playerId &&
          candidate.phase === "playing" &&
          candidate.action.type === "play-card"
      );

      if (decision === undefined) {
        throw new Error(`Expected a playing decision for ${playerId}.`);
      }

      return decision;
    });

    const biddingLiveInput = createPolicyOnnxBiddingInput(
      biddingDecision.observation,
      source.playerIds
    );
    const biddingSample = createBiddingTrainingSample(source, biddingDecision);
    if (biddingSample === null) {
      throw new Error("Expected a bidding sample.");
    }
    expect(biddingLiveInput.legalBidMask).toEqual(biddingSample.observation.legalBidMask);
    expect(Buffer.from(biddingLiveInput.modelInput.buffer)).toEqual(
      Buffer.from(encodeBiddingModelInput(biddingSample.observation).buffer)
    );

    const adjutantLiveInput = createPolicyOnnxAdjutantInput(
      adjutantDecision.observation,
      source.playerIds
    );
    const adjutantSample = createAdjutantTrainingSample(source, adjutantDecision);
    if (adjutantSample === null) {
      throw new Error("Expected an adjutant sample.");
    }
    expect(adjutantLiveInput.legalAdjutantMask).toEqual(
      adjutantSample.observation.legalAdjutantMask
    );
    expect(Buffer.from(adjutantLiveInput.modelInput.buffer)).toEqual(
      Buffer.from(encodeAdjutantModelInput(adjutantSample.observation).buffer)
    );

    const exchangeLiveInput = createPolicyOnnxExchangeInput(
      exchangeDecision.observation,
      source.playerIds
    );
    const exchangeSample = createExchangeTrainingSample(source, exchangeDecision);
    if (exchangeSample === null) {
      throw new Error("Expected an exchange sample.");
    }
    expect(exchangeLiveInput.legalDiscardCardMask).toEqual(
      exchangeSample.observation.legalDiscardCardMask
    );
    expect(Buffer.from(exchangeLiveInput.modelInput.buffer)).toEqual(
      Buffer.from(encodeExchangeModelInput(exchangeSample.observation).buffer)
    );

    for (const decision of playingDecisions) {
      const defaultLiveInput = createPolicyOnnxPlayInput(decision.observation);
      const liveInput = createPolicyOnnxPlayInput(decision.observation, source.playerIds);
      const sample = createPlayingTrainingSample(source, decision);

      if (sample === null) {
        throw new Error("Expected a playing sample.");
      }

      expect(Buffer.from(defaultLiveInput.modelInput.buffer)).toEqual(
        Buffer.from(liveInput.modelInput.buffer)
      );
      expect(defaultLiveInput.legalPlayMask).toEqual(liveInput.legalPlayMask);
      expect(liveInput.legalPlayMask).toEqual(sample.observation.legalPlayMask);
      expect(Buffer.from(liveInput.modelInput.buffer)).toEqual(
        Buffer.from(encodePlayingModelInput(sample.observation).buffer)
      );
    }
  });

  it("does not silently fall back to RuleBasedAgent when inference fails", async () => {
    const source = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = source.decisions.find((candidate) => candidate.phase === "playing");

    if (decision === undefined) {
      throw new Error("Expected a playing decision.");
    }

    const failingPolicy = {
      async selectLegalPlay() {
        throw new Error("inference failed for test");
      }
    } as unknown as PolicyOnnxModel;
    const agent = new PolicyOnnxAgent({ policy: failingPolicy });

    await expect(agent.selectAction(decision.observation)).rejects.toThrow(
      "inference failed for test"
    );
  });

  it("rejects an ONNX-selected card that is outside the legal action set", async () => {
    const source = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = source.decisions.find((candidate) => candidate.phase === "playing");

    if (decision === undefined) {
      throw new Error("Expected a playing decision.");
    }

    const legalCardIds = new Set(
      decision.legalActions.flatMap((action) => action.type === "play-card" ? [action.cardId] : [])
    );
    const illegalCardIndex = Array.from({ length: CARD_COUNT }, (_, index) => index)
      .find((index) => !legalCardIds.has(getCardId(index)));

    if (illegalCardIndex === undefined) {
      throw new Error("Expected at least one illegal card index.");
    }

    const illegalPolicy = {
      async selectLegalPlay() {
        return {
          selectedCardIndex: illegalCardIndex,
          logits: new Float32Array(CARD_COUNT)
        };
      }
    } as unknown as PolicyOnnxModel;
    const agent = new PolicyOnnxAgent({ policy: illegalPolicy });

    await expect(agent.selectAction(decision.observation)).rejects.toThrow("outside legal actions");
  });

  it("rejects explicit playerIds that do not match the observation players", async () => {
    const source = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = source.decisions.find((candidate) => candidate.phase === "playing");

    if (decision === undefined) {
      throw new Error("Expected a playing decision.");
    }

    expect(() =>
      createPolicyOnnxPlayInput(decision.observation, [
        ...source.playerIds.slice(0, -1),
        "missing-player"
      ])
    ).toThrow("must contain exactly the same players");
    expect(() =>
      createPolicyOnnxPlayInput(decision.observation, [...source.playerIds].reverse())
    ).toThrow("must match the observation player order");
  });

  it("rejects live observations that do not carry public action history", async () => {
    const policy = await createIncreasingLogitPolicy();
    const source = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const biddingDecision = requiredDecision(source, "bidding");
    const adjutantDecision = requiredDecision(source, "choosing-adjutant");
    const exchangeDecision = requiredDecision(source, "exchanging");
    const playDecision = source.decisions.find((candidate) => candidate.phase === "playing");

    if (playDecision === undefined) {
      throw new Error("Expected a playing decision.");
    }

    const agent = new PolicyOnnxAgent({ policy });

    expect(() =>
      createPolicyOnnxBiddingInput({
        ...biddingDecision.observation,
        publicActionHistory: undefined
      })
    ).toThrow("publicActionHistory");
    expect(() =>
      createPolicyOnnxAdjutantInput({
        ...adjutantDecision.observation,
        publicActionHistory: undefined
      })
    ).toThrow("publicActionHistory");
    expect(() =>
      createPolicyOnnxExchangeInput({
        ...exchangeDecision.observation,
        publicActionHistory: undefined
      })
    ).toThrow("publicActionHistory");
    await expect(
      agent.selectAction({
        ...playDecision.observation,
        publicActionHistory: undefined
      })
    ).rejects.toThrow("publicActionHistory");
  });
});

const externalOnnxPath = process.env.NAPOLEON_POLICY_ONNX_PATH;
const externalMetadataPath = process.env.NAPOLEON_POLICY_METADATA_PATH;
const maybeExternalIt = externalOnnxPath !== undefined && externalMetadataPath !== undefined
  ? it
  : it.skip;

describe("PolicyOnnxAgent external artifact smoke", () => {
  maybeExternalIt("runs a complete deterministic game with the supplied trained artifact", async () => {
    if (externalOnnxPath === undefined || externalMetadataPath === undefined) {
      throw new Error("Expected external policy artifact paths.");
    }

    const policy = await loadPolicyOnnxModel({
      onnxPath: externalOnnxPath,
      metadataPath: externalMetadataPath
    });
    const createRecord = () =>
      runAutomatedGame({
        seed: 424242,
        createAgent: ({ rng }) => new PolicyOnnxAgent({ policy, rng })
      });
    const first = await createRecord();
    const second = await createRecord();

    expect(first.result.winner).toMatch(/^(napoleon-team|alliance)$/);
    expect(countIllegalPlayActions(first.decisions.filter((decision) => decision.phase === "playing"))).toBe(0);
    expect(second.decisions.map((decision) => decision.action)).toEqual(
      first.decisions.map((decision) => decision.action)
    );
  });
});

const externalBiddingOnnxPath = process.env.NAPOLEON_BIDDING_POLICY_ONNX_PATH;
const externalBiddingMetadataPath = process.env.NAPOLEON_BIDDING_POLICY_METADATA_PATH;
const externalAdjutantOnnxPath = process.env.NAPOLEON_ADJUTANT_POLICY_ONNX_PATH;
const externalAdjutantMetadataPath = process.env.NAPOLEON_ADJUTANT_POLICY_METADATA_PATH;
const externalExchangeOnnxPath = process.env.NAPOLEON_EXCHANGE_POLICY_ONNX_PATH;
const externalExchangeMetadataPath = process.env.NAPOLEON_EXCHANGE_POLICY_METADATA_PATH;
const maybeExternalFullIt =
  externalOnnxPath !== undefined &&
  externalMetadataPath !== undefined &&
  externalBiddingOnnxPath !== undefined &&
  externalBiddingMetadataPath !== undefined &&
  externalAdjutantOnnxPath !== undefined &&
  externalAdjutantMetadataPath !== undefined &&
  externalExchangeOnnxPath !== undefined &&
  externalExchangeMetadataPath !== undefined
    ? it
    : it.skip;

describe("PolicyOnnxAgent external full-policy artifact smoke", () => {
  maybeExternalFullIt("runs a deterministic full game with supplied phase artifacts", async () => {
    if (
      externalOnnxPath === undefined ||
      externalMetadataPath === undefined ||
      externalBiddingOnnxPath === undefined ||
      externalBiddingMetadataPath === undefined ||
      externalAdjutantOnnxPath === undefined ||
      externalAdjutantMetadataPath === undefined ||
      externalExchangeOnnxPath === undefined ||
      externalExchangeMetadataPath === undefined
    ) {
      throw new Error("Expected all external full-policy artifact paths.");
    }

    const policy = await loadPolicyOnnxModel({
      onnxPath: externalOnnxPath,
      metadataPath: externalMetadataPath
    });
    const biddingPolicy = await loadNonPlayingPolicyOnnxModel({
      onnxPath: externalBiddingOnnxPath,
      metadataPath: externalBiddingMetadataPath
    });
    const adjutantPolicy = await loadNonPlayingPolicyOnnxModel({
      onnxPath: externalAdjutantOnnxPath,
      metadataPath: externalAdjutantMetadataPath
    });
    const exchangePolicy = await loadNonPlayingPolicyOnnxModel({
      onnxPath: externalExchangeOnnxPath,
      metadataPath: externalExchangeMetadataPath
    });
    const createRecord = () =>
      runAutomatedGame({
        seed: 424242,
        createAgent: ({ rng }) =>
          new PolicyOnnxAgent({
            policy,
            biddingPolicy,
            adjutantPolicy,
            exchangePolicy,
            rng
          })
      });
    const first = await createRecord();
    const second = await createRecord();

    expect(first.result.winner).toMatch(/^(napoleon-team|alliance)$/);
    expect(countIllegalActions(first.decisions)).toBe(0);
    expect(exchangeActionsAreLegalThreeCardDiscards(first.decisions)).toBe(true);
    expect(second.decisions.map((decision) => decision.action)).toEqual(
      first.decisions.map((decision) => decision.action)
    );
  });
});

async function createFullPolicyFixture(): Promise<{
  policy: PolicyOnnxModel;
  biddingPolicy: NonPlayingPolicyOnnxModel;
  adjutantPolicy: NonPlayingPolicyOnnxModel;
  exchangePolicy: NonPlayingPolicyOnnxModel;
}> {
  return {
    policy: await createIncreasingLogitPolicy(),
    biddingPolicy: await createNonPlayingPolicy("bidding"),
    adjutantPolicy: await createNonPlayingPolicy("adjutant"),
    exchangePolicy: await createNonPlayingPolicy("exchange")
  };
}

async function createIncreasingLogitPolicy(): Promise<PolicyOnnxModel> {
  const logits = new Float32Array(CARD_COUNT);
  for (let index = 0; index < CARD_COUNT; index += 1) {
    logits[index] = index;
  }

  const directory = await temporaryDirectory();
  const onnxPath = join(directory, "policy.onnx");
  const metadataPath = join(directory, "policy.json");
  await writeFile(onnxPath, createConstantPolicyOnnx(logits));
  await writeFile(metadataPath, JSON.stringify(createMetadata()) + "\n", "utf8");

  return loadPolicyOnnxModel({ onnxPath, metadataPath });
}

async function createNonPlayingPolicy(
  policyType: NonPlayingPolicyType,
  metadataOverrides: Partial<NonPlayingPolicyOnnxMetadata> = {}
): Promise<NonPlayingPolicyOnnxModel> {
  const spec = nonPlayingSpec(policyType);
  const logits = new Float32Array(spec.outputCount);
  for (let index = 0; index < logits.length; index += 1) {
    logits[index] = index;
  }

  const directory = await temporaryDirectory();
  const onnxPath = join(directory, `${policyType}.onnx`);
  const metadataPath = join(directory, `${policyType}.json`);
  await writeFile(
    onnxPath,
    createConstantPolicyOnnx(logits, ONNX_OUTPUT_NAME, {
      inputFeatureCount: spec.inputFeatureCount,
      outputCount: spec.outputCount
    })
  );
  await writeFile(
    metadataPath,
    JSON.stringify({
      ...createNonPlayingMetadata(policyType),
      ...metadataOverrides
    }) + "\n",
    "utf8"
  );

  return loadNonPlayingPolicyOnnxModel({ onnxPath, metadataPath });
}

function countIllegalPlayActions(
  decisions: readonly {
    action: GameAction;
    legalActions: readonly GameAction[];
  }[]
): number {
  return decisions.filter(
    (decision) =>
      decision.action.type === "play-card" &&
      !decision.legalActions.some((action) => action.type === "play-card" && action.cardId === decision.action.cardId)
  ).length;
}

function countIllegalActions(
  decisions: readonly {
    action: GameAction;
    legalActions: readonly GameAction[];
  }[]
): number {
  return decisions.filter(
    (decision) =>
      !decision.legalActions.some((action) => actionsEqual(action, decision.action))
  ).length;
}

function nonPlayingActions(
  record: Awaited<ReturnType<typeof runAutomatedGame>>
): readonly GameAction[] {
  return record.decisions
    .filter((decision) => decision.phase !== "playing")
    .map((decision) => decision.action);
}

function exchangeActionsAreLegalThreeCardDiscards(
  decisions: Awaited<ReturnType<typeof runAutomatedGame>>["decisions"]
): boolean {
  return decisions
    .filter((decision) => decision.phase === "exchanging")
    .every((decision) => {
      if (decision.action.type !== "discard-cards") {
        return false;
      }

      const self = decision.observation.view.players.find(
        (player) => player.id === decision.playerId
      );
      const handIds = new Set(self?.hand?.map((card) => card.id) ?? []);

      return decision.action.cardIds.length === 3 &&
        new Set(decision.action.cardIds).size === 3 &&
        decision.action.cardIds.every((cardId) => handIds.has(cardId));
    });
}

function requiredDecision(
  record: Awaited<ReturnType<typeof runAutomatedGame>>,
  phase: "bidding" | "choosing-adjutant" | "exchanging"
): Awaited<ReturnType<typeof runAutomatedGame>>["decisions"][number] {
  const decision = record.decisions.find((candidate) => candidate.phase === phase);

  if (decision === undefined) {
    throw new Error(`Expected a ${phase} decision.`);
  }

  return decision;
}

function createMetadata() {
  return {
    metadataSchemaVersion: 1,
    checkpointSchemaVersion: 1,
    datasetSchemaVersion: 1,
    playingEncoderSchemaVersion: 2,
    modelInputSchemaVersion: 2,
    cardIdsSha256: calculateCardIdsSha256(),
    onnx: {
      opsetVersion: ONNX_OPSET_VERSION,
      inputs: [
        {
          name: ONNX_INPUT_NAME,
          shape: ["batch", MODEL_INPUT_FEATURE_COUNT],
          dtype: "float32"
        }
      ],
      outputs: [
        {
          name: ONNX_OUTPUT_NAME,
          shape: ["batch", CARD_COUNT],
          dtype: "float32"
        }
      ]
    },
    policyModel: {
      input_dim: MODEL_INPUT_FEATURE_COUNT,
      hidden_dim: 8,
      hidden_layers: 1,
      dropout: 0
    }
  };
}

function createNonPlayingMetadata(policyType: NonPlayingPolicyType): NonPlayingPolicyOnnxMetadata {
  const spec = nonPlayingSpec(policyType);
  const metadata: NonPlayingPolicyOnnxMetadata = {
    metadataSchemaVersion: NONPLAYING_ONNX_METADATA_SCHEMA_VERSION,
    artifactType: spec.artifactType,
    policyType,
    checkpointSchemaVersion: 1,
    datasetSchemaVersion: MULTIPHASE_DATASET_SCHEMA_VERSION,
    encoderSchemaVersion: spec.encoderSchemaVersion,
    modelInputSchemaVersion: spec.modelInputSchemaVersion,
    modelInputFeatureCount: spec.inputFeatureCount,
    outputLogitCount: spec.outputCount,
    actionCount: spec.outputCount,
    cardIdsSha256: calculateCardIdsSha256(),
    inputName: ONNX_INPUT_NAME,
    outputName: ONNX_OUTPUT_NAME,
    inputShape: ["batch", spec.inputFeatureCount],
    outputShape: ["batch", spec.outputCount],
    inputDtype: "float32",
    outputDtype: "float32",
    onnx: {
      opsetVersion: ONNX_OPSET_VERSION,
      inputs: [
        {
          name: ONNX_INPUT_NAME,
          shape: ["batch", spec.inputFeatureCount],
          dtype: "float32"
        }
      ],
      outputs: [
        {
          name: ONNX_OUTPUT_NAME,
          shape: ["batch", spec.outputCount],
          dtype: "float32"
        }
      ]
    },
    modelConfig: {
      input_dim: spec.inputFeatureCount,
      hidden_dim: 8,
      hidden_layers: 1,
      dropout: 0
    },
    checkpointSeed: 123,
    checkpointCompatibilityMetadata: {
      modelInputFeatureCount: spec.inputFeatureCount,
      outputCount: spec.outputCount
    }
  };

  if (policyType === "exchange") {
    metadata.discardCount = 3;
  }

  return metadata;
}

function nonPlayingSpec(policyType: NonPlayingPolicyType): {
  artifactType: string;
  inputFeatureCount: number;
  outputCount: number;
  encoderSchemaVersion: number;
  modelInputSchemaVersion: number;
} {
  if (policyType === "bidding") {
    return {
      artifactType: "napoleon-bidding-policy-onnx",
      inputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT,
      outputCount: BIDDING_ACTION_COUNT,
      encoderSchemaVersion: 1,
      modelInputSchemaVersion: BIDDING_MODEL_INPUT_SCHEMA_VERSION
    };
  }
  if (policyType === "exchange") {
    return {
      artifactType: "napoleon-exchange-policy-onnx",
      inputFeatureCount: EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
      outputCount: CARD_COUNT,
      encoderSchemaVersion: 2,
      modelInputSchemaVersion: 2
    };
  }
  return {
    artifactType: "napoleon-adjutant-policy-onnx",
    inputFeatureCount: ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
    outputCount: CARD_COUNT,
    encoderSchemaVersion: 1,
    modelInputSchemaVersion: 1
  };
}

function actionsEqual(left: GameAction, right: GameAction): boolean {
  if (left.type !== right.type || left.playerId !== right.playerId) {
    return false;
  }

  switch (left.type) {
    case "bid":
      return right.type === "bid" &&
        left.suit === right.suit &&
        left.targetPointCards === right.targetPointCards;
    case "pass":
      return right.type === "pass";
    case "choose-adjutant":
      return right.type === "choose-adjutant" && left.cardId === right.cardId;
    case "discard-cards":
      return right.type === "discard-cards" && sameCardIds(left.cardIds, right.cardIds);
    case "play-card":
      return right.type === "play-card" && left.cardId === right.cardId;
  }
}

function sameCardIds(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) {
    return false;
  }

  const sortedLeft = [...left].sort();
  const sortedRight = [...right].sort();

  return sortedLeft.every((cardId, index) => cardId === sortedRight[index]);
}

async function temporaryDirectory(): Promise<string> {
  const directory = join(tmpdir(), `policy-onnx-agent-${process.pid}-${Date.now()}-${Math.random()}`);
  await mkdir(directory, { recursive: true });
  return directory;
}
