import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, it } from "vitest";
import { RuleBasedAgent, runAutomatedGame } from "@napoleon/ai";
import {
  createPlayingTrainingSample,
  encodePlayingModelInput,
  getCardId
} from "@napoleon/ai-observation";
import type { GameAction } from "@napoleon/game-core";
import {
  CARD_COUNT,
  MODEL_INPUT_FEATURE_COUNT,
  ONNX_INPUT_NAME,
  ONNX_OPSET_VERSION,
  ONNX_OUTPUT_NAME,
  PolicyOnnxAgent,
  calculateCardIdsSha256,
  createPolicyOnnxPlayInput,
  loadPolicyOnnxModel
} from "../src/index.js";
import type { PolicyOnnxModel } from "../src/index.js";
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

  it("builds the same live model input as the training sample pipeline for every seat", async () => {
    const source = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decisions = source.playerIds.map((playerId) => {
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

    for (const decision of decisions) {
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

  it("rejects playing observations that do not carry public action history", async () => {
    const policy = await createIncreasingLogitPolicy();
    const source = await runAutomatedGame({
      seed: 12345,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    });
    const decision = source.decisions.find((candidate) => candidate.phase === "playing");

    if (decision === undefined) {
      throw new Error("Expected a playing decision.");
    }

    const agent = new PolicyOnnxAgent({ policy });

    await expect(
      agent.selectAction({
        ...decision.observation,
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

function nonPlayingActions(
  record: Awaited<ReturnType<typeof runAutomatedGame>>
): readonly GameAction[] {
  return record.decisions
    .filter((decision) => decision.phase !== "playing")
    .map((decision) => decision.action);
}

function createMetadata() {
  return {
    metadataSchemaVersion: 1,
    checkpointSchemaVersion: 1,
    datasetSchemaVersion: 1,
    playingEncoderSchemaVersion: 1,
    modelInputSchemaVersion: 1,
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

async function temporaryDirectory(): Promise<string> {
  const directory = join(tmpdir(), `policy-onnx-agent-${process.pid}-${Date.now()}-${Math.random()}`);
  await mkdir(directory, { recursive: true });
  return directory;
}
