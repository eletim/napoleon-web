import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { describe, expect, it } from "vitest";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { RuleBasedAgent } from "@napoleon/ai";
import {
  createInitialGame,
  createPlayerView,
  getLegalActions
} from "@napoleon/game-core";
import {
  ADJUTANT_ENCODER_SCHEMA_VERSION,
  ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
  ADJUTANT_MODEL_INPUT_SCHEMA_VERSION,
  BIDDING_ACTION_COUNT,
  BIDDING_ENCODER_SCHEMA_VERSION,
  BIDDING_MODEL_INPUT_FEATURE_COUNT,
  BIDDING_MODEL_INPUT_SCHEMA_VERSION,
  CARD_COUNT,
  DATASET_SCHEMA_VERSION,
  EXCHANGE_DECISION_MODE_SEQUENTIAL_CARD,
  EXCHANGE_ENCODER_SCHEMA_VERSION,
  EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
  EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
  MODEL_INPUT_FEATURE_COUNT,
  MODEL_INPUT_SCHEMA_VERSION,
  MULTIPHASE_DATASET_SCHEMA_VERSION,
  NONPLAYING_ONNX_METADATA_SCHEMA_VERSION,
  ONNX_INPUT_NAME,
  ONNX_OPSET_VERSION,
  ONNX_OUTPUT_NAME,
  PLAYING_ENCODER_SCHEMA_VERSION,
  POLICY_CHECKPOINT_SCHEMA_VERSION,
  POLICY_ONNX_METADATA_SCHEMA_VERSION,
  calculateCardIdsSha256
} from "@napoleon/policy-onnx";
import type {
  NonPlayingPolicyOnnxMetadata,
  NonPlayingPolicyOnnxModel,
  NonPlayingPolicyType,
  PolicyOnnxMetadata,
  PolicyOnnxModel
} from "@napoleon/policy-onnx";
import {
  AgentUnavailableError,
  FULL_POLICY_ONNX_AGENT_ID,
  FULL_POLICY_ONNX_AGENT_IDS,
  PLAYING_POLICY_ONNX_AGENT_IDS,
  PLAYING_POLICY_ONNX_AGENT_ID,
  RULE_BASED_AGENT_ID,
  UnknownAgentIdError,
  createAgentRegistryFromEnvironment,
  createAgentRegistry,
  readFullPolicyOnnxAgentConfigs,
  readPlayingPolicyOnnxAgentConfigs
} from "../src/agentRegistry.js";
import { createAgentConfiguration } from "../src/store.js";

const workspaceRoot = fileURLToPath(new URL("../../..", import.meta.url));

describe("agent registry", () => {
  it("lists stable agent ids and availability", () => {
    const registry = createAgentRegistry();

    expect(registry.listAgents()).toEqual([
      {
        id: RULE_BASED_AGENT_ID,
        displayName: "Rule-based AI",
        isAvailable: true
      }
    ]);
  });

  it("lists configured learned ONNX agents by display name", () => {
    const registry = createAgentRegistry({
      playingPolicyOnnxAgents: [
        {
          id: PLAYING_POLICY_ONNX_AGENT_IDS[0],
          displayName: "RL v900",
          onnxPath: "/models/v900.onnx",
          metadataPath: "/models/v900.json"
        },
        {
          id: PLAYING_POLICY_ONNX_AGENT_IDS[2],
          displayName: "RL v1200",
          onnxPath: "/models/v1200.onnx",
          metadataPath: "/models/v1200.json"
        }
      ]
    });

    expect(registry.listAgents()).toEqual([
      {
        id: RULE_BASED_AGENT_ID,
        displayName: "Rule-based AI",
        isAvailable: true
      },
      {
        id: PLAYING_POLICY_ONNX_AGENT_IDS[0],
        displayName: "RL v900",
        isAvailable: true
      },
      {
        id: PLAYING_POLICY_ONNX_AGENT_IDS[2],
        displayName: "RL v1200",
        isAvailable: true
      }
    ]);
  });

  it("lists configured full-policy ONNX agents by display name", () => {
    const registry = createAgentRegistry({
      fullPolicyOnnxAgents: [
        {
          id: FULL_POLICY_ONNX_AGENT_ID,
          displayName: "Full policy v1",
          playing: { onnxPath: "/models/playing.onnx", metadataPath: "/models/playing.json" },
          bidding: { onnxPath: "/models/bidding.onnx", metadataPath: "/models/bidding.json" },
          adjutant: { onnxPath: "/models/adjutant.onnx", metadataPath: "/models/adjutant.json" },
          exchange: { onnxPath: "/models/exchange.onnx", metadataPath: "/models/exchange.json" }
        }
      ]
    });

    expect(registry.listAgents()).toEqual([
      {
        id: RULE_BASED_AGENT_ID,
        displayName: "Rule-based AI",
        isAvailable: true
      },
      {
        id: FULL_POLICY_ONNX_AGENT_ID,
        displayName: "Full policy v1",
        isAvailable: true
      }
    ]);
  });

  it("creates rule-based agents by default", () => {
    const registry = createAgentRegistry();
    const configuration = createAgentConfiguration(["player-1", "player-2"], registry);

    expect(configuration.agentIds).toEqual(
      new Map([
        ["player-1", RULE_BASED_AGENT_ID],
        ["player-2", RULE_BASED_AGENT_ID]
      ])
    );
    expect(configuration.agents.get("player-1")).toBeInstanceOf(RuleBasedAgent);
    expect(configuration.agents.get("player-2")).toBeInstanceOf(RuleBasedAgent);
  });

  it("rejects unknown agent ids", () => {
    const registry = createAgentRegistry();

    expect(() =>
      createAgentConfiguration(
        ["player-1"],
        registry,
        [
          {
            playerId: "player-1",
            agentId: "missing-agent"
          }
        ]
      )
    ).toThrow(UnknownAgentIdError);
  });

  it("reads up to five learned ONNX agent slots from environment variables", () => {
    expect(
      readPlayingPolicyOnnxAgentConfigs({
        NAPOLEON_POLICY_1_DISPLAY_NAME: "RL v900",
        NAPOLEON_POLICY_1_ONNX_PATH: "/models/v900.onnx",
        NAPOLEON_POLICY_1_METADATA_PATH: "/models/v900.json",
        NAPOLEON_POLICY_2_DISPLAY_NAME: "",
        NAPOLEON_POLICY_2_ONNX_PATH: "/models/unused.onnx",
        NAPOLEON_POLICY_2_METADATA_PATH: "/models/unused.json",
        NAPOLEON_POLICY_5_DISPLAY_NAME: "RL v1400",
        NAPOLEON_POLICY_5_ONNX_PATH: "/models/v1400.onnx",
        NAPOLEON_POLICY_5_METADATA_PATH: "/models/v1400.json"
      })
    ).toEqual([
      {
        id: PLAYING_POLICY_ONNX_AGENT_IDS[0],
        displayName: "RL v900",
        onnxPath: "/models/v900.onnx",
        metadataPath: "/models/v900.json"
      },
      {
        id: PLAYING_POLICY_ONNX_AGENT_IDS[4],
        displayName: "RL v1400",
        onnxPath: "/models/v1400.onnx",
        metadataPath: "/models/v1400.json"
      }
    ]);
  });

  it("reads full-policy ONNX agent slots from environment variables", () => {
    expect(
      readFullPolicyOnnxAgentConfigs({
        NAPOLEON_FULL_POLICY_1_DISPLAY_NAME: "Full v1",
        NAPOLEON_FULL_POLICY_1_PLAYING_ONNX_PATH: "/models/playing.onnx",
        NAPOLEON_FULL_POLICY_1_PLAYING_METADATA_PATH: "/models/playing.json",
        NAPOLEON_FULL_POLICY_1_BIDDING_ONNX_PATH: "/models/bidding.onnx",
        NAPOLEON_FULL_POLICY_1_BIDDING_METADATA_PATH: "/models/bidding.json",
        NAPOLEON_FULL_POLICY_1_ADJUTANT_ONNX_PATH: "/models/adjutant.onnx",
        NAPOLEON_FULL_POLICY_1_ADJUTANT_METADATA_PATH: "/models/adjutant.json",
        NAPOLEON_FULL_POLICY_1_EXCHANGE_ONNX_PATH: "/models/exchange.onnx",
        NAPOLEON_FULL_POLICY_1_EXCHANGE_METADATA_PATH: "/models/exchange.json",
        NAPOLEON_FULL_POLICY_2_DISPLAY_NAME: "",
        NAPOLEON_FULL_POLICY_5_DISPLAY_NAME: "Full v5",
        NAPOLEON_FULL_POLICY_5_PLAYING_ONNX_PATH: "/models/playing5.onnx",
        NAPOLEON_FULL_POLICY_5_PLAYING_METADATA_PATH: "/models/playing5.json",
        NAPOLEON_FULL_POLICY_5_BIDDING_ONNX_PATH: "/models/bidding5.onnx",
        NAPOLEON_FULL_POLICY_5_BIDDING_METADATA_PATH: "/models/bidding5.json",
        NAPOLEON_FULL_POLICY_5_ADJUTANT_ONNX_PATH: "/models/adjutant5.onnx",
        NAPOLEON_FULL_POLICY_5_ADJUTANT_METADATA_PATH: "/models/adjutant5.json",
        NAPOLEON_FULL_POLICY_5_EXCHANGE_ONNX_PATH: "/models/exchange5.onnx",
        NAPOLEON_FULL_POLICY_5_EXCHANGE_METADATA_PATH: "/models/exchange5.json"
      })
    ).toEqual([
      {
        id: FULL_POLICY_ONNX_AGENT_IDS[0],
        displayName: "Full v1",
        playing: { onnxPath: "/models/playing.onnx", metadataPath: "/models/playing.json" },
        bidding: { onnxPath: "/models/bidding.onnx", metadataPath: "/models/bidding.json" },
        adjutant: { onnxPath: "/models/adjutant.onnx", metadataPath: "/models/adjutant.json" },
        exchange: { onnxPath: "/models/exchange.onnx", metadataPath: "/models/exchange.json" }
      },
      {
        id: FULL_POLICY_ONNX_AGENT_IDS[4],
        displayName: "Full v5",
        playing: { onnxPath: "/models/playing5.onnx", metadataPath: "/models/playing5.json" },
        bidding: { onnxPath: "/models/bidding5.onnx", metadataPath: "/models/bidding5.json" },
        adjutant: { onnxPath: "/models/adjutant5.onnx", metadataPath: "/models/adjutant5.json" },
        exchange: { onnxPath: "/models/exchange5.onnx", metadataPath: "/models/exchange5.json" }
      }
    ]);
  });

  it("resolves relative learned ONNX paths from the workspace root", () => {
    expect(
      readPlayingPolicyOnnxAgentConfigs(
        {
          NAPOLEON_POLICY_1_DISPLAY_NAME: "RL v740",
          NAPOLEON_POLICY_1_ONNX_PATH: "benchmarks/playing-policies/rl-v740/policy.onnx",
          NAPOLEON_POLICY_1_METADATA_PATH: "benchmarks/playing-policies/rl-v740/policy.json"
        },
        join(workspaceRoot, "apps/server")
      )
    ).toEqual([
      {
        id: PLAYING_POLICY_ONNX_AGENT_IDS[0],
        displayName: "RL v740",
        onnxPath: join(workspaceRoot, "benchmarks/playing-policies/rl-v740/policy.onnx"),
        metadataPath: join(workspaceRoot, "benchmarks/playing-policies/rl-v740/policy.json")
      }
    ]);
  });

  it("fails clearly when an enabled learned ONNX slot is incomplete", () => {
    expect(() =>
      readPlayingPolicyOnnxAgentConfigs({
        NAPOLEON_POLICY_3_DISPLAY_NAME: "Broken RL",
        NAPOLEON_POLICY_3_ONNX_PATH: "/models/broken.onnx"
      })
    ).toThrow(
      "Incomplete learned ONNX agent configuration for slot 3: NAPOLEON_POLICY_3_METADATA_PATH must be set when NAPOLEON_POLICY_3_DISPLAY_NAME is non-empty."
    );
  });

  it("fails clearly when an enabled full-policy ONNX slot is incomplete", () => {
    expect(() =>
      readFullPolicyOnnxAgentConfigs({
        NAPOLEON_FULL_POLICY_3_DISPLAY_NAME: "Broken full",
        NAPOLEON_FULL_POLICY_3_PLAYING_ONNX_PATH: "/models/playing.onnx"
      })
    ).toThrow(
      "Incomplete full-policy ONNX agent configuration for slot 3: " +
        "NAPOLEON_FULL_POLICY_3_PLAYING_METADATA_PATH must be set when " +
        "NAPOLEON_FULL_POLICY_3_DISPLAY_NAME is non-empty."
    );
  });

  it("fails at startup when a configured full-policy artifact file is missing", () => {
    expect(() =>
      createAgentRegistryFromEnvironment({
        NAPOLEON_FULL_POLICY_1_DISPLAY_NAME: "Broken full",
        NAPOLEON_FULL_POLICY_1_PLAYING_ONNX_PATH: "/models/missing-playing.onnx",
        NAPOLEON_FULL_POLICY_1_PLAYING_METADATA_PATH: "/models/missing-playing.json",
        NAPOLEON_FULL_POLICY_1_BIDDING_ONNX_PATH: "/models/missing-bidding.onnx",
        NAPOLEON_FULL_POLICY_1_BIDDING_METADATA_PATH: "/models/missing-bidding.json",
        NAPOLEON_FULL_POLICY_1_ADJUTANT_ONNX_PATH: "/models/missing-adjutant.onnx",
        NAPOLEON_FULL_POLICY_1_ADJUTANT_METADATA_PATH: "/models/missing-adjutant.json",
        NAPOLEON_FULL_POLICY_1_EXCHANGE_ONNX_PATH: "/models/missing-exchange.onnx",
        NAPOLEON_FULL_POLICY_1_EXCHANGE_METADATA_PATH: "/models/missing-exchange.json"
      })
    ).toThrow(
      "Full policy ONNX agent full-policy-onnx playing ONNX file does not exist: " +
        "/models/missing-playing.onnx"
    );
  });

  it("fails at startup when a configured full-policy exchange artifact is not sequential-card-v1", () => {
    const fixture = writeFullPolicyArtifactFixture({ exchangeDecisionMode: "top3-set-v1" });

    try {
      expect(() => createAgentRegistryFromEnvironment(fixture.env)).toThrow(
        "Full policy ONNX agent full-policy-onnx exchange policy decisionMode mismatch: " +
          "expected sequential-card-v1, got top3-set-v1."
      );
    } finally {
      rmSync(fixture.directory, { recursive: true, force: true });
    }
  });

  it("uses the selected learned ONNX slot's path pair", async () => {
    const registry = createAgentRegistry({
      playingPolicyOnnxAgents: [
        {
          id: PLAYING_POLICY_ONNX_AGENT_IDS[0],
          displayName: "RL v900",
          onnxPath: "/models/v900.onnx",
          metadataPath: "/models/v900.json"
        },
        {
          id: PLAYING_POLICY_ONNX_AGENT_IDS[1],
          displayName: "RL v901",
          onnxPath: "/models/v901.onnx",
          metadataPath: "/models/v901.json"
        }
      ],
      loadPlayingPolicyOnnxModel: async (paths) => {
        throw new Error(`${paths.onnxPath}|${paths.metadataPath}`);
      }
    });
    const agent = registry.createAgent(PLAYING_POLICY_ONNX_AGENT_IDS[1]);
    const state = createInitialGame();
    const playerId = "player-0";
    const observation = {
      playerId,
      view: createPlayerView(state, playerId),
      legalActions: getLegalActions(state, playerId),
      publicActionHistory: []
    };

    await expect(agent.selectAction(observation)).rejects.toMatchObject({
      name: AgentUnavailableError.name,
      agentId: PLAYING_POLICY_ONNX_AGENT_IDS[1],
      message: "Playing policy ONNX could not be loaded: /models/v901.onnx|/models/v901.json"
    });
  });

  it("retries loading the learned policy after a failed lazy load", async () => {
    let loadCount = 0;
    const registry = createAgentRegistry({
      playingPolicyOnnx: {
        onnxPath: "/tmp/policy.onnx",
        metadataPath: "/tmp/policy.json"
      },
      loadPlayingPolicyOnnxModel: async () => {
        loadCount += 1;

        if (loadCount === 1) {
          throw new Error("temporary load failure");
        }

        return {} as PolicyOnnxModel;
      }
    });
    const agent = registry.createAgent(PLAYING_POLICY_ONNX_AGENT_ID);
    const state = createInitialGame();
    const playerId = "player-0";
    const observation = {
      playerId,
      view: createPlayerView(state, playerId),
      legalActions: getLegalActions(state, playerId),
      publicActionHistory: []
    };

    await expect(agent.selectAction(observation)).rejects.toThrow(
      "Playing policy ONNX could not be loaded: temporary load failure"
    );
    await expect(agent.selectAction(observation)).resolves.toMatchObject({
      playerId
    });
    expect(loadCount).toBe(2);
  });

  it("fails fast when the full-policy exchange artifact is not sequential-card-v1", async () => {
    const registry = createAgentRegistry({
      fullPolicyOnnxAgents: [
        {
          id: FULL_POLICY_ONNX_AGENT_ID,
          displayName: "Full policy",
          playing: { onnxPath: "/models/playing.onnx", metadataPath: "/models/playing.json" },
          bidding: { onnxPath: "/models/bidding.onnx", metadataPath: "/models/bidding.json" },
          adjutant: { onnxPath: "/models/adjutant.onnx", metadataPath: "/models/adjutant.json" },
          exchange: { onnxPath: "/models/exchange.onnx", metadataPath: "/models/exchange.json" }
        }
      ],
      loadPlayingPolicyOnnxModel: async () => ({} as PolicyOnnxModel),
      loadNonPlayingPolicyOnnxModel: async (paths) => {
        if (paths.onnxPath.includes("bidding")) {
          return fakeNonPlayingPolicy("bidding");
        }
        if (paths.onnxPath.includes("adjutant")) {
          return fakeNonPlayingPolicy("adjutant");
        }
        return fakeNonPlayingPolicy("exchange", { decisionMode: "top3-set-v1" });
      }
    });
    const agent = registry.createAgent(FULL_POLICY_ONNX_AGENT_ID);
    const state = createInitialGame();
    const playerId = "player-0";
    const observation = {
      playerId,
      view: createPlayerView(state, playerId),
      legalActions: getLegalActions(state, playerId),
      publicActionHistory: []
    };

    await expect(agent.selectAction(observation)).rejects.toMatchObject({
      name: AgentUnavailableError.name,
      agentId: FULL_POLICY_ONNX_AGENT_ID,
      message:
        "Full policy ONNX could not be loaded: exchange policy decisionMode mismatch: " +
        "expected sequential-card-v1, got top3-set-v1."
    });
  });
});

function fakeNonPlayingPolicy(
  policyType: "bidding" | "adjutant" | "exchange",
  metadata: Record<string, unknown> = {}
): NonPlayingPolicyOnnxModel {
  return {
    policyType,
    metadata: {
      policyType,
      ...metadata
    }
  } as NonPlayingPolicyOnnxModel;
}

function writeFullPolicyArtifactFixture(options: {
  exchangeDecisionMode: "top3-set-v1" | "sequential-card-v1";
}): { directory: string; env: NodeJS.ProcessEnv } {
  const directory = mkdtempSync(join(tmpdir(), "napoleon-full-policy-"));
  const paths = {
    playingOnnx: join(directory, "playing.onnx"),
    playingMetadata: join(directory, "playing.json"),
    biddingOnnx: join(directory, "bidding.onnx"),
    biddingMetadata: join(directory, "bidding.json"),
    adjutantOnnx: join(directory, "adjutant.onnx"),
    adjutantMetadata: join(directory, "adjutant.json"),
    exchangeOnnx: join(directory, "exchange.onnx"),
    exchangeMetadata: join(directory, "exchange.json")
  };

  for (const onnxPath of [
    paths.playingOnnx,
    paths.biddingOnnx,
    paths.adjutantOnnx,
    paths.exchangeOnnx
  ]) {
    writeFileSync(onnxPath, "");
  }
  writeJson(paths.playingMetadata, createPlayingMetadata());
  writeJson(paths.biddingMetadata, createNonPlayingMetadata("bidding"));
  writeJson(paths.adjutantMetadata, createNonPlayingMetadata("adjutant"));
  writeJson(paths.exchangeMetadata, createNonPlayingMetadata("exchange", {
    decisionMode: options.exchangeDecisionMode
  }));

  return {
    directory,
    env: {
      NAPOLEON_FULL_POLICY_1_DISPLAY_NAME: "Full policy",
      NAPOLEON_FULL_POLICY_1_PLAYING_ONNX_PATH: paths.playingOnnx,
      NAPOLEON_FULL_POLICY_1_PLAYING_METADATA_PATH: paths.playingMetadata,
      NAPOLEON_FULL_POLICY_1_BIDDING_ONNX_PATH: paths.biddingOnnx,
      NAPOLEON_FULL_POLICY_1_BIDDING_METADATA_PATH: paths.biddingMetadata,
      NAPOLEON_FULL_POLICY_1_ADJUTANT_ONNX_PATH: paths.adjutantOnnx,
      NAPOLEON_FULL_POLICY_1_ADJUTANT_METADATA_PATH: paths.adjutantMetadata,
      NAPOLEON_FULL_POLICY_1_EXCHANGE_ONNX_PATH: paths.exchangeOnnx,
      NAPOLEON_FULL_POLICY_1_EXCHANGE_METADATA_PATH: paths.exchangeMetadata
    }
  };
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, JSON.stringify(value) + "\n", "utf8");
}

function createPlayingMetadata(): PolicyOnnxMetadata {
  return {
    metadataSchemaVersion: POLICY_ONNX_METADATA_SCHEMA_VERSION,
    checkpointSchemaVersion: POLICY_CHECKPOINT_SCHEMA_VERSION,
    datasetSchemaVersion: DATASET_SCHEMA_VERSION,
    playingEncoderSchemaVersion: PLAYING_ENCODER_SCHEMA_VERSION,
    modelInputSchemaVersion: MODEL_INPUT_SCHEMA_VERSION,
    modelInputFeatureCount: MODEL_INPUT_FEATURE_COUNT,
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
    }
  };
}

function createNonPlayingMetadata(
  policyType: NonPlayingPolicyType,
  overrides: Partial<NonPlayingPolicyOnnxMetadata> = {}
): NonPlayingPolicyOnnxMetadata {
  const specs = {
    bidding: {
      artifactType: "napoleon-bidding-policy-onnx",
      encoderSchemaVersion: BIDDING_ENCODER_SCHEMA_VERSION,
      modelInputSchemaVersion: BIDDING_MODEL_INPUT_SCHEMA_VERSION,
      modelInputFeatureCount: BIDDING_MODEL_INPUT_FEATURE_COUNT,
      outputCount: BIDDING_ACTION_COUNT
    },
    adjutant: {
      artifactType: "napoleon-adjutant-policy-onnx",
      encoderSchemaVersion: ADJUTANT_ENCODER_SCHEMA_VERSION,
      modelInputSchemaVersion: ADJUTANT_MODEL_INPUT_SCHEMA_VERSION,
      modelInputFeatureCount: ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
      outputCount: CARD_COUNT
    },
    exchange: {
      artifactType: "napoleon-exchange-policy-onnx",
      encoderSchemaVersion: EXCHANGE_ENCODER_SCHEMA_VERSION,
      modelInputSchemaVersion: EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
      modelInputFeatureCount: EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
      outputCount: CARD_COUNT,
      discardCount: 3,
      decisionMode: EXCHANGE_DECISION_MODE_SEQUENTIAL_CARD
    }
  } as const;
  const spec = specs[policyType];

  return {
    metadataSchemaVersion: NONPLAYING_ONNX_METADATA_SCHEMA_VERSION,
    artifactType: spec.artifactType,
    policyType,
    checkpointSchemaVersion: POLICY_CHECKPOINT_SCHEMA_VERSION,
    datasetSchemaVersion: MULTIPHASE_DATASET_SCHEMA_VERSION,
    encoderSchemaVersion: spec.encoderSchemaVersion,
    modelInputSchemaVersion: spec.modelInputSchemaVersion,
    modelInputFeatureCount: spec.modelInputFeatureCount,
    outputLogitCount: spec.outputCount,
    actionCount: spec.outputCount,
    cardIdsSha256: calculateCardIdsSha256(),
    inputName: ONNX_INPUT_NAME,
    outputName: ONNX_OUTPUT_NAME,
    inputShape: ["batch", spec.modelInputFeatureCount],
    outputShape: ["batch", spec.outputCount],
    inputDtype: "float32",
    outputDtype: "float32",
    ...(policyType === "exchange"
      ? {
          discardCount: spec.discardCount,
          decisionMode: spec.decisionMode
        }
      : {}),
    onnx: {
      opsetVersion: ONNX_OPSET_VERSION,
      inputs: [
        {
          name: ONNX_INPUT_NAME,
          shape: ["batch", spec.modelInputFeatureCount],
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
    ...overrides
  };
}
