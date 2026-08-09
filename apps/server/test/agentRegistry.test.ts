import { describe, expect, it } from "vitest";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { RuleBasedAgent } from "@napoleon/ai";
import {
  createInitialGame,
  createPlayerView,
  getLegalActions
} from "@napoleon/game-core";
import type { PolicyOnnxModel } from "@napoleon/policy-onnx";
import {
  AgentUnavailableError,
  PLAYING_POLICY_ONNX_AGENT_IDS,
  PLAYING_POLICY_ONNX_AGENT_ID,
  RULE_BASED_AGENT_ID,
  UnknownAgentIdError,
  createAgentRegistry,
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
});
