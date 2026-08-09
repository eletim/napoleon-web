import { describe, expect, it } from "vitest";
import { RuleBasedAgent } from "@napoleon/ai";
import {
  createInitialGame,
  createPlayerView,
  getLegalActions
} from "@napoleon/game-core";
import type { PolicyOnnxModel } from "@napoleon/policy-onnx";
import {
  PLAYING_POLICY_ONNX_AGENT_ID,
  RULE_BASED_AGENT_ID,
  UnknownAgentIdError,
  createAgentRegistry
} from "../src/agentRegistry.js";
import { createAgentConfiguration } from "../src/store.js";

describe("agent registry", () => {
  it("lists stable agent ids and availability", () => {
    const registry = createAgentRegistry();

    expect(registry.listAgents()).toEqual([
      {
        id: RULE_BASED_AGENT_ID,
        displayName: "Rule-based AI",
        isAvailable: true
      },
      {
        id: PLAYING_POLICY_ONNX_AGENT_ID,
        displayName: "Playing Policy ONNX",
        isAvailable: false
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
