import { describe, expect, it } from "vitest";
import type { AiPolicyComposition } from "@napoleon/protocol";
import { createAgentRegistry } from "../src/agentRegistry.js";
import {
  BUILTIN_AI_PRESETS,
  COM_AI_PRESET_ID,
  COM_RULE_BASE_PRESET_ID,
  InvalidAiPresetCompositionError,
  UnknownAiPresetIdError,
  createAiPresetRegistry
} from "../src/aiPresetRegistry.js";

describe("AI preset registry", () => {
  it("defines the two builtin presets using phase policy IDs only", () => {
    expect(BUILTIN_AI_PRESETS).toEqual([
      {
        id: COM_RULE_BASE_PRESET_ID,
        displayName: "COM-RuleBase",
        composition: {
          playing: "rule-based",
          bidding: "rule-based",
          nonPlaying: "rule-based"
        }
      },
      {
        id: COM_AI_PRESET_ID,
        displayName: "COM-AI",
        composition: {
          playing: "ppo-separated-v1000",
          bidding: "frozen-raise-v1",
          nonPlaying: "parameterized-adjutant-exchange-v1"
        }
      }
    ]);
    expect(JSON.stringify(BUILTIN_AI_PRESETS)).not.toMatch(/path|onnx|artifact/i);
  });

  it("saves, loads, and resolves an updated preset composition", () => {
    const registry = createAiPresetRegistry(createAgentRegistry());
    const mixed: AiPolicyComposition = {
      playing: "rule-based",
      bidding: "frozen-raise-v1",
      nonPlaying: "parameterized-adjutant-exchange-v1"
    };

    expect(registry.update(COM_AI_PRESET_ID, mixed).composition).toEqual(mixed);
    expect(registry.resolve(COM_AI_PRESET_ID).composition).toEqual(mixed);
    expect(registry.list().find(({ id }) => id === COM_AI_PRESET_ID)?.composition).toEqual(mixed);
  });

  it("rejects unknown preset and policy IDs without fallback", () => {
    const registry = createAiPresetRegistry(createAgentRegistry());

    expect(() => registry.resolve("missing-preset")).toThrow(UnknownAiPresetIdError);
    expect(() => registry.update(COM_AI_PRESET_ID, {
      playing: "missing-policy",
      bidding: "frozen-raise-v1",
      nonPlaying: "parameterized-adjutant-exchange-v1"
    } as AiPolicyComposition)).toThrow(InvalidAiPresetCompositionError);
    expect(registry.resolve(COM_AI_PRESET_ID).composition.playing).toBe(
      "ppo-separated-v1000"
    );
  });

  it("rejects a policy that becomes unavailable before save or resolution", () => {
    const base = createAgentRegistry();
    let learnedPlayingAvailable = true;
    const agentRegistry = {
      ...base,
      listPhasePolicies: () => ({
        ...base.listPhasePolicies(),
        playing: base.listPhasePolicies().playing.map((policy) =>
          policy.id === "ppo-separated-v1000"
            ? { ...policy, isAvailable: learnedPlayingAvailable }
            : policy
        )
      })
    };
    const registry = createAiPresetRegistry(agentRegistry);
    learnedPlayingAvailable = false;

    expect(() => registry.resolve(COM_AI_PRESET_ID)).toThrow(
      "Unavailable playing policy id"
    );
    expect(() => registry.update(COM_RULE_BASE_PRESET_ID, {
      playing: "ppo-separated-v1000",
      bidding: "rule-based",
      nonPlaying: "rule-based"
    })).toThrow(InvalidAiPresetCompositionError);
  });
});
