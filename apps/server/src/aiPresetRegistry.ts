import type {
  AiPolicyComposition,
  AiPreset,
  AiPresetId,
  PublicPhasePolicyRegistry
} from "@napoleon/protocol";
import type { AgentRegistry } from "./agentRegistry.js";

export const COM_RULE_BASE_PRESET_ID = "com-rule-base" as const;
export const COM_AI_PRESET_ID = "com-ai" as const;

export const BUILTIN_AI_PRESETS: readonly AiPreset[] = [
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
] as const;

export interface AiPresetRegistry {
  list(): readonly AiPreset[];
  resolve(presetId: string): AiPreset;
  update(presetId: string, composition: AiPolicyComposition): AiPreset;
}

export class UnknownAiPresetIdError extends Error {
  constructor(readonly presetId: string) {
    super(`Unknown AI preset id: ${presetId}.`);
    this.name = "UnknownAiPresetIdError";
  }
}

export class InvalidAiPresetCompositionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InvalidAiPresetCompositionError";
  }
}

export function createAiPresetRegistry(agentRegistry: AgentRegistry): AiPresetRegistry {
  const presets = new Map<AiPresetId, AiPreset>(
    BUILTIN_AI_PRESETS.map((preset) => [preset.id, clonePreset(preset)])
  );
  validateAllPresets(presets.values(), agentRegistry.listPhasePolicies());

  return {
    list: () => [...presets.values()].map(clonePreset),
    resolve: (presetId) => {
      const preset = presets.get(readKnownPresetId(presetId));
      if (preset === undefined) {
        throw new UnknownAiPresetIdError(presetId);
      }
      validatePresetComposition(preset.composition, agentRegistry.listPhasePolicies());
      return clonePreset(preset);
    },
    update: (presetId, composition) => {
      const id = readKnownPresetId(presetId);
      const current = presets.get(id);
      if (current === undefined) {
        throw new UnknownAiPresetIdError(presetId);
      }
      validatePresetComposition(composition, agentRegistry.listPhasePolicies());
      const updated = { ...current, composition: { ...composition } };
      presets.set(id, updated);
      return clonePreset(updated);
    }
  };
}

export function validatePresetComposition(
  composition: AiPolicyComposition,
  policies: PublicPhasePolicyRegistry
): void {
  assertAvailable("playing", composition.playing, policies.playing);
  assertAvailable("bidding", composition.bidding, policies.bidding);
  assertAvailable("nonPlaying", composition.nonPlaying, policies.nonPlaying);
}

function validateAllPresets(
  presets: Iterable<AiPreset>,
  policies: PublicPhasePolicyRegistry
): void {
  for (const preset of presets) {
    validatePresetComposition(preset.composition, policies);
  }
}

function assertAvailable(
  phase: "playing" | "bidding" | "nonPlaying",
  policyId: string,
  policies: PublicPhasePolicyRegistry[typeof phase]
): void {
  const policy = policies.find((candidate) => candidate.id === policyId);
  if (policy === undefined) {
    throw new InvalidAiPresetCompositionError(
      `Unknown ${phase} policy id in AI preset: ${policyId}.`
    );
  }
  if (!policy.isAvailable) {
    throw new InvalidAiPresetCompositionError(
      `Unavailable ${phase} policy id in AI preset: ${policyId}.`
    );
  }
}

function readKnownPresetId(value: string): AiPresetId {
  if (value !== COM_RULE_BASE_PRESET_ID && value !== COM_AI_PRESET_ID) {
    throw new UnknownAiPresetIdError(value);
  }
  return value;
}

function clonePreset(preset: AiPreset): AiPreset {
  return { ...preset, composition: { ...preset.composition } };
}
