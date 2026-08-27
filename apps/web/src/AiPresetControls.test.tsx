import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type {
  AiPolicyComposition,
  AiPreset,
  AiPresetId,
  PublicPhasePolicyRegistry
} from "@napoleon/protocol";
import {
  AiSettingsPanel,
  GamePresetSelector,
  isPresetCompositionAvailable
} from "./AiPresetControls";

const presets: readonly AiPreset[] = [
  {
    id: "com-rule-base",
    displayName: "COM-RuleBase",
    composition: {
      playing: "rule-based",
      bidding: "rule-based",
      nonPlaying: "rule-based"
    }
  },
  {
    id: "com-ai",
    displayName: "COM-AI",
    composition: {
      playing: "ppo-separated-v1000",
      bidding: "frozen-raise-v1",
      nonPlaying: "parameterized-adjutant-exchange-v1"
    }
  }
];

const policyRegistry: PublicPhasePolicyRegistry = {
  playing: [policy("rule-based"), policy("ppo-separated-v1000")],
  bidding: [policy("rule-based"), policy("frozen-raise-v1")],
  nonPlaying: [policy("rule-based"), policy("parameterized-adjutant-exchange-v1")]
};

describe("AI preset controls", () => {
  it("shows only preset display names on the normal game start selector", () => {
    const html = renderToStaticMarkup(
      <GamePresetSelector
        disabled={false}
        onChange={() => undefined}
        presets={presets}
        selectedPresetId="com-ai"
      />
    );

    expect(html).toContain("対戦AI");
    expect(html).toContain("COM-RuleBase");
    expect(html).toContain("COM-AI");
    expect(html).not.toContain("ppo-separated-v1000");
    expect(html).not.toContain("frozen-raise-v1");
    expect(html).not.toContain("parameterized-adjutant-exchange-v1");
  });

  it("shows both presets and their editable phase policies in AI settings", () => {
    const drafts = Object.fromEntries(
      presets.map((preset) => [preset.id, preset.composition])
    ) as Record<AiPresetId, AiPolicyComposition>;
    const html = renderToStaticMarkup(
      <AiSettingsPanel
        disabled={false}
        drafts={drafts}
        message="設定済み"
        onChange={() => undefined}
        onSave={() => undefined}
        policyRegistry={policyRegistry}
        presets={presets}
      />
    );

    expect(html).toContain("AI設定");
    expect(html).toContain("COM-RuleBase");
    expect(html).toContain("COM-AI");
    expect(html).toContain("Playing policy");
    expect(html).toContain("Bidding policy");
    expect(html).toContain("Non-playing policy");
    expect(html).toContain("ppo-separated-v1000");
    expect(html).toContain("frozen-raise-v1");
    expect(html).toContain("parameterized-adjutant-exchange-v1");
    expect(html).toContain("保存・適用");
  });

  it("rejects unknown or unavailable composition entries", () => {
    expect(isPresetCompositionAvailable(presets[1].composition, policyRegistry)).toBe(true);
    expect(isPresetCompositionAvailable({
      ...presets[1].composition,
      playing: "missing-policy"
    } as unknown as AiPolicyComposition, policyRegistry)).toBe(false);
    expect(isPresetCompositionAvailable(presets[1].composition, {
      ...policyRegistry,
      playing: policyRegistry.playing.map((entry) =>
        entry.id === "ppo-separated-v1000" ? { ...entry, isAvailable: false } : entry
      )
    })).toBe(false);
  });
});

function policy(id: string) {
  return { id, displayName: id, isAvailable: true, artifactProvenance: null };
}
