// @vitest-environment happy-dom

import { act } from "react";
import { createRoot } from "react-dom/client";
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
  SeatPresetSelector,
  isPresetCompositionAvailable
} from "./AiPresetControls";

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

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

const seats = [
  { id: "player-1", label: "左側AI" },
  { id: "player-2", label: "奥左AI" },
  { id: "player-3", label: "奥右AI" },
  { id: "player-4", label: "右側AI" }
];

describe("AI preset controls", () => {
  it("shows an independent preset selector per seat with only preset display names", () => {
    const html = renderToStaticMarkup(
      <SeatPresetSelector
        disabled={false}
        onChange={() => undefined}
        presets={presets}
        seats={seats}
        selections={{
          "player-1": "com-rule-base",
          "player-2": "com-ai",
          "player-3": "com-rule-base",
          "player-4": "com-ai"
        }}
      />
    );

    expect(html).toContain("対戦AI");
    for (const seat of seats) {
      expect(html).toContain(seat.label);
    }
    expect(html).toContain("COM-RuleBase");
    expect(html).toContain("COM-AI");
    expect(html).not.toContain("ppo-separated-v1000");
    expect(html).not.toContain("frozen-raise-v1");
    expect(html).not.toContain("parameterized-adjutant-exchange-v1");

    // Each seat's <select> marks only its own preset's <option> as selected.
    const selects = [...html.matchAll(/<select[^>]*>[\s\S]*?<\/select>/g)].map((match) => match[0]);
    expect(selects).toHaveLength(4);
    const expectedSelectedIds = ["com-rule-base", "com-ai", "com-rule-base", "com-ai"];
    selects.forEach((select, index) => {
      expect(selectedOptionValue(select)).toBe(expectedSelectedIds[index]);
    });
  });

  it("changes only the edited seat's preset, independently of the other three", async () => {
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    const changes: Array<[string, string]> = [];

    await act(async () => {
      root.render(
        <SeatPresetSelector
          disabled={false}
          onChange={(playerId, presetId) => changes.push([playerId, presetId])}
          presets={presets}
          seats={seats}
          selections={{
            "player-1": "com-ai",
            "player-2": "com-ai",
            "player-3": "com-ai",
            "player-4": "com-ai"
          }}
        />
      );
    });

    const select = container.querySelector<HTMLSelectElement>(
      '[aria-label="奥右AIの対戦AI"]'
    );
    expect(select).not.toBeNull();

    await act(async () => {
      if (select !== null) {
        select.value = "com-rule-base";
        select.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });

    expect(changes).toEqual([["player-3", "com-rule-base"]]);

    await act(async () => root.unmount());
    container.remove();
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

function selectedOptionValue(selectHtml: string): string | undefined {
  const options = [...selectHtml.matchAll(/<option[^>]*value="([^"]*)"[^>]*>/g)];
  const selected = options.find(([optionHtml]) => optionHtml.includes("selected"));
  return selected?.[1];
}
