import { describe, expect, it } from "vitest";
import type { PublicGameState } from "@napoleon/protocol";
import {
  createAdjutantCardId,
  createAdjutantSelectionLabel,
  createAdjutantShortcutOptions,
  defaultAdjutantSelection,
  selectAdjutantRank,
  selectAdjutantShortcut,
  selectAdjutantSuitOption
} from "./adjutantSelection";

describe("adjutant selection display helpers", () => {
  it("creates a standard card id from suit and rank", () => {
    expect(createAdjutantCardId(defaultAdjutantSelection)).toBe("spades-A");
    expect(
      createAdjutantCardId({
        suitOption: "hearts",
        rank: "Q",
        shortcutId: null
      })
    ).toBe("hearts-Q");
  });

  it("creates the joker card id when joker is selected", () => {
    const selection = selectAdjutantSuitOption(defaultAdjutantSelection, "joker");

    expect(createAdjutantCardId(selection)).toBe("joker");
    expect(createAdjutantSelectionLabel(selection, [])).toBe("ジョーカー");
  });

  it("creates special-card shortcut options from public special card ids", () => {
    expect(createAdjutantShortcutOptions(createSpecialCards())).toEqual([
      { id: "oruma", label: "オルマ", cardId: "spades-A", display: "A♠" },
      { id: "yoromeki", label: "よろめき", cardId: "hearts-Q", display: "Q♥" },
      { id: "seiJack", label: "正J", cardId: "diamonds-J", display: "J♦" },
      { id: "uraJack", label: "裏J", cardId: "hearts-J", display: "J♥" }
    ]);
  });

  it("omits shortcut options whose card id is not decided", () => {
    expect(
      createAdjutantShortcutOptions({
        ...createSpecialCards(),
        seiJackCardId: null,
        uraJackCardId: null
      }).map((option) => option.id)
    ).toEqual(["oruma", "yoromeki"]);
  });

  it("selects a shortcut and keeps the manual controls in sync", () => {
    const shortcuts = createAdjutantShortcutOptions(createSpecialCards());
    const shortcut = shortcuts.find((option) => option.id === "uraJack");

    expect(shortcut).toBeDefined();
    if (shortcut === undefined) {
      throw new Error("Expected ura jack shortcut.");
    }

    const selection = selectAdjutantShortcut(defaultAdjutantSelection, shortcut);

    expect(selection).toEqual({
      suitOption: "hearts",
      rank: "J",
      shortcutId: "uraJack"
    });
    expect(createAdjutantCardId(selection)).toBe("hearts-J");
    expect(createAdjutantSelectionLabel(selection, shortcuts)).toBe("裏J J♥");
  });

  it("clears shortcut selection when manually changing suit or rank", () => {
    const shortcut = createAdjutantShortcutOptions(createSpecialCards())[0];

    if (shortcut === undefined) {
      throw new Error("Expected shortcut.");
    }

    const selected = selectAdjutantShortcut(defaultAdjutantSelection, shortcut);

    expect(selectAdjutantSuitOption(selected, "clubs").shortcutId).toBeNull();
    expect(selectAdjutantRank(selected, "K")).toEqual({
      suitOption: "spades",
      rank: "K",
      shortcutId: null
    });
  });
});

function createSpecialCards(): PublicGameState["specialCards"] {
  return {
    orumaCardId: "spades-A",
    yoromekiCardId: "hearts-Q",
    seiJackCardId: "diamonds-J",
    uraJackCardId: "hearts-J"
  };
}
