import type { PublicGameState, PublicRank, PublicSuit } from "@napoleon/protocol";
import { formatCardId } from "./displayText";

export type AdjutantSuitOption = PublicSuit | "joker";
export type AdjutantShortcutId = "oruma" | "yoromeki" | "seiJack" | "uraJack";

export interface AdjutantSelection {
  suitOption: AdjutantSuitOption;
  rank: PublicRank;
  shortcutId: AdjutantShortcutId | null;
}

export interface AdjutantShortcutOption {
  id: AdjutantShortcutId;
  label: string;
  cardId: string;
  display: string;
}

export const defaultAdjutantSelection: AdjutantSelection = {
  suitOption: "spades",
  rank: "A",
  shortcutId: null
};

export function createAdjutantCardId(selection: AdjutantSelection): string {
  return selection.suitOption === "joker"
    ? "joker"
    : `${selection.suitOption}-${selection.rank}`;
}

export function createAdjutantSelectionLabel(
  selection: AdjutantSelection,
  shortcuts: readonly AdjutantShortcutOption[]
): string {
  const cardId = createAdjutantCardId(selection);
  const shortcut =
    selection.shortcutId === null
      ? undefined
      : shortcuts.find(
          (candidate) =>
            candidate.id === selection.shortcutId && candidate.cardId === cardId
        );

  return shortcut === undefined
    ? formatCardId(cardId)
    : `${shortcut.label} ${shortcut.display}`;
}

export function createAdjutantShortcutOptions(
  specialCards: PublicGameState["specialCards"] | undefined
): readonly AdjutantShortcutOption[] {
  if (specialCards === undefined) {
    return [];
  }

  const rawShortcuts: readonly {
    id: AdjutantShortcutId;
    label: string;
    cardId: string | null;
  }[] = [
    { id: "oruma", label: "オルマ", cardId: specialCards.orumaCardId },
    { id: "yoromeki", label: "よろめき", cardId: specialCards.yoromekiCardId },
    { id: "seiJack", label: "正J", cardId: specialCards.seiJackCardId },
    { id: "uraJack", label: "裏J", cardId: specialCards.uraJackCardId }
  ];

  return rawShortcuts.flatMap((shortcut) =>
    shortcut.cardId === null
      ? []
      : [
          {
            id: shortcut.id,
            label: shortcut.label,
            cardId: shortcut.cardId,
            display: formatCardId(shortcut.cardId)
          }
        ]
  );
}

export function selectAdjutantSuitOption(
  current: AdjutantSelection,
  suitOption: AdjutantSuitOption
): AdjutantSelection {
  return {
    ...current,
    suitOption,
    shortcutId: null
  };
}

export function selectAdjutantRank(
  current: AdjutantSelection,
  rank: PublicRank
): AdjutantSelection {
  return {
    suitOption: current.suitOption === "joker" ? "spades" : current.suitOption,
    rank,
    shortcutId: null
  };
}

export function selectAdjutantShortcut(
  current: AdjutantSelection,
  shortcut: AdjutantShortcutOption
): AdjutantSelection {
  if (shortcut.cardId === "joker") {
    return {
      ...current,
      suitOption: "joker",
      shortcutId: shortcut.id
    };
  }

  const [suit, rank] = shortcut.cardId.split("-");

  if (!isPublicSuit(suit) || !isPublicRank(rank)) {
    return current;
  }

  return {
    suitOption: suit,
    rank,
    shortcutId: shortcut.id
  };
}

function isPublicSuit(value: string | undefined): value is PublicSuit {
  return (
    value === "spades" ||
    value === "hearts" ||
    value === "diamonds" ||
    value === "clubs"
  );
}

const publicRanks: readonly PublicRank[] = [
  "A",
  "K",
  "Q",
  "J",
  "10",
  "9",
  "8",
  "7",
  "6",
  "5",
  "4",
  "3",
  "2"
];

function isPublicRank(value: string | undefined): value is PublicRank {
  return publicRanks.some((rank) => rank === value);
}
