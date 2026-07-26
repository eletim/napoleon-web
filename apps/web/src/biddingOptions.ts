import type { PublicBidAction, PublicSuit } from "@napoleon/protocol";

export interface BidSelection {
  suit: PublicSuit;
  targetPointCards: number;
}

export const biddingSuitOptions: readonly PublicSuit[] = [
  "spades",
  "clubs",
  "hearts",
  "diamonds"
];

export function getBidTargetsForSuit(
  legalBidActions: readonly PublicBidAction[],
  suit: PublicSuit
): readonly number[] {
  return Array.from(
    new Set(
      legalBidActions
        .filter((action) => action.suit === suit)
        .map((action) => action.targetPointCards)
    )
  ).sort((left, right) => left - right);
}

export function findBidAction(
  legalBidActions: readonly PublicBidAction[],
  selection: BidSelection | null
): PublicBidAction | undefined {
  if (selection === null) {
    return undefined;
  }

  return legalBidActions.find(
    (action) =>
      action.suit === selection.suit &&
      action.targetPointCards === selection.targetPointCards
  );
}

export function normalizeBidSelection(
  legalBidActions: readonly PublicBidAction[],
  selection: BidSelection | null
): BidSelection | null {
  if (legalBidActions.length === 0) {
    return null;
  }

  if (findBidAction(legalBidActions, selection) !== undefined) {
    return selection;
  }

  if (selection !== null) {
    const targetsForSelectedSuit = getBidTargetsForSuit(legalBidActions, selection.suit);

    if (targetsForSelectedSuit.length > 0) {
      return {
        suit: selection.suit,
        targetPointCards: targetsForSelectedSuit[0]
      };
    }
  }

  const firstAction = legalBidActions[0];

  return {
    suit: firstAction.suit,
    targetPointCards: firstAction.targetPointCards
  };
}

export function getBidStepperState(
  legalBidActions: readonly PublicBidAction[],
  selection: BidSelection | null
): {
  targets: readonly number[];
  selectedIndex: number;
  previousTarget: number | null;
  nextTarget: number | null;
} {
  if (selection === null) {
    return {
      targets: [],
      selectedIndex: -1,
      previousTarget: null,
      nextTarget: null
    };
  }

  const targets = getBidTargetsForSuit(legalBidActions, selection.suit);
  const selectedIndex = targets.findIndex(
    (targetPointCards) => targetPointCards === selection.targetPointCards
  );

  return {
    targets,
    selectedIndex,
    previousTarget: selectedIndex > 0 ? targets[selectedIndex - 1] : null,
    nextTarget:
      selectedIndex >= 0 && selectedIndex < targets.length - 1
        ? targets[selectedIndex + 1]
        : null
  };
}
