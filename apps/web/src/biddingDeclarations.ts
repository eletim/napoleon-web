import type { PublicBiddingState, PublicSuit } from "@napoleon/protocol";
import { isRedSuit, suitSymbols } from "./cardSymbols";

export type BiddingDeclarationDisplay =
  | {
      type: "bid";
      label: string;
      suit: PublicSuit;
      color: "red" | "black";
    }
  | {
      type: "pass";
      label: "パス";
    }
  | {
      type: "none";
      label: "未宣言";
    };

export const emptyBiddingDeclaration: BiddingDeclarationDisplay = {
  type: "none",
  label: "未宣言"
};

export function createLatestBiddingDeclarations(
  bidding: PublicBiddingState | null | undefined
): ReadonlyMap<string, BiddingDeclarationDisplay> {
  const declarations = new Map<string, BiddingDeclarationDisplay>();

  for (const entry of bidding?.history ?? []) {
    if (entry.type === "pass") {
      declarations.set(entry.playerId, {
        type: "pass",
        label: "パス"
      });
      continue;
    }

    declarations.set(entry.playerId, {
      type: "bid",
      label: `${suitSymbols[entry.suit]} ${entry.targetPointCards}`,
      suit: entry.suit,
      color: isRedSuit(entry.suit) ? "red" : "black"
    });
  }

  return declarations;
}
