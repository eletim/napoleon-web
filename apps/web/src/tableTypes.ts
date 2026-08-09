import type { PublicStandardCard } from "@napoleon/protocol";
import type { BiddingDeclarationDisplay } from "./biddingDeclarations";

export type Seat = "left" | "top-left" | "top-right" | "right" | "self";

export interface TablePlayer {
  id: string;
  label: string;
  seat: Seat;
  handCount: number;
  capturedPointCards: readonly PublicStandardCard[];
  isSelf: boolean;
  biddingDeclaration?: BiddingDeclarationDisplay;
}
