import type { BidAction, PassAction, GameAction, PlayerId, PlayerView } from "@napoleon/game-core";

export interface PublicActionRecord {
  /**
   * Monotonic ordering value from the producer. Consumers must use array order
   * for bidding-history semantics instead of comparing step values across
   * different producers.
   */
  step: number;
  playerId: PlayerId;
  phase: "bidding";
  action: BidAction | PassAction;
}

export interface PlayerObservation {
  playerId: PlayerId;
  view: PlayerView;
  legalActions: readonly GameAction[];
  publicActionHistory?: readonly PublicActionRecord[];
}

export interface Agent {
  selectAction(observation: PlayerObservation): Promise<GameAction>;
}
