import type { GameAction, PlayerId, PlayerView } from "@napoleon/game-core";

export interface PlayerObservation {
  playerId: PlayerId;
  view: PlayerView;
  legalActions: readonly GameAction[];
}

export interface Agent {
  selectAction(observation: PlayerObservation): Promise<GameAction>;
}
