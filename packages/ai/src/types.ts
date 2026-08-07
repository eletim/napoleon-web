import type { GameAction, GamePhase, PlayerId, PlayerView } from "@napoleon/game-core";

export interface PublicActionRecord {
  step: number;
  playerId: PlayerId;
  phase: GamePhase;
  action: GameAction;
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
