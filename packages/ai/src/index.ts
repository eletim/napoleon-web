import type { GameAction, PlayerId, PlayerView } from "@napoleon/game-core";

export interface PlayerObservation {
  playerId: PlayerId;
  view: PlayerView;
  legalActions: readonly GameAction[];
}

export interface Agent {
  selectAction(observation: PlayerObservation): Promise<GameAction>;
}

export class NoLegalActionsError extends Error {
  constructor(playerId: PlayerId) {
    super(`No legal actions are available for ${playerId}.`);
    this.name = "NoLegalActionsError";
  }
}

export class RandomAgent implements Agent {
  private readonly rng: () => number;

  constructor(rng: () => number = Math.random) {
    this.rng = rng;
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    if (observation.legalActions.length === 0) {
      throw new NoLegalActionsError(observation.playerId);
    }

    const index = Math.floor(this.rng() * observation.legalActions.length);
    return observation.legalActions[Math.min(index, observation.legalActions.length - 1)];
  }
}
