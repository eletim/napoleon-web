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

    const candidates = getRandomCandidates(observation.legalActions);
    const index = Math.floor(this.rng() * candidates.length);
    return candidates[Math.min(index, candidates.length - 1)];
  }
}

function getRandomCandidates(legalActions: readonly GameAction[]): readonly GameAction[] {
  const passAction = legalActions.find((action) => action.type === "pass");
  const firstBidAction = legalActions.find((action) => action.type === "bid");

  if (passAction !== undefined && firstBidAction !== undefined) {
    return [passAction, firstBidAction];
  }

  return legalActions;
}
