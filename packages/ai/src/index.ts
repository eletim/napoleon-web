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
    if (observation.view.phase === "exchanging") {
      const discardCount = observation.view.exchangeRequirement?.discardCount;
      const self = observation.view.players.find((player) => player.id === observation.playerId);

      if (discardCount !== undefined && self?.hand !== undefined) {
        return {
          type: "discard-cards",
          playerId: observation.playerId,
          cardIds: self.hand.slice(0, discardCount).map((card) => card.id)
        };
      }
    }

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
