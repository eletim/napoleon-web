import { isStandardCard } from "@napoleon/game-core";
import type { GameAction, PlayerId, PlayerView, Rank, Suit } from "@napoleon/game-core";

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

    if (observation.view.phase === "choosing-adjutant") {
      const self = observation.view.players.find((player) => player.id === observation.playerId);

      if (observation.view.adjutantChoiceRequirement !== null && self?.hand !== undefined) {
        const selfStandardCardIds = new Set(
          self.hand.filter(isStandardCard).map((card) => card.id)
        );
        const preferredCardId =
          standardAdjutantCardIds.find((cardId) => !selfStandardCardIds.has(cardId)) ??
          standardAdjutantCardIds[0];

        return {
          type: "choose-adjutant",
          playerId: observation.playerId,
          cardId: preferredCardId
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

const adjutantSuitPreference: readonly Suit[] = ["spades", "hearts", "diamonds", "clubs"];
const adjutantRankPreference: readonly Rank[] = [
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

const standardAdjutantCardIds = adjutantSuitPreference.flatMap((suit) =>
  adjutantRankPreference.map((rank) => `${suit}-${rank}`)
);

function getRandomCandidates(legalActions: readonly GameAction[]): readonly GameAction[] {
  const passAction = legalActions.find((action) => action.type === "pass");
  const firstBidAction = legalActions.find((action) => action.type === "bid");

  if (passAction !== undefined && firstBidAction !== undefined) {
    return [passAction, firstBidAction];
  }

  return legalActions;
}
