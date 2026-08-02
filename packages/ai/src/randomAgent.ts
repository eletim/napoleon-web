import type { GameAction, Rank, Suit } from "@napoleon/game-core";
import { NoLegalActionsError } from "./errors.js";
import type { Agent, PlayerObservation } from "./types.js";
import { selectRandom } from "./random.js";

export class RandomAgent implements Agent {
  private readonly rng: () => number;

  constructor(rng: () => number = Math.random) {
    this.rng = rng;
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    if (observation.view.phase === "choosing-adjutant") {
      const self = observation.view.players.find((player) => player.id === observation.playerId);

      if (observation.view.adjutantChoiceRequirement !== null && self?.hand !== undefined) {
        const selfCardIds = new Set(self.hand.map((card) => card.id));
        const candidateCardIds = observation.view.adjutantChoiceRequirement.jokerAllowed
          ? adjutantCardIds
          : standardAdjutantCardIds;
        const preferredCardId =
          candidateCardIds.find((cardId) => !selfCardIds.has(cardId)) ?? candidateCardIds[0];

        return {
          type: "choose-adjutant",
          playerId: observation.playerId,
          cardId: preferredCardId
        };
      }
    }

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

    return selectRandom(getRandomCandidates(observation.legalActions), this.rng);
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

const standardAdjutantCardIds = adjutantRankPreference.flatMap((rank) =>
  adjutantSuitPreference.map((suit) => `${suit}-${rank}`)
);
const adjutantCardIds = [...standardAdjutantCardIds, "joker"];

function getRandomCandidates(legalActions: readonly GameAction[]): readonly GameAction[] {
  const passAction = legalActions.find((action) => action.type === "pass");
  const firstBidAction = legalActions.find((action) => action.type === "bid");

  if (passAction !== undefined && firstBidAction !== undefined) {
    return [passAction, firstBidAction];
  }

  return legalActions;
}
