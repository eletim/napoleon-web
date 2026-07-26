import { isPointCard } from "./cards.js";
import { GameRuleError } from "./errors.js";
import type { GameResult, GameState, PlayerId } from "./types.js";

const expectedCompletedTrickCount = 10;
const expectedBuriedCardCount = 3;
const expectedPointCardCount = 20;

export function calculateGameResult(state: GameState): GameResult {
  if (state.contract === null || state.adjutant === null) {
    throw new GameRuleError("INVALID_RESULT_STATE", "A contract and adjutant state are required.");
  }

  if (state.completedTricks.length !== expectedCompletedTrickCount) {
    throw new GameRuleError("INVALID_RESULT_STATE", "All 10 tricks must be completed.");
  }

  if (!state.players.every((player) => player.hand.length === 0)) {
    throw new GameRuleError("INVALID_RESULT_STATE", "All player hands must be empty.");
  }

  if (state.buriedCards.length !== expectedBuriedCardCount) {
    throw new GameRuleError("INVALID_RESULT_STATE", "Exactly 3 buried cards are required.");
  }

  const napoleonTeamPlayerIds = new Set<PlayerId>([state.contract.napoleonPlayerId]);

  if (state.adjutant.playerId !== null) {
    napoleonTeamPlayerIds.add(state.adjutant.playerId);
  }

  let napoleonTeamPointCards = 0;
  let alliancePointCards = 0;

  for (const trick of state.completedTricks) {
    if (trick.cards.length !== state.players.length) {
      throw new GameRuleError("INVALID_RESULT_STATE", "Every completed trick must contain one card per player.");
    }

    const trickPointCardCount = trick.cards.filter((playedCard) => isPointCard(playedCard.card)).length;

    if (napoleonTeamPlayerIds.has(trick.winnerId)) {
      napoleonTeamPointCards += trickPointCardCount;
    } else {
      alliancePointCards += trickPointCardCount;
    }
  }

  const buriedPointCards = state.buriedCards.filter(isPointCard).length;
  napoleonTeamPointCards += buriedPointCards;

  if (napoleonTeamPointCards + alliancePointCards !== expectedPointCardCount) {
    throw new GameRuleError(
      "POINT_CARD_COUNT_MISMATCH",
      "The total point card count must be exactly 20."
    );
  }

  return {
    winner:
      napoleonTeamPointCards >= state.contract.targetPointCards
        ? "napoleon-team"
        : "alliance",
    napoleonTeamPointCards,
    alliancePointCards,
    buriedPointCards,
    targetPointCards: state.contract.targetPointCards,
    napoleonPlayerId: state.contract.napoleonPlayerId,
    adjutantPlayerId: state.adjutant.playerId
  };
}
