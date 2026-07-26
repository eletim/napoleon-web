import {
  createPlayerView,
  type Card,
  type GameAction,
  type GameState,
  type PlayedCard,
  type PlayerId
} from "@napoleon/game-core";
import type {
  PublicCard,
  PublicGameState,
  PublicLegalAction,
  PublicOpponentPlayer,
  PublicPlayedCard,
  PublicSelfPlayer
} from "@napoleon/protocol";

export function toPublicGameState(state: GameState, playerId: PlayerId): PublicGameState {
  const view = createPlayerView(state, playerId);
  const selfPlayer = view.players.find((player) => player.id === playerId);

  if (selfPlayer?.hand === undefined) {
    throw new Error(`Public self view was not available for ${playerId}.`);
  }

  const self: PublicSelfPlayer = {
    id: selfPlayer.id,
    handCount: selfPlayer.handCount,
    hand: selfPlayer.hand.map(toPublicCard)
  };

  const opponents: readonly PublicOpponentPlayer[] = view.players
    .filter((player) => player.id !== playerId)
    .map((player) => ({
      id: player.id,
      handCount: player.handCount
    }));

  return {
    self,
    opponents,
    currentPlayerId: view.currentPlayerId,
    currentTrick: view.currentTrick.map(toPublicPlayedCard),
    completedTrickCount: view.completedTrickCount,
    trickNumber: view.trickNumber,
    isTrickComplete: view.isTrickComplete,
    isGameOver: view.isGameOver,
    legalActions: view.legalActions.map(toPublicLegalAction)
  };
}

function toPublicCard(card: Card): PublicCard {
  return {
    id: card.id,
    suit: card.suit,
    rank: card.rank
  };
}

function toPublicPlayedCard(playedCard: PlayedCard): PublicPlayedCard {
  return {
    playerId: playedCard.playerId,
    card: toPublicCard(playedCard.card)
  };
}

function toPublicLegalAction(action: GameAction): PublicLegalAction {
  return {
    type: action.type,
    cardId: action.cardId
  };
}
