import {
  createPlayerView,
  isJokerCard,
  type Bid,
  type BiddingHistoryEntry,
  type BiddingState,
  type Card,
  type Contract,
  type GameAction,
  type GameState,
  type PlayedCard,
  type PlayerId
} from "@napoleon/game-core";
import type {
  PublicCard,
  PublicBid,
  PublicBiddingHistoryEntry,
  PublicBiddingState,
  PublicContract,
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
    phase: view.phase,
    trumpSuit: view.trumpSuit,
    contract: view.contract === null ? null : toPublicContract(view.contract),
    bidding: view.bidding === null ? null : toPublicBiddingState(view.bidding),
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
  if (isJokerCard(card)) {
    return {
      type: "joker",
      id: card.id
    };
  }

  return {
    type: "standard",
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
  switch (action.type) {
    case "play-card":
      return {
        type: "play-card",
        cardId: action.cardId
      };
    case "bid":
      return {
        type: "bid",
        suit: action.suit,
        targetPointCards: action.targetPointCards
      };
    case "pass":
      return {
        type: "pass"
      };
  }
}

function toPublicBid(bid: Bid): PublicBid {
  return {
    playerId: bid.playerId,
    suit: bid.suit,
    targetPointCards: bid.targetPointCards
  };
}

function toPublicContract(contract: Contract): PublicContract {
  return {
    napoleonPlayerId: contract.napoleonPlayerId,
    trumpSuit: contract.trumpSuit,
    targetPointCards: contract.targetPointCards
  };
}

function toPublicBiddingState(bidding: BiddingState): PublicBiddingState {
  return {
    starterPlayerId: bidding.starterPlayerId,
    highestBid: bidding.highestBid === null ? null : toPublicBid(bidding.highestBid),
    consecutivePassCount: bidding.consecutivePassCount,
    history: bidding.history.map(toPublicBiddingHistoryEntry)
  };
}

function toPublicBiddingHistoryEntry(
  entry: BiddingHistoryEntry
): PublicBiddingHistoryEntry {
  switch (entry.type) {
    case "bid":
      return {
        type: "bid",
        playerId: entry.playerId,
        suit: entry.suit,
        targetPointCards: entry.targetPointCards
      };
    case "pass":
      return {
        type: "pass",
        playerId: entry.playerId
      };
  }
}
