import {
  createPlayerView,
  isJokerCard,
  type Bid,
  type BiddingHistoryEntry,
  type BiddingState,
  type Card,
  type Contract,
  type GameAction,
  type GameResult,
  type GameState,
  type PlayedCard,
  type PlayerId,
  type StandardCard
} from "@napoleon/game-core";
import type {
  PublicCard,
  PublicBid,
  PublicBiddingHistoryEntry,
  PublicBiddingState,
  PublicContract,
  PublicAdjutantChoiceState,
  PublicExchangeState,
  PublicGameState,
  PublicGameResult,
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
    hand: selfPlayer.hand.map(toPublicCard),
    capturedPointCards: selfPlayer.capturedPointCards.map(toPublicStandardCard)
  };

  const opponents: readonly PublicOpponentPlayer[] = view.players
    .filter((player) => player.id !== playerId)
    .map((player) => ({
      id: player.id,
      handCount: player.handCount,
      capturedPointCards: player.capturedPointCards.map(toPublicStandardCard)
    }));

  return {
    self,
    opponents,
    phase: view.phase,
    trumpSuit: view.trumpSuit,
    contract: view.contract === null ? null : toPublicContract(view.contract),
    specialCards: {
      orumaCardId: view.specialCards.orumaCardId,
      yoromekiCardId: view.specialCards.yoromekiCardId,
      seiJackCardId: view.specialCards.seiJackCardId,
      uraJackCardId: view.specialCards.uraJackCardId
    },
    adjutant:
      view.adjutant === null
        ? null
        : {
            calledCardId: view.adjutant.calledCardId,
            revealedPlayerId: view.adjutant.revealedPlayerId
          },
    buriedCards:
      view.buriedCards === null
        ? null
        : {
            revealedPointCards: view.buriedCards.revealedPointCards.map(toPublicStandardCard),
            hiddenCardCount: view.buriedCards.hiddenCardCount
          },
    result: view.result === null ? null : toPublicGameResult(view.result),
    bidding: view.bidding === null ? null : toPublicBiddingState(view.bidding),
    exchange: toPublicExchangeState(view),
    adjutantChoice: toPublicAdjutantChoiceState(view),
    currentPlayerId: view.currentPlayerId,
    currentTrick: view.currentTrick.map(toPublicPlayedCard),
    completedTrickCount: view.completedTrickCount,
    trickNumber: view.trickNumber,
    isTrickComplete: view.isTrickComplete,
    isGameOver: view.isGameOver,
    legalActions: view.legalActions.map(toPublicLegalAction)
  };
}

function toPublicAdjutantChoiceState(
  view: ReturnType<typeof createPlayerView>
): PublicAdjutantChoiceState | null {
  if (view.phase !== "choosing-adjutant" || view.contract === null) {
    return null;
  }

  return {
    napoleonPlayerId: view.contract.napoleonPlayerId,
    standardCardsOnly: true
  };
}

function toPublicExchangeState(
  view: ReturnType<typeof createPlayerView>
): PublicExchangeState | null {
  if (view.phase !== "exchanging" || view.contract === null) {
    return null;
  }

  return {
    napoleonPlayerId: view.contract.napoleonPlayerId,
    requiredDiscardCount: view.exchangeRequirement?.discardCount ?? 3
  };
}

function toPublicCard(card: Card): PublicCard {
  if (isJokerCard(card)) {
    return {
      type: "joker",
      id: card.id
    };
  }

  return toPublicStandardCard(card);
}

function toPublicStandardCard(card: StandardCard): Extract<PublicCard, { type: "standard" }> {
  return {
    type: "standard",
    id: card.id,
    suit: card.suit,
    rank: card.rank
  };
}

function toPublicGameResult(result: GameResult): PublicGameResult {
  return {
    winner: result.winner,
    napoleonTeamPointCards: result.napoleonTeamPointCards,
    alliancePointCards: result.alliancePointCards,
    buriedPointCards: result.buriedPointCards,
    targetPointCards: result.targetPointCards,
    napoleonPlayerId: result.napoleonPlayerId,
    adjutantPlayerId: result.adjutantPlayerId
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
    case "discard-cards":
      throw new Error("Discard action should not be exposed as an enumerated legal action.");
    case "choose-adjutant":
      throw new Error("Choose-adjutant action should not be exposed as an enumerated legal action.");
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
