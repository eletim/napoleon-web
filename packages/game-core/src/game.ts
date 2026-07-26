import { createDeck, shuffleDeck } from "./deck.js";
import { GameRuleError } from "./errors.js";
import { determineTrickWinner, getPlayableCards } from "./trick.js";
import { getLegalBidActions, isBidHigher, isSuit, validateBidRange } from "./bidding.js";
import type {
  Bid,
  Card,
  Contract,
  CreateInitialGameOptions,
  GameAction,
  GameState,
  PlayerId,
  PlayerState,
  PlayerView
} from "./types.js";

const playerCount = 5;
const defaultPlayerIds: readonly PlayerId[] = [
  "player-0",
  "player-1",
  "player-2",
  "player-3",
  "player-4"
];

export function createInitialGame(options: CreateInitialGameOptions = {}): GameState {
  const playerIds = options.playerIds ?? defaultPlayerIds;

  if (playerIds.length !== playerCount) {
    throw new GameRuleError("INVALID_PLAYER_COUNT", "A game must have exactly 5 players.");
  }

  const deck = shuffleDeck(createDeck(), options.rng);
  const cardsPerPlayer = Math.floor(deck.length / playerCount);
  const dealtCardCount = cardsPerPlayer * playerCount;
  const dealtCards = deck.slice(0, dealtCardCount);
  const unusedCards = deck.slice(dealtCardCount);

  const players = playerIds.map((id, playerIndex) => {
    const start = playerIndex * cardsPerPlayer;
    const hand = dealtCards.slice(start, start + cardsPerPlayer);
    return { id, hand };
  });

  return {
    players,
    phase: "bidding",
    currentPlayerId: playerIds[0],
    currentTrick: [],
    completedTricks: [],
    trumpSuit: null,
    contract: null,
    bidding: {
      starterPlayerId: playerIds[0],
      highestBid: null,
      consecutivePassCount: 0,
      history: []
    },
    trickNumber: 1,
    isTrickComplete: false,
    isGameOver: false,
    unusedCards
  };
}

export function getLegalActions(state: GameState, playerId: PlayerId): readonly GameAction[] {
  if (state.phase === "finished" || state.isGameOver) {
    return [];
  }

  if (state.currentPlayerId !== playerId) {
    return [];
  }

  if (state.phase === "bidding") {
    if (state.bidding === null) {
      return [];
    }

    return getLegalBidActions(playerId, state.bidding.highestBid);
  }

  if (state.phase !== "playing" || state.isTrickComplete) {
    return [];
  }

  const player = getPlayer(state, playerId);
  return getPlayableCards(player.hand, state.currentTrick).map((card) => ({
    type: "play-card",
    playerId,
    cardId: card.id
  }));
}

export function applyAction(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case "play-card":
      return playCard(state, action.playerId, action.cardId);
    case "bid":
      return bid(state, action);
    case "pass":
      return pass(state, action.playerId);
  }
}

export function advanceToNextTrick(state: GameState): GameState {
  if (state.phase === "finished" || state.isGameOver) {
    throw new GameRuleError("GAME_OVER", "The game is already over.");
  }

  if (state.phase !== "playing") {
    throw new GameRuleError(
      "INVALID_ACTION_FOR_PHASE",
      "This action is not allowed in the current game phase."
    );
  }

  if (!state.isTrickComplete) {
    throw new GameRuleError("TRICK_NOT_COMPLETE", "The current trick is not complete.");
  }

  return {
    ...state,
    currentTrick: [],
    trickNumber: state.trickNumber + 1,
    isTrickComplete: false
  };
}

export function createPlayerView(state: GameState, playerId: PlayerId): PlayerView {
  ensurePlayerExists(state, playerId);

  return {
    selfId: playerId,
    players: state.players.map((player) => ({
      id: player.id,
      handCount: player.hand.length,
      ...(player.id === playerId ? { hand: player.hand } : {})
    })),
    phase: state.phase,
    trumpSuit: state.trumpSuit,
    contract: state.contract,
    bidding: state.bidding,
    currentPlayerId: state.currentPlayerId,
    currentTrick: state.currentTrick,
    completedTrickCount: state.completedTricks.length,
    trickNumber: state.trickNumber,
    isTrickComplete: state.isTrickComplete,
    isGameOver: state.isGameOver,
    legalActions: getLegalActions(state, playerId)
  };
}

function playCard(state: GameState, playerId: PlayerId, cardId: string): GameState {
  if (state.phase === "finished" || state.isGameOver) {
    throw new GameRuleError("GAME_OVER", "The game is already over.");
  }

  if (state.phase !== "playing") {
    throw new GameRuleError(
      "INVALID_ACTION_FOR_PHASE",
      "This action is not allowed in the current game phase."
    );
  }

  if (state.isTrickComplete) {
    throw new GameRuleError("TRICK_COMPLETE", "Advance to the next trick before playing.");
  }

  if (state.currentPlayerId !== playerId) {
    throw new GameRuleError("NOT_PLAYERS_TURN", "It is not this player's turn.");
  }

  const player = getPlayer(state, playerId);
  const card = player.hand.find((candidate) => candidate.id === cardId);

  if (card === undefined) {
    throw new GameRuleError("CARD_NOT_IN_HAND", "The card is not in this player's hand.");
  }

  const playableCards = getPlayableCards(player.hand, state.currentTrick);

  if (!playableCards.some((candidate) => candidate.id === cardId)) {
    throw new GameRuleError("MUST_FOLLOW_SUIT", "The player must follow the lead suit.");
  }

  const players = state.players.map((candidate) =>
    candidate.id === playerId
      ? { ...candidate, hand: removeCard(candidate.hand, cardId) }
      : candidate
  );

  const currentTrick = [...state.currentTrick, { playerId, card }];
  const trickComplete = currentTrick.length === state.players.length;
  const winnerId = trickComplete
    ? determineTrickWinner(currentTrick, { trumpSuit: state.trumpSuit })
    : getNextPlayerId(state, playerId);
  const completedTricks = trickComplete
    ? [
        ...state.completedTricks,
        {
          trickNumber: state.trickNumber,
          winnerId,
          cards: currentTrick
        }
      ]
    : state.completedTricks;
  const isGameOver = trickComplete && players.every((candidate) => candidate.hand.length === 0);

  return {
    ...state,
    players,
    phase: isGameOver ? "finished" : state.phase,
    currentPlayerId: winnerId,
    currentTrick,
    completedTricks,
    isTrickComplete: trickComplete,
    isGameOver
  };
}

function bid(state: GameState, action: Extract<GameAction, { type: "bid" }>): GameState {
  ensureBiddingActionAllowed(state, action.playerId);

  if (!isSuit(action.suit)) {
    throw new GameRuleError("INVALID_BID", "Bid suit is invalid.");
  }

  validateBidRange(action.targetPointCards);

  const bidding = state.bidding;

  if (bidding === null) {
    throw new GameRuleError("BIDDING_NOT_AVAILABLE", "Bidding state is not available.");
  }

  const candidate: Bid = {
    playerId: action.playerId,
    suit: action.suit,
    targetPointCards: action.targetPointCards
  };

  if (bidding.highestBid !== null && !isBidHigher(candidate, bidding.highestBid)) {
    throw new GameRuleError("BID_TOO_LOW", "Bid must be higher than the current highest bid.");
  }

  return {
    ...state,
    currentPlayerId: getNextPlayerId(state, action.playerId),
    bidding: {
      ...bidding,
      highestBid: candidate,
      consecutivePassCount: 0,
      history: [
        ...bidding.history,
        {
          type: "bid",
          playerId: action.playerId,
          suit: action.suit,
          targetPointCards: action.targetPointCards
        }
      ]
    }
  };
}

function pass(state: GameState, playerId: PlayerId): GameState {
  ensureBiddingActionAllowed(state, playerId);

  const bidding = state.bidding;

  if (bidding === null) {
    throw new GameRuleError("BIDDING_NOT_AVAILABLE", "Bidding state is not available.");
  }

  const nextBidding = {
    ...bidding,
    consecutivePassCount: bidding.consecutivePassCount + 1,
    history: [
      ...bidding.history,
      {
        type: "pass" as const,
        playerId
      }
    ]
  };

  if (
    nextBidding.highestBid !== null &&
    nextBidding.consecutivePassCount === state.players.length - 1
  ) {
    return completeBidding(state, {
      napoleonPlayerId: nextBidding.highestBid.playerId,
      trumpSuit: nextBidding.highestBid.suit,
      targetPointCards: nextBidding.highestBid.targetPointCards
    });
  }

  if (
    nextBidding.highestBid === null &&
    nextBidding.consecutivePassCount === state.players.length
  ) {
    return completeBidding(state, {
      napoleonPlayerId: nextBidding.starterPlayerId,
      trumpSuit: "spades",
      targetPointCards: 12
    });
  }

  return {
    ...state,
    currentPlayerId: getNextPlayerId(state, playerId),
    bidding: nextBidding
  };
}

function ensureBiddingActionAllowed(state: GameState, playerId: PlayerId): void {
  if (state.phase === "finished" || state.isGameOver) {
    throw new GameRuleError("GAME_OVER", "The game is already over.");
  }

  if (state.phase !== "bidding") {
    throw new GameRuleError(
      "INVALID_ACTION_FOR_PHASE",
      "This action is not allowed in the current game phase."
    );
  }

  if (state.currentPlayerId !== playerId) {
    throw new GameRuleError("NOT_PLAYERS_TURN", "It is not this player's turn.");
  }
}

function completeBidding(state: GameState, contract: Contract): GameState {
  return {
    ...state,
    phase: "playing",
    currentPlayerId: contract.napoleonPlayerId,
    trumpSuit: contract.trumpSuit,
    contract,
    bidding: null
  };
}

function getPlayer(state: GameState, playerId: PlayerId): PlayerState {
  const player = state.players.find((candidate) => candidate.id === playerId);

  if (player === undefined) {
    throw new GameRuleError("PLAYER_NOT_FOUND", "Player was not found.");
  }

  return player;
}

function ensurePlayerExists(state: GameState, playerId: PlayerId): void {
  getPlayer(state, playerId);
}

function removeCard(cards: readonly Card[], cardId: string): readonly Card[] {
  return cards.filter((card) => card.id !== cardId);
}

function getNextPlayerId(state: GameState, playerId: PlayerId): PlayerId {
  const playerIndex = state.players.findIndex((player) => player.id === playerId);

  if (playerIndex === -1) {
    throw new GameRuleError("PLAYER_NOT_FOUND", "Player was not found.");
  }

  return state.players[(playerIndex + 1) % state.players.length].id;
}
