import { createDeck, shuffleDeck } from "./deck.js";
import { GameRuleError } from "./errors.js";
import { determineTrickWinner, getPlayableCards } from "./trick.js";
import { getLegalBidActions, isBidHigher, isSuit, validateBidRange } from "./bidding.js";
import { isStandardCardId } from "./cards.js";
import type {
  AdjutantState,
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
    adjutant: null,
    bidding: {
      starterPlayerId: playerIds[0],
      highestBid: null,
      consecutivePassCount: 0,
      history: []
    },
    buriedCards: [],
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

  if (state.phase === "exchanging" || state.phase === "choosing-adjutant") {
    return [];
  }

  if (state.phase !== "playing" || state.isTrickComplete) {
    return [];
  }

  const player = getPlayer(state, playerId);
  return getPlayableCards(player.hand, state.currentTrick, { trumpSuit: state.trumpSuit }).map((card) => ({
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
    case "discard-cards":
      return discardCards(state, action.playerId, action.cardIds);
    case "choose-adjutant":
      return chooseAdjutant(state, action.playerId, action.cardId);
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
    adjutant: toAdjutantView(state.adjutant ?? null),
    bidding: state.bidding,
    exchangeRequirement:
      state.phase === "exchanging" &&
      state.contract !== null &&
      state.contract.napoleonPlayerId === playerId
        ? { discardCount: 3 }
        : null,
    adjutantChoiceRequirement:
      state.phase === "choosing-adjutant" &&
      state.contract !== null &&
      state.contract.napoleonPlayerId === playerId
        ? { standardCardsOnly: true }
        : null,
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

  if (state.trumpSuit === null) {
    throw new GameRuleError("TRUMP_NOT_SET", "Trump suit must be set before playing cards.");
  }

  if (state.currentPlayerId !== playerId) {
    throw new GameRuleError("NOT_PLAYERS_TURN", "It is not this player's turn.");
  }

  const player = getPlayer(state, playerId);
  const card = player.hand.find((candidate) => candidate.id === cardId);

  if (card === undefined) {
    throw new GameRuleError("CARD_NOT_IN_HAND", "The card is not in this player's hand.");
  }

  const playableCards = getPlayableCards(player.hand, state.currentTrick, {
    trumpSuit: state.trumpSuit
  });

  if (!playableCards.some((candidate) => candidate.id === cardId)) {
    throw new GameRuleError("MUST_FOLLOW_SUIT", "The player must follow the lead suit.");
  }

  const players = state.players.map((candidate) =>
    candidate.id === playerId
      ? { ...candidate, hand: removeCard(candidate.hand, cardId) }
      : candidate
  );

  const currentTrick = [...state.currentTrick, { playerId, card }];
  const adjutant = revealAdjutantIfNeeded(state.adjutant ?? null, playerId, card.id);
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
    adjutant,
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
  const players = state.players.map((player) =>
    player.id === contract.napoleonPlayerId
      ? {
          ...player,
          hand: [...player.hand, ...state.unusedCards]
        }
      : player
  );

  return {
    ...state,
    players,
    phase: "exchanging",
    currentPlayerId: contract.napoleonPlayerId,
    trumpSuit: contract.trumpSuit,
    contract,
    bidding: null,
    unusedCards: [],
    buriedCards: [],
    adjutant: null
  };
}

function discardCards(
  state: GameState,
  playerId: PlayerId,
  cardIds: readonly string[]
): GameState {
  if (state.phase === "finished" || state.isGameOver) {
    throw new GameRuleError("GAME_OVER", "The game is already over.");
  }

  if (state.phase !== "exchanging") {
    throw new GameRuleError(
      "INVALID_ACTION_FOR_PHASE",
      "This action is not allowed in the current game phase."
    );
  }

  if (state.contract === null) {
    throw new GameRuleError("INVALID_EXCHANGE_STATE", "A contract is required for exchange.");
  }

  if (playerId !== state.contract.napoleonPlayerId) {
    throw new GameRuleError("NOT_NAPOLEON", "Only Napoleon can discard buried cards.");
  }

  if (state.currentPlayerId !== playerId) {
    throw new GameRuleError("NOT_PLAYERS_TURN", "It is not this player's turn.");
  }

  if (state.buriedCards.length !== 0) {
    throw new GameRuleError("EXCHANGE_ALREADY_COMPLETED", "Buried card exchange is already complete.");
  }

  if (cardIds.length !== 3) {
    throw new GameRuleError("INVALID_DISCARD_COUNT", "Exactly 3 cards must be discarded.");
  }

  if (new Set(cardIds).size !== cardIds.length) {
    throw new GameRuleError("DUPLICATE_CARD_ID", "Discarded card ids must be distinct.");
  }

  const player = getPlayer(state, playerId);

  if (player.hand.length !== 13) {
    throw new GameRuleError("INVALID_EXCHANGE_STATE", "Napoleon must have 13 cards before exchange.");
  }

  const discardedCards = cardIds.map((cardId) => {
    const card = player.hand.find((candidate) => candidate.id === cardId);

    if (card === undefined) {
      throw new GameRuleError("CARD_NOT_IN_HAND", "The card is not in this player's hand.");
    }

    return card;
  });
  const discardedIdSet = new Set(cardIds);
  const players = state.players.map((candidate) =>
    candidate.id === playerId
      ? {
          ...candidate,
          hand: candidate.hand.filter((card) => !discardedIdSet.has(card.id))
        }
      : candidate
  );

  return {
    ...state,
    players,
    buriedCards: discardedCards,
    phase: "choosing-adjutant",
    adjutant: null,
    currentPlayerId: playerId,
    currentTrick: [],
    trickNumber: 1,
    isTrickComplete: false,
    isGameOver: false
  };
}

function chooseAdjutant(state: GameState, playerId: PlayerId, cardId: string): GameState {
  if (state.phase === "finished" || state.isGameOver) {
    throw new GameRuleError("GAME_OVER", "The game is already over.");
  }

  if (state.phase !== "choosing-adjutant") {
    throw new GameRuleError(
      "INVALID_ACTION_FOR_PHASE",
      "This action is not allowed in the current game phase."
    );
  }

  if (state.contract === null) {
    throw new GameRuleError("INVALID_ADJUTANT_STATE", "A contract is required to choose an adjutant.");
  }

  if (playerId !== state.contract.napoleonPlayerId) {
    throw new GameRuleError("NOT_NAPOLEON", "Only Napoleon can choose the adjutant card.");
  }

  if (state.currentPlayerId !== playerId) {
    throw new GameRuleError("NOT_PLAYERS_TURN", "It is not this player's turn.");
  }

  if (state.adjutant !== null) {
    throw new GameRuleError("ADJUTANT_ALREADY_CHOSEN", "The adjutant card has already been chosen.");
  }

  if (!isStandardCardId(cardId)) {
    throw new GameRuleError("INVALID_ADJUTANT_CARD", "The adjutant card must be a standard card.");
  }

  if (
    state.currentTrick.length !== 0 ||
    state.completedTricks.length !== 0 ||
    state.trickNumber !== 1 ||
    state.buriedCards.length !== 3
  ) {
    throw new GameRuleError("INVALID_ADJUTANT_STATE", "Adjutant must be chosen before play starts.");
  }

  const adjutant: AdjutantState = {
    calledCardId: cardId,
    playerId: resolveAdjutantPlayerId(state, cardId),
    revealed: false
  };

  return {
    ...state,
    phase: "playing",
    currentPlayerId: playerId,
    adjutant
  };
}

function resolveAdjutantPlayerId(state: GameState, cardId: string): PlayerId | null {
  if (state.contract === null) {
    throw new GameRuleError("INVALID_ADJUTANT_STATE", "A contract is required to choose an adjutant.");
  }

  const owner = state.players.find((player) => player.hand.some((card) => card.id === cardId));

  if (owner !== undefined) {
    return owner.id === state.contract.napoleonPlayerId ? null : owner.id;
  }

  if (state.buriedCards.some((card) => card.id === cardId)) {
    return null;
  }

  throw new GameRuleError("INVALID_ADJUTANT_STATE", "The adjutant card is not in the game state.");
}

function revealAdjutantIfNeeded(
  adjutant: AdjutantState | null,
  playerId: PlayerId,
  cardId: string
): AdjutantState | null {
  if (adjutant === null || adjutant.revealed || adjutant.calledCardId !== cardId) {
    return adjutant;
  }

  if (adjutant.playerId === null) {
    return adjutant;
  }

  if (adjutant.playerId !== playerId) {
    throw new GameRuleError("INVALID_ADJUTANT_STATE", "The called adjutant card owner is inconsistent.");
  }

  return {
    ...adjutant,
    revealed: true
  };
}

function toAdjutantView(adjutant: AdjutantState | null): PlayerView["adjutant"] {
  if (adjutant === null) {
    return null;
  }

  return {
    calledCardId: adjutant.calledCardId,
    revealedPlayerId: adjutant.revealed ? adjutant.playerId : null
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
