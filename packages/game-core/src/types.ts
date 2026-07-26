export type Suit = "spades" | "hearts" | "diamonds" | "clubs";

export type Rank =
  | "A"
  | "2"
  | "3"
  | "4"
  | "5"
  | "6"
  | "7"
  | "8"
  | "9"
  | "10"
  | "J"
  | "Q"
  | "K";

export interface Card {
  id: string;
  suit: Suit;
  rank: Rank;
}

export type PlayerId = string;

export interface PlayerState {
  id: PlayerId;
  hand: readonly Card[];
}

export interface PlayedCard {
  playerId: PlayerId;
  card: Card;
}

export interface CompletedTrick {
  trickNumber: number;
  winnerId: PlayerId;
  cards: readonly PlayedCard[];
}

export interface GameState {
  players: readonly PlayerState[];
  currentPlayerId: PlayerId;
  currentTrick: readonly PlayedCard[];
  completedTricks: readonly CompletedTrick[];
  trickNumber: number;
  isTrickComplete: boolean;
  isGameOver: boolean;
  unusedCards: readonly Card[];
}

export interface PlayCardAction {
  type: "play-card";
  playerId: PlayerId;
  cardId: string;
}

export type GameAction = PlayCardAction;

export interface PublicPlayerState {
  id: PlayerId;
  handCount: number;
  hand?: readonly Card[];
}

export interface PlayerView {
  selfId: PlayerId;
  players: readonly PublicPlayerState[];
  currentPlayerId: PlayerId;
  currentTrick: readonly PlayedCard[];
  completedTrickCount: number;
  trickNumber: number;
  isTrickComplete: boolean;
  isGameOver: boolean;
  legalActions: readonly GameAction[];
}

export interface CreateInitialGameOptions {
  playerIds?: readonly PlayerId[];
  rng?: () => number;
}
