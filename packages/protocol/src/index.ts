export type PublicSuit = "spades" | "hearts" | "diamonds" | "clubs";

export type PublicRank =
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

export interface PublicCard {
  id: string;
  suit: PublicSuit;
  rank: PublicRank;
}

export interface PublicPlayedCard {
  playerId: string;
  card: PublicCard;
}

export interface PublicSelfPlayer {
  id: string;
  handCount: number;
  hand: readonly PublicCard[];
}

export interface PublicOpponentPlayer {
  id: string;
  handCount: number;
}

export type PublicLegalAction = {
  type: "play-card";
  cardId: string;
};

export interface PublicGameState {
  self: PublicSelfPlayer;
  opponents: readonly PublicOpponentPlayer[];
  currentPlayerId: string;
  currentTrick: readonly PublicPlayedCard[];
  completedTrickCount: number;
  trickNumber: number;
  isTrickComplete: boolean;
  isGameOver: boolean;
  legalActions: readonly PublicLegalAction[];
}

export type CreateGameRequest = Record<string, never>;

export interface CreateGameResponse {
  gameId: string;
  playerId: string;
  state: PublicGameState;
}

export interface PlayCardRequest {
  type: "play-card";
  cardId: string;
}

export interface SendActionRequest {
  action: PlayCardRequest;
}

export interface SendActionResponse {
  gameId: string;
  playerId: string;
  state: PublicGameState;
}

export interface GetGameResponse {
  gameId: string;
  playerId: string;
  state: PublicGameState;
}

export interface NextTrickResponse {
  gameId: string;
  playerId: string;
  state: PublicGameState;
}

export interface ApiError {
  error: {
    code: string;
    message: string;
  };
}
