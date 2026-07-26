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

export type PublicGamePhase = "bidding" | "playing" | "finished";

export interface PublicBid {
  playerId: string;
  suit: PublicSuit;
  targetPointCards: number;
}

export interface PublicContract {
  napoleonPlayerId: string;
  trumpSuit: PublicSuit;
  targetPointCards: number;
}

export type PublicBiddingHistoryEntry =
  | {
      type: "bid";
      playerId: string;
      suit: PublicSuit;
      targetPointCards: number;
    }
  | {
      type: "pass";
      playerId: string;
    };

export interface PublicBiddingState {
  starterPlayerId: string;
  highestBid: PublicBid | null;
  consecutivePassCount: number;
  history: readonly PublicBiddingHistoryEntry[];
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

export interface PublicPlayCardAction {
  type: "play-card";
  cardId: string;
}

export interface PublicBidAction {
  type: "bid";
  suit: PublicSuit;
  targetPointCards: number;
}

export interface PublicPassAction {
  type: "pass";
}

export type PublicLegalAction = PublicPlayCardAction | PublicBidAction | PublicPassAction;

export interface PublicGameState {
  self: PublicSelfPlayer;
  opponents: readonly PublicOpponentPlayer[];
  phase: PublicGamePhase;
  trumpSuit: PublicSuit | null;
  contract: PublicContract | null;
  bidding: PublicBiddingState | null;
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

export interface BidRequest {
  type: "bid";
  suit: PublicSuit;
  targetPointCards: number;
}

export interface PassRequest {
  type: "pass";
}

export type PublicGameAction = PlayCardRequest | BidRequest | PassRequest;

export interface SendActionRequest {
  action: PublicGameAction;
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
