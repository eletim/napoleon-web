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

export interface StandardCard {
  type: "standard";
  id: string;
  suit: Suit;
  rank: Rank;
}

export interface JokerCard {
  type: "joker";
  id: "joker";
}

export type Card = StandardCard | JokerCard;

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

export type GamePhase = "bidding" | "exchanging" | "playing" | "finished";

export interface Bid {
  playerId: PlayerId;
  suit: Suit;
  targetPointCards: number;
}

export interface Contract {
  napoleonPlayerId: PlayerId;
  trumpSuit: Suit;
  targetPointCards: number;
}

export type BiddingHistoryEntry =
  | {
      type: "bid";
      playerId: PlayerId;
      suit: Suit;
      targetPointCards: number;
    }
  | {
      type: "pass";
      playerId: PlayerId;
    };

export interface BiddingState {
  starterPlayerId: PlayerId;
  highestBid: Bid | null;
  consecutivePassCount: number;
  history: readonly BiddingHistoryEntry[];
}

export type PublicBiddingView = BiddingState;

export interface GameState {
  players: readonly PlayerState[];
  phase: GamePhase;
  currentPlayerId: PlayerId;
  currentTrick: readonly PlayedCard[];
  completedTricks: readonly CompletedTrick[];
  trumpSuit: Suit | null;
  contract: Contract | null;
  bidding: BiddingState | null;
  buriedCards: readonly Card[];
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

export interface BidAction {
  type: "bid";
  playerId: PlayerId;
  suit: Suit;
  targetPointCards: number;
}

export interface PassAction {
  type: "pass";
  playerId: PlayerId;
}

export interface DiscardCardsAction {
  type: "discard-cards";
  playerId: PlayerId;
  cardIds: readonly string[];
}

export type GameAction = PlayCardAction | BidAction | PassAction | DiscardCardsAction;

export interface ExchangeRequirement {
  discardCount: number;
}

export interface PublicPlayerState {
  id: PlayerId;
  handCount: number;
  hand?: readonly Card[];
}

export interface PlayerView {
  selfId: PlayerId;
  players: readonly PublicPlayerState[];
  phase: GamePhase;
  trumpSuit: Suit | null;
  contract: Contract | null;
  bidding: PublicBiddingView | null;
  exchangeRequirement: ExchangeRequirement | null;
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
