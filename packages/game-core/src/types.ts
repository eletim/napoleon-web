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

export type GamePhase =
  | "bidding"
  | "exchanging"
  | "choosing-adjutant"
  | "playing"
  | "finished";

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

export interface AdjutantState {
  calledCardId: string;
  playerId: PlayerId | null;
  revealed: boolean;
}

export interface AdjutantView {
  calledCardId: string;
  revealedPlayerId: PlayerId | null;
}

export type WinningTeam = "napoleon-team" | "alliance";

export interface GameResult {
  winner: WinningTeam;
  napoleonTeamPointCards: number;
  alliancePointCards: number;
  targetPointCards: number;
  napoleonPlayerId: PlayerId;
  adjutantPlayerId: PlayerId | null;
}

export interface AwardedPointCards {
  playerId: PlayerId;
  cards: readonly StandardCard[];
}

export interface BuriedCardsResolvedEvent {
  type: "buried-cards-resolved";
  napoleonPlayerId: PlayerId;
  awardedPointCards: readonly StandardCard[];
  hiddenNonPointCardCount: number;
}

export type GameEvent = BuriedCardsResolvedEvent;

export interface SpecialCardsView {
  orumaCardId: string;
  yoromekiCardId: string;
  seiJackCardId: string | null;
  uraJackCardId: string | null;
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
  adjutant: AdjutantState | null;
  bidding: BiddingState | null;
  awardedPointCards: readonly AwardedPointCards[];
  excludedCards: readonly Card[];
  latestEvent: GameEvent | null;
  result: GameResult | null;
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

export interface ChooseAdjutantAction {
  type: "choose-adjutant";
  playerId: PlayerId;
  cardId: string;
}

export type GameAction =
  | PlayCardAction
  | BidAction
  | PassAction
  | DiscardCardsAction
  | ChooseAdjutantAction;

export interface ExchangeRequirement {
  discardCount: number;
}

export interface AdjutantChoiceRequirement {
  standardCardsOnly: true;
}

export interface PublicPlayerState {
  id: PlayerId;
  handCount: number;
  capturedPointCards: readonly StandardCard[];
  hand?: readonly Card[];
}

export interface PlayerView {
  selfId: PlayerId;
  players: readonly PublicPlayerState[];
  phase: GamePhase;
  trumpSuit: Suit | null;
  contract: Contract | null;
  specialCards: SpecialCardsView;
  adjutant: AdjutantView | null;
  latestEvent: GameEvent | null;
  result: GameResult | null;
  bidding: PublicBiddingView | null;
  exchangeRequirement: ExchangeRequirement | null;
  adjutantChoiceRequirement: AdjutantChoiceRequirement | null;
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
