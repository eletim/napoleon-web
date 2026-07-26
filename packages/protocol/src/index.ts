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

export interface PublicStandardCard {
  type: "standard";
  id: string;
  suit: PublicSuit;
  rank: PublicRank;
}

export interface PublicJokerCard {
  type: "joker";
  id: "joker";
}

export type PublicCard = PublicStandardCard | PublicJokerCard;

export interface PublicPlayedCard {
  playerId: string;
  card: PublicCard;
}

export type PublicGamePhase =
  | "bidding"
  | "exchanging"
  | "choosing-adjutant"
  | "playing"
  | "finished";

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

export interface PublicExchangeState {
  napoleonPlayerId: string;
  requiredDiscardCount: number;
}

export interface PublicAdjutantState {
  calledCardId: string;
  revealedPlayerId: string | null;
}

export interface PublicAdjutantChoiceState {
  napoleonPlayerId: string;
  standardCardsOnly: true;
}

export interface PublicBuriedCardsState {
  revealedPointCards: readonly PublicStandardCard[];
  hiddenCardCount: number;
}

export type PublicWinningTeam = "napoleon-team" | "alliance";

export interface PublicGameResult {
  winner: PublicWinningTeam;
  napoleonTeamPointCards: number;
  alliancePointCards: number;
  buriedPointCards: number;
  targetPointCards: number;
  napoleonPlayerId: string;
  adjutantPlayerId: string | null;
}

export interface PublicSpecialCardsState {
  orumaCardId: string;
  yoromekiCardId: string;
  seiJackCardId: string | null;
  uraJackCardId: string | null;
}

export interface PublicSelfPlayer {
  id: string;
  handCount: number;
  hand: readonly PublicCard[];
  capturedPointCards: readonly PublicStandardCard[];
}

export interface PublicOpponentPlayer {
  id: string;
  handCount: number;
  capturedPointCards: readonly PublicStandardCard[];
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

export interface PublicDiscardCardsAction {
  type: "discard-cards";
  cardIds: readonly string[];
}

export interface PublicChooseAdjutantAction {
  type: "choose-adjutant";
  cardId: string;
}

export type PublicLegalAction = PublicPlayCardAction | PublicBidAction | PublicPassAction;

export interface PublicGameState {
  self: PublicSelfPlayer;
  opponents: readonly PublicOpponentPlayer[];
  phase: PublicGamePhase;
  trumpSuit: PublicSuit | null;
  contract: PublicContract | null;
  specialCards: PublicSpecialCardsState;
  adjutant: PublicAdjutantState | null;
  buriedCards: PublicBuriedCardsState | null;
  result: PublicGameResult | null;
  bidding: PublicBiddingState | null;
  exchange: PublicExchangeState | null;
  adjutantChoice: PublicAdjutantChoiceState | null;
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

export interface DiscardCardsRequest {
  type: "discard-cards";
  cardIds: readonly string[];
}

export interface ChooseAdjutantRequest {
  type: "choose-adjutant";
  cardId: string;
}

export type PublicGameAction =
  | PlayCardRequest
  | BidRequest
  | PassRequest
  | DiscardCardsRequest
  | ChooseAdjutantRequest;

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
