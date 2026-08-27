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
  jokerAllowed: true;
}

export type PublicWinningTeam = "napoleon-team" | "alliance";

export interface PublicPlayerPayoff {
  playerId: string;
  payoff: number;
}

export interface PublicStandardGameResult {
  resultType: "standard";
  winner: PublicWinningTeam;
  napoleonTeamPointCards: number;
  alliancePointCards: number;
  targetPointCards: number;
  napoleonPlayerId: string;
  adjutantPlayerId: string | null;
}

export interface PublicAllPassGameResult {
  resultType: "all-pass";
  starterPlayerId: string;
  payoffs: readonly PublicPlayerPayoff[];
}

export type PublicGameResult = PublicStandardGameResult | PublicAllPassGameResult;

export interface PublicBuriedCardsResolvedEvent {
  type: "buried-cards-resolved";
  napoleonPlayerId: string;
  awardedPointCards: readonly PublicStandardCard[];
  hiddenNonPointCardCount: number;
}

export type PublicGameEvent = PublicBuriedCardsResolvedEvent;

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
  latestEvent: PublicGameEvent | null;
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

export interface PublicAgentDescriptor {
  id: string;
  displayName: string;
  isAvailable: boolean;
}

export type PlayingPolicyId = "rule-based" | "ppo-separated-v1000";
export type BiddingPolicyId = "rule-based" | "frozen-raise-v1";
export type NonPlayingPolicyId =
  | "rule-based"
  | "parameterized-adjutant-exchange-v1";

export interface AiPolicyComposition {
  playing: PlayingPolicyId;
  bidding: BiddingPolicyId;
  nonPlaying: NonPlayingPolicyId;
}

export type AiPresetId = "com-rule-base" | "com-ai";

export interface AiPreset {
  id: AiPresetId;
  displayName: string;
  composition: AiPolicyComposition;
}

export interface PublicPhasePolicyDescriptor {
  id: string;
  displayName: string;
  isAvailable: boolean;
  artifactProvenance: Readonly<Record<string, string>> | null;
}

export interface PublicPhasePolicyRegistry {
  playing: readonly PublicPhasePolicyDescriptor[];
  bidding: readonly PublicPhasePolicyDescriptor[];
  nonPlaying: readonly PublicPhasePolicyDescriptor[];
}

export interface GetAgentsResponse {
  agents: readonly PublicAgentDescriptor[];
  policyRegistry?: PublicPhasePolicyRegistry;
}

export interface GetAiPresetsResponse {
  presets: readonly AiPreset[];
  policyRegistry: PublicPhasePolicyRegistry;
}

export interface UpdateAiPresetRequest {
  composition: AiPolicyComposition;
}

export interface CreateGameLegacyAgentSelection {
  playerId: string;
  agentId: string;
}

export interface CreateGamePolicyCompositionSelection {
  playerId: string;
  policyComposition: AiPolicyComposition;
}

export type CreateGameAgentSelection =
  | CreateGameLegacyAgentSelection
  | CreateGamePolicyCompositionSelection;

export interface CreateGameRequest {
  aiAgents?: readonly CreateGameAgentSelection[];
  aiPresetId?: AiPresetId;
}

export interface RunAutomatedSimulationRequest {
  seed: number;
  policyComposition?: AiPolicyComposition;
  aiPresetId?: AiPresetId;
}

export interface PublicAiPhaseCallDiagnostics {
  composition: AiPolicyComposition;
  playingCalls: number;
  biddingCalls: number;
  adjutantCalls: number;
  exchangeCalls: number;
  fallbackCount: number;
  illegalCount: number;
}

export interface GetGamePolicyDiagnosticsResponse {
  gameId: string;
  presetId?: AiPresetId;
  diagnostics: Readonly<Record<string, PublicAiPhaseCallDiagnostics>>;
}

export interface PublicSimulationObservedPlayer {
  id: string;
  handCount: number;
  hand?: readonly PublicCard[];
  capturedPointCards: readonly PublicStandardCard[];
}

export interface PublicSimulationCompletedTrick {
  trickNumber: number;
  winnerId: string;
  cards: readonly PublicPlayedCard[];
}

export interface PublicSimulationObservation {
  selfId: string;
  players: readonly PublicSimulationObservedPlayer[];
  phase: PublicGamePhase;
  trumpSuit: PublicSuit | null;
  contract: PublicContract | null;
  specialCards: PublicSpecialCardsState;
  adjutant: PublicAdjutantState | null;
  latestEvent: PublicGameEvent | null;
  bidding: PublicBiddingState | null;
  exchangeRequirement: {
    discardCount: number;
  } | null;
  adjutantChoiceRequirement: {
    jokerAllowed: true;
  } | null;
  currentPlayerId: string;
  currentTrick: readonly PublicPlayedCard[];
  completedTricks: readonly PublicSimulationCompletedTrick[];
  completedTrickCount: number;
  trickNumber: number;
  isTrickComplete: boolean;
  isGameOver: boolean;
}

export interface PublicSimulationActualState {
  hands: Readonly<Record<string, readonly string[]>>;
  unusedCardIds: readonly string[];
  excludedCardIds: readonly string[];
  awardedPointCardIds: Readonly<Record<string, readonly string[]>>;
  currentTrickCardIds: readonly string[];
  completedTrickCardIds: readonly string[];
}

export interface PublicSimulationDecision {
  step: number;
  playerId: string;
  phase: PublicGamePhase;
  trickNumber: number;
  observation: PublicSimulationObservation;
  legalActions: readonly PublicGameAction[];
  action: PublicGameAction;
  legalActionCount: number;
  handCounts: Readonly<Record<string, number>>;
  actualHands: Readonly<Record<string, readonly string[]>>;
  actualState: PublicSimulationActualState;
}

export interface PublicSimulationSummary {
  totalDecisionCount: number;
  decisionCountByPlayer: Readonly<Record<string, number>>;
  decisionCountByPhase: Readonly<Record<PublicGamePhase, number>>;
  actionCountByType: Readonly<Record<string, number>>;
}

export interface RunAutomatedSimulationResponse {
  schemaVersion: 1;
  seed: number;
  playerIds: readonly string[];
  initialHands: Readonly<Record<string, readonly string[]>>;
  initialActualState: PublicSimulationActualState;
  decisions: readonly PublicSimulationDecision[];
  summary: PublicSimulationSummary;
  result: PublicGameResult;
  presetId?: AiPresetId;
  policyDiagnostics?: Readonly<Record<string, PublicAiPhaseCallDiagnostics>>;
}

export interface ApiError {
  error: {
    code: string;
    message: string;
  };
}
