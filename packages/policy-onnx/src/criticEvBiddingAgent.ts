import {
  createSpecialCardsForTrump,
  RuleBasedAgent
} from "@napoleon/ai";
import type { Agent, PlayerObservation, PublicActionRecord } from "@napoleon/ai";
import {
  createDeck,
  isPointCard,
  isStandardCard,
  orumaCardId
} from "@napoleon/game-core";
import type {
  Bid,
  BidAction,
  Card,
  GameAction,
  PlayerId,
  PlayerView,
  PlayingSelfRole,
  PublicPlayerState,
  Rank,
  Suit
} from "@napoleon/game-core";
import {
  createPlayingModelInput,
  createRelativePlayerOrder,
  encodeBiddingHistoryFromPublicActions,
  encodePlayingObservation
} from "@napoleon/ai-observation";
import { criticValueToWinRateEquivalent } from "./policyOnnx.js";

export interface PolicyCriticValueModel {
  predictValuesBatch(modelInputs: readonly (Float32Array | readonly number[])[]): Promise<readonly number[]>;
}

export interface CriticEvBiddingAgentOptions {
  critic: PolicyCriticValueModel;
  delegateAgent?: Agent;
  playerIds?: readonly PlayerId[];
}

export interface CriticEvBiddingEvaluation {
  action: Extract<GameAction, { type: "bid" | "pass" }>;
  criticValue: number;
  baseWinRateEquivalent: number;
  effectiveNapoleonWinRate: number;
  expectedValue: number;
  role: PlayingSelfRole;
  contract: Bid;
  calledAdjutantCardId: string;
}

const trumpRankPriority: readonly Rank[] = ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"];

/**
 * Simple Issue #193 EV bidding baseline.
 *
 * This intentionally does not sample hidden hands, search adjutants, or optimize exchange.
 * It evaluates one virtual start-of-play observation per legal bid/pass, assumes the buried
 * cards are three hidden non-point cards, and applies the fixed Napoleon bonus/reward rules.
 */
export class CriticEvBiddingAgent implements Agent {
  private readonly critic: PolicyCriticValueModel;
  private readonly delegateAgent: Agent;
  private readonly playerIds: readonly PlayerId[] | null;

  constructor(options: CriticEvBiddingAgentOptions) {
    this.critic = options.critic;
    this.delegateAgent = options.delegateAgent ?? new RuleBasedAgent();
    this.playerIds = options.playerIds ?? null;
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    if (observation.view.phase !== "bidding") {
      return this.delegateAgent.selectAction(observation);
    }

    const evaluations = await this.evaluateLegalBiddingActions(observation);
    if (evaluations.length === 0) {
      return this.delegateAgent.selectAction(observation);
    }

    return evaluations
      .map((evaluation, index) => ({ evaluation, index }))
      .sort((left, right) =>
        right.evaluation.expectedValue - left.evaluation.expectedValue ||
        left.index - right.index
      )[0].evaluation.action;
  }

  async evaluateLegalBiddingActions(
    observation: PlayerObservation
  ): Promise<readonly CriticEvBiddingEvaluation[]> {
    if (observation.view.phase !== "bidding") {
      throw new Error(`CriticEvBiddingAgent can evaluate only bidding observations, got ${observation.view.phase}.`);
    }

    const candidates = observation.legalActions.filter(isBiddingDecisionAction);
    const modelInputs = candidates.map((action) =>
      createCriticPlayingInputForBiddingAction(observation, action, this.resolvePlayerIds(observation))
    );
    const criticValues = await this.critic.predictValuesBatch(modelInputs.map((input) => input.modelInput));

    return modelInputs.map((input, index) =>
      evaluateBiddingActionFromCriticValue(input, candidates[index], criticValues[index])
    );
  }

  private resolvePlayerIds(observation: PlayerObservation): readonly PlayerId[] {
    return this.playerIds ?? observation.view.players.map((player) => player.id);
  }
}

interface CriticBiddingInput {
  modelInput: Float32Array;
  role: PlayingSelfRole;
  contract: Bid;
  calledAdjutantCardId: string;
}

function createCriticPlayingInputForBiddingAction(
  observation: PlayerObservation,
  action: Extract<GameAction, { type: "bid" | "pass" }>,
  playerIds: readonly PlayerId[]
): CriticBiddingInput {
  const contract = contractForBiddingAction(observation, action);
  const selfHand = getSelfHand(observation);
  const calledAdjutantCardId =
    action.type === "bid"
      ? selectSimpleAdjutantCardId(selfHand, action.suit)
      : orumaCardId;
  const role = roleForVirtualContract(observation.playerId, contract, calledAdjutantCardId, selfHand);
  const view = createVirtualPlayingView({
    sourceView: observation.view,
    playerId: observation.playerId,
    contract,
    calledAdjutantCardId,
    role
  });
  const relativePlayerIds = createRelativePlayerOrder(playerIds, observation.playerId);
  const biddingHistory = encodeBiddingHistoryFromPublicActions(
    appendBiddingActionHistory(observation, action),
    relativePlayerIds
  );
  const encoded = encodePlayingObservation(
    {
      playerId: observation.playerId,
      view,
      legalActions: view.legalActions,
      publicActionHistory: appendBiddingActionHistory(observation, action)
    },
    playerIds,
    biddingHistory
  );

  return {
    modelInput: createPlayingModelInput(encoded).modelInput,
    role,
    contract,
    calledAdjutantCardId
  };
}

function evaluateBiddingActionFromCriticValue(
  input: CriticBiddingInput,
  action: Extract<GameAction, { type: "bid" | "pass" }>,
  criticValue: number
): CriticEvBiddingEvaluation {
  const baseWinRateEquivalent = criticValueToWinRateEquivalent(criticValue);
  const target = input.contract.targetPointCards;
  let effectiveNapoleonWinRate = baseWinRateEquivalent;
  let expectedValue: number;

  if (action.type === "bid") {
    effectiveNapoleonWinRate = (1 + baseWinRateEquivalent) / 2;
    expectedValue = effectiveNapoleonWinRate * target + (1 - effectiveNapoleonWinRate) * -3;
  } else if (input.role === "adjutant") {
    expectedValue = baseWinRateEquivalent * (target - 7);
  } else if (input.role === "alliance") {
    effectiveNapoleonWinRate = 1 - baseWinRateEquivalent;
    expectedValue = (1 - effectiveNapoleonWinRate) * 7;
  } else {
    expectedValue = baseWinRateEquivalent * target + (1 - baseWinRateEquivalent) * -3;
  }

  return {
    action,
    criticValue,
    baseWinRateEquivalent,
    effectiveNapoleonWinRate,
    expectedValue,
    role: input.role,
    contract: input.contract,
    calledAdjutantCardId: input.calledAdjutantCardId
  };
}

function contractForBiddingAction(
  observation: PlayerObservation,
  action: Extract<GameAction, { type: "bid" | "pass" }>
): Bid {
  if (action.type === "bid") {
    return {
      playerId: action.playerId,
      suit: action.suit,
      targetPointCards: action.targetPointCards
    };
  }

  const bidding = observation.view.bidding;
  if (bidding?.highestBid !== null && bidding?.highestBid !== undefined) {
    return bidding.highestBid;
  }

  return {
    playerId: bidding?.starterPlayerId ?? observation.view.players[0].id,
    suit: "spades",
    targetPointCards: 12
  };
}

function createVirtualPlayingView({
  sourceView,
  playerId,
  contract,
  calledAdjutantCardId,
  role
}: {
  sourceView: PlayerView;
  playerId: PlayerId;
  contract: Bid;
  calledAdjutantCardId: string;
  role: PlayingSelfRole;
}): PlayerView {
  const specialCards = createSpecialCardsForTrump(contract.suit);
  const selfHand = getSelfHandFromView(sourceView, playerId);
  const players = sourceView.players.map((player): PublicPlayerState => ({
    id: player.id,
    handCount: 10,
    capturedPointCards: [],
    ...(player.id === playerId ? { hand: selfHand } : {})
  }));

  return {
    ...sourceView,
    players,
    phase: "playing",
    playingSelfRole: role,
    trumpSuit: contract.suit,
    contract: {
      napoleonPlayerId: contract.playerId,
      trumpSuit: contract.suit,
      targetPointCards: contract.targetPointCards
    },
    specialCards,
    adjutant: {
      calledCardId: calledAdjutantCardId,
      revealedPlayerId: null
    },
    bidding: null,
    exchangeRequirement: null,
    adjutantChoiceRequirement: null,
    currentPlayerId: playerId,
    currentTrick: [],
    completedTricks: [],
    completedTrickCount: 0,
    trickNumber: 1,
    isTrickComplete: false,
    isGameOver: false,
    result: null,
    latestEvent: {
      type: "buried-cards-resolved",
      napoleonPlayerId: contract.playerId,
      awardedPointCards: [],
      hiddenNonPointCardCount: 3
    },
    legalActions: selfHand.map((card) => ({
      type: "play-card" as const,
      playerId,
      cardId: card.id
    }))
  };
}

function selectSimpleAdjutantCardId(selfHand: readonly Card[], trumpSuit: Suit): string {
  const selfCardIds = new Set(selfHand.map((card) => card.id));
  const specialCards = createSpecialCardsForTrump(trumpSuit);
  const specialPriority = [
    specialCards.orumaCardId,
    specialCards.seiJackCardId,
    specialCards.uraJackCardId
  ].filter((cardId): cardId is string => cardId !== null);
  const trumpPriority = trumpRankPriority.map((rank) => `${trumpSuit}-${rank}`);

  for (const cardId of [...specialPriority, ...trumpPriority]) {
    if (!selfCardIds.has(cardId)) {
      return cardId;
    }
  }

  const fallback = createDeck()
    .filter(isStandardCard)
    .find((card) => !selfCardIds.has(card.id));
  return fallback?.id ?? "joker";
}

function roleForVirtualContract(
  selfId: PlayerId,
  contract: Bid,
  calledAdjutantCardId: string,
  selfHand: readonly Card[]
): PlayingSelfRole {
  if (contract.playerId === selfId) {
    return selfHand.some((card) => card.id === calledAdjutantCardId)
      ? "napoleon-solo"
      : "napoleon";
  }

  return selfHand.some((card) => card.id === calledAdjutantCardId)
    ? "adjutant"
    : "alliance";
}

function appendBiddingActionHistory(
  observation: PlayerObservation,
  action: Extract<GameAction, { type: "bid" | "pass" }>
): readonly PublicActionRecord[] {
  const history = observation.publicActionHistory ?? biddingViewHistoryToPublicActions(observation.view);
  const nextStep = history.length === 0 ? 1 : Math.max(...history.map((record) => record.step)) + 1;

  return [
    ...history,
    {
      step: nextStep,
      playerId: action.playerId,
      phase: "bidding" as const,
      action
    }
  ];
}

function biddingViewHistoryToPublicActions(view: PlayerView): readonly PublicActionRecord[] {
  return view.bidding?.history.map((action, index) => ({
    step: index + 1,
    playerId: action.playerId,
    phase: "bidding" as const,
    action
  })) ?? [];
}

function getSelfHand(observation: PlayerObservation): readonly Card[] {
  return getSelfHandFromView(observation.view, observation.playerId);
}

function getSelfHandFromView(view: PlayerView, playerId: PlayerId): readonly Card[] {
  const self = view.players.find((player) => player.id === playerId);
  if (self?.hand === undefined) {
    throw new Error(`Self hand is missing from observation for ${playerId}.`);
  }

  return self.hand;
}

function isBiddingDecisionAction(action: GameAction): action is BidAction | Extract<GameAction, { type: "pass" }> {
  return action.type === "bid" || action.type === "pass";
}

export function isNonPointCard(card: Card): boolean {
  return !isPointCard(card);
}
