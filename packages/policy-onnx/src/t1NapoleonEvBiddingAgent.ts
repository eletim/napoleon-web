import {
  RuleBasedAgent,
  evaluateHandForTrump
} from "@napoleon/ai";
import type { Agent, PlayerObservation } from "@napoleon/ai";
import { decodeBiddingAction } from "@napoleon/ai-observation";
import type { BidAction, GameAction, PlayerId, Suit } from "@napoleon/game-core";
import { createPolicyOnnxBiddingInput } from "./policyOnnxAgent.js";
import type { BiddingMarginOnnxModel } from "./policyOnnx.js";
import type { CriticEvBiddingAgent } from "./criticEvBiddingAgent.js";
import { PolicyOnnxCompatibilityError } from "./errors.js";

export type T1BiddingDecisionKind = "opening" | "raise";
export type T1HandStrengthBucket = "low" | "medium" | "strong" | "veryStrong";

export interface T1NapoleonEvBiddingAgentOptions {
  marginModel: BiddingMarginOnnxModel;
  passEvAgent: CriticEvBiddingAgent;
  delegateAgent?: Agent;
  playerIds?: readonly PlayerId[];
  diagnostics?: T1NapoleonEvBiddingDiagnostics;
  fallbackOnInferenceError?: boolean;
}

export interface T1NapoleonEvBiddingDiagnostics {
  decisionRecords: T1NapoleonEvBiddingDecisionRecord[];
  fallbackCount: number;
  inferenceFailureCount: number;
  invalidMaskCount: number;
}

export interface T1NapoleonEvBiddingDecisionRecord {
  playerId: PlayerId;
  kind: T1BiddingDecisionKind;
  handStrengthScore: number;
  handStrengthBucket: T1HandStrengthBucket;
  currentHighestBid: {
    playerId: PlayerId;
    targetPointCards: number;
    suit: Suit;
  } | null;
  legalBidActionCount: number;
  selectedActionIndex: number;
  selectedActionType: "bid" | "pass";
  selectedTargetPointCards: number | null;
  selectedSuit: Suit | null;
  selectedPWin: number | null;
  selectedMean: number | null;
  selectedSigma: number | null;
  selectedNapoleonEv: number | null;
  bestBidActionIndex: number | null;
  bestBidPWin: number | null;
  bestBidNapoleonEv: number | null;
  passEv: number;
  evDelta: number | null;
}

export interface T1NapoleonEvCandidateEvaluation {
  actionIndex: number;
  action: BidAction;
  mean: number;
  sigma: number;
  pWin: number;
  napoleonEv: number;
}

export function createT1NapoleonEvBiddingDiagnostics(): T1NapoleonEvBiddingDiagnostics {
  return {
    decisionRecords: [],
    fallbackCount: 0,
    inferenceFailureCount: 0,
    invalidMaskCount: 0
  };
}

export class T1NapoleonEvBiddingAgent implements Agent {
  private readonly marginModel: BiddingMarginOnnxModel;
  private readonly passEvAgent: CriticEvBiddingAgent;
  private readonly delegateAgent: Agent;
  private readonly playerIds: readonly PlayerId[] | null;
  private readonly diagnostics: T1NapoleonEvBiddingDiagnostics | null;
  private readonly fallbackOnInferenceError: boolean;

  constructor(options: T1NapoleonEvBiddingAgentOptions) {
    this.marginModel = options.marginModel;
    this.passEvAgent = options.passEvAgent;
    this.delegateAgent = options.delegateAgent ?? new RuleBasedAgent();
    this.playerIds = options.playerIds ?? null;
    this.diagnostics = options.diagnostics ?? null;
    this.fallbackOnInferenceError = options.fallbackOnInferenceError ?? true;
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    if (observation.view.phase !== "bidding") {
      return this.delegateAgent.selectAction(observation);
    }

    try {
      return await this.selectBiddingAction(observation);
    } catch (error) {
      if (this.diagnostics !== null) {
        this.diagnostics.inferenceFailureCount += 1;
      }
      if (!this.fallbackOnInferenceError) {
        throw error;
      }
      if (this.diagnostics !== null) {
        this.diagnostics.fallbackCount += 1;
      }
      return this.delegateAgent.selectAction(observation);
    }
  }

  async evaluateLegalBidCandidates(
    observation: PlayerObservation
  ): Promise<readonly T1NapoleonEvCandidateEvaluation[]> {
    const { modelInput, legalBidMask } = createPolicyOnnxBiddingInput(
      observation,
      this.resolvePlayerIds(observation)
    );
    const prediction = await this.marginModel.predict(modelInput);
    const evaluations: T1NapoleonEvCandidateEvaluation[] = [];
    for (let actionIndex = 1; actionIndex < legalBidMask.length; actionIndex += 1) {
      if (legalBidMask[actionIndex] !== 1) {
        continue;
      }
      const action = decodeBiddingAction(actionIndex, observation.playerId);
      if (action.type !== "bid") {
        throw new PolicyOnnxCompatibilityError(`decoded legal bid action ${actionIndex} is not a bid.`);
      }
      const mean = prediction.mean[actionIndex];
      const sigma = prediction.sigma[actionIndex];
      const pWin = gaussianSuccessProbability(mean, sigma);
      evaluations.push({
        actionIndex,
        action,
        mean,
        sigma,
        pWin,
        napoleonEv: napoleonRelativeEv(pWin, action.targetPointCards)
      });
    }
    return evaluations.sort((left, right) =>
      right.napoleonEv - left.napoleonEv || left.actionIndex - right.actionIndex
    );
  }

  private async selectBiddingAction(observation: PlayerObservation): Promise<GameAction> {
    const legalPass = observation.legalActions.find((action) => action.type === "pass");
    const currentHighestBid = observation.view.bidding?.highestBid ?? null;
    const kind: T1BiddingDecisionKind = currentHighestBid === null ? "opening" : "raise";
    const bidEvaluations = await this.evaluateLegalBidCandidates(observation);
    const bestBid = bidEvaluations[0] ?? null;
    const passEv = kind === "opening" ? 0 : await this.evaluatePassEv(observation);
    const selected = bestBid !== null && bestBid.napoleonEv > passEv
      ? bestBid.action
      : legalPass;

    if (selected === undefined) {
      if (this.diagnostics !== null) {
        this.diagnostics.invalidMaskCount += 1;
      }
      return bestBid?.action ?? this.delegateAgent.selectAction(observation);
    }

    if (!observation.legalActions.some((legal) => actionsEqual(legal, selected))) {
      if (this.diagnostics !== null) {
        this.diagnostics.invalidMaskCount += 1;
      }
      throw new PolicyOnnxCompatibilityError("T1 Napoleon EV selected an action outside legalActions.");
    }

    this.recordDecision({
      observation,
      kind,
      currentHighestBid,
      bidEvaluations,
      bestBid,
      selected,
      passEv
    });
    return selected;
  }

  private async evaluatePassEv(observation: PlayerObservation): Promise<number> {
    const evaluations = await this.passEvAgent.evaluateLegalBiddingActions(observation);
    const pass = evaluations.find((evaluation) => evaluation.action.type === "pass");
    return pass?.expectedValue ?? 0;
  }

  private recordDecision(args: {
    observation: PlayerObservation;
    kind: T1BiddingDecisionKind;
    currentHighestBid: T1NapoleonEvBiddingDecisionRecord["currentHighestBid"];
    bidEvaluations: readonly T1NapoleonEvCandidateEvaluation[];
    bestBid: T1NapoleonEvCandidateEvaluation | null;
    selected: Extract<GameAction, { type: "bid" | "pass" }>;
    passEv: number;
  }): void {
    if (this.diagnostics === null) {
      return;
    }
    const strength = handStrength(args.observation);
    const selectedBidActionIndex = args.selected.type === "bid"
      ? biddingActionIndex(args.selected)
      : null;
    const selectedBid = selectedBidActionIndex === null
      ? null
      : args.bidEvaluations.find((evaluation) => evaluation.actionIndex === selectedBidActionIndex);
    this.diagnostics.decisionRecords.push({
      playerId: args.observation.playerId,
      kind: args.kind,
      handStrengthScore: strength.score,
      handStrengthBucket: strength.bucket,
      currentHighestBid: args.currentHighestBid === null
        ? null
        : {
            playerId: args.currentHighestBid.playerId,
            targetPointCards: args.currentHighestBid.targetPointCards,
            suit: args.currentHighestBid.suit
          },
      legalBidActionCount: args.bidEvaluations.length,
      selectedActionIndex: args.selected.type === "pass" ? 0 : biddingActionIndex(args.selected),
      selectedActionType: args.selected.type,
      selectedTargetPointCards: args.selected.type === "bid" ? args.selected.targetPointCards : null,
      selectedSuit: args.selected.type === "bid" ? args.selected.suit : null,
      selectedPWin: selectedBid?.pWin ?? null,
      selectedMean: selectedBid?.mean ?? null,
      selectedSigma: selectedBid?.sigma ?? null,
      selectedNapoleonEv: selectedBid?.napoleonEv ?? null,
      bestBidActionIndex: args.bestBid?.actionIndex ?? null,
      bestBidPWin: args.bestBid?.pWin ?? null,
      bestBidNapoleonEv: args.bestBid?.napoleonEv ?? null,
      passEv: args.passEv,
      evDelta: args.bestBid === null ? null : args.bestBid.napoleonEv - args.passEv
    });
  }

  private resolvePlayerIds(observation: PlayerObservation): readonly PlayerId[] {
    return this.playerIds ?? observation.view.players.map((player) => player.id);
  }
}

export function napoleonRelativeEv(pWin: number, targetPointCards: number): number {
  if (!Number.isFinite(pWin) || pWin < 0 || pWin > 1) {
    throw new PolicyOnnxCompatibilityError(`pWin must be in [0, 1], got ${pWin}.`);
  }
  return pWin * ((7 * targetPointCards) / 4) + (1 - pWin) * ((-3 * targetPointCards) / 4);
}

export function gaussianSuccessProbability(mean: number, sigma: number): number {
  if (!Number.isFinite(mean) || !Number.isFinite(sigma) || sigma <= 0) {
    throw new PolicyOnnxCompatibilityError(`invalid Gaussian parameters mean=${mean}, sigma=${sigma}.`);
  }
  return clamp(0.5 * (1 + erf(mean / (sigma * Math.SQRT2))), 0, 1);
}

export function handStrength(observation: PlayerObservation): {
  score: number;
  bucket: T1HandStrengthBucket;
} {
  const self = observation.view.players.find((player) => player.id === observation.playerId);
  const hand = self?.hand ?? [];
  const suits: readonly Suit[] = ["spades", "hearts", "diamonds", "clubs"];
  const score = Math.max(...suits.map((suit) => evaluateHandForTrump(hand, suit)));
  return {
    score,
    bucket: handStrengthBucket(score)
  };
}

export function handStrengthBucket(score: number): T1HandStrengthBucket {
  if (score < 200) return "low";
  if (score < 280) return "medium";
  if (score < 330) return "strong";
  return "veryStrong";
}

function biddingActionIndex(action: BidAction): number {
  const suitIndex = ["spades", "hearts", "diamonds", "clubs"].indexOf(action.suit);
  if (suitIndex === -1) {
    throw new PolicyOnnxCompatibilityError(`unsupported bidding suit ${action.suit}.`);
  }
  return 1 + (action.targetPointCards - 13) * 4 + suitIndex;
}

function actionsEqual(left: GameAction, right: GameAction): boolean {
  if (left.type !== right.type || left.playerId !== right.playerId) {
    return false;
  }
  if (left.type === "pass" && right.type === "pass") {
    return true;
  }
  if (left.type === "bid" && right.type === "bid") {
    return left.suit === right.suit && left.targetPointCards === right.targetPointCards;
  }
  return false;
}

function erf(value: number): number {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return sign * y;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
