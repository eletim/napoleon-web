import {
  RuleBasedAgent,
  runAutomatedGame
} from "@napoleon/ai";
import type { ActualCardState, Agent, AutomatedGameRecord, PlayerObservation } from "@napoleon/ai";
import { calculateNonPlayingLearningTerminalReward } from "@napoleon/training-data";
import type { BidAction, Card, GameAction, GameResult, PlayerId, Suit } from "@napoleon/game-core";
import { createDeck } from "@napoleon/game-core";
import { CriticEvBiddingAgent } from "./criticEvBiddingAgent.js";
import { PolicyOnnxAgent, createPolicyOnnxAgentDecisionMetrics } from "./policyOnnxAgent.js";
import type { PolicyCriticOnnxModel, PolicyOnnxModel, BiddingMarginOnnxModel } from "./policyOnnx.js";
import {
  T1NapoleonEvBiddingAgent,
  createT1NapoleonEvBiddingDiagnostics,
  handStrength
} from "./t1NapoleonEvBiddingAgent.js";
import type {
  T1HandStrengthBucket,
  T1NapoleonEvBiddingDecisionRecord,
  T1NapoleonEvBiddingDiagnostics
} from "./t1NapoleonEvBiddingAgent.js";

const playerIds = ["player-0", "player-1", "player-2", "player-3", "player-4"] as const;
const targetValues = [13, 14, 15, 16, 17, 18, 19] as const;
const suitValues = ["spades", "hearts", "diamonds", "clubs"] as const;
const strengthBuckets = ["low", "medium", "strong", "veryStrong"] as const;
const decisionBoundaryEpsilon = 0.5;
const lowPWinBidThreshold = 0.1;
const highPWinPassThreshold = 0.7;
const cardById = new Map(createDeck().map((card) => [card.id, card]));

export interface RunIssue429T1BiddingRuntimeEvaluationOptions {
  startSeed: number;
  gameCount: number;
  playingPolicy: PolicyOnnxModel;
  critic: PolicyCriticOnnxModel;
  t1MarginModel: BiddingMarginOnnxModel;
  maxDecisionSteps?: number;
  progress?: (message: string) => void;
}

export interface Issue429T1BiddingRuntimeEvaluationResult {
  schemaVersion: 1;
  configuration: {
    startSeed: number;
    endSeed: number;
    gameCount: number;
    playerIds: readonly PlayerId[];
    candidatePlayerId: PlayerId;
    playingPolicy: unknown;
    passCitizenEvSource: "CriticEvBiddingAgent";
    t1MarginModel: unknown;
    t1DecisionRule: string;
  };
  candidates: readonly Issue429CandidateSummary[];
  pairedComparison: {
    t1MinusCurrentCandidateRelativeReward: number | null;
    t1MinusCurrentContractSuccessRate: number | null;
    t1MinusCurrentRaiseRate: number | null;
    t1MinusCurrentOpeningBidRate: number | null;
  };
}

export interface Issue429CandidateSummary {
  label: "A-current-runtime" | "B-t1-runtime";
  agentName: string;
  games: {
    total: number;
    completed: number;
    crashed: number;
    completionRate: number | null;
  };
  bidding: {
    actionsPerGame: number | null;
    candidateDecisionCount: number;
    passCount: number;
    passRate: number | null;
    bidCount: number;
    bidRate: number | null;
    openingDecisionCount: number;
    openingBidCount: number;
    openingBidRate: number | null;
    raiseOpportunityCount: number;
    raiseBidCount: number;
    raiseRate: number | null;
  };
  contracts: {
    allPassCount: number;
    allPassRate: number | null;
    noContractRate: number | null;
    finalCandidateNapoleonCount: number;
    finalCandidateNapoleonRate: number | null;
    targetPointCards: Readonly<Record<string, number>>;
    suits: Readonly<Record<Suit, number>>;
    meanFinalTarget: number | null;
    medianFinalTarget: number | null;
    contractSuccessCount: number;
    contractSuccessRate: number | null;
    contractSuccessRateByTarget: Readonly<Record<string, RateSummary>>;
  };
  handStrength: Readonly<Record<T1HandStrengthBucket, StrengthBucketSummary>>;
  rewards: {
    meanRelativeRewardPerPlayerGame: MeanSummary;
    candidateRelativeReward: MeanSummary;
    byRole: Readonly<Record<string, MeanSummary>>;
  };
  safety: {
    illegalBidCount: number;
    invalidMaskCount: number;
    modelInferenceFailureCount: number;
    fallbackCount: number;
    crashCount: number;
  };
  decisionQuality: {
    available: boolean;
    openingNearZeroRate: RateSummary;
    raiseNearZeroDeltaRate: RateSummary;
    lowPWinBidRate: RateSummary;
    highPWinPassRate: RateSummary;
  };
}

export interface RateSummary {
  numerator: number;
  denominator: number;
  rate: number | null;
}

export interface MeanSummary {
  count: number;
  mean: number | null;
  ci95: { lower: number | null; upper: number | null };
}

export interface StrengthBucketSummary {
  gameCount: number;
  openingBidRate: RateSummary;
  raiseRate: RateSummary;
  finalCandidateNapoleonRate: RateSummary;
  contractSuccessRate: RateSummary;
}

export async function runIssue429T1BiddingRuntimeEvaluation(
  options: RunIssue429T1BiddingRuntimeEvaluationOptions
): Promise<Issue429T1BiddingRuntimeEvaluationResult> {
  const current = await runCandidate({
    label: "A-current-runtime",
    agentName: "CriticEvBidding",
    options,
    createBiddingAgent: () => new CriticEvBiddingAgent({
      critic: options.critic,
      playerIds
    })
  });
  const t1Diagnostics = createT1NapoleonEvBiddingDiagnostics();
  const t1 = await runCandidate({
    label: "B-t1-runtime",
    agentName: "T1NapoleonEvBidding",
    options,
    t1Diagnostics,
    createBiddingAgent: () => new T1NapoleonEvBiddingAgent({
      marginModel: options.t1MarginModel,
      passEvAgent: new CriticEvBiddingAgent({
        critic: options.critic,
        playerIds
      }),
      playerIds,
      diagnostics: t1Diagnostics
    })
  });

  return {
    schemaVersion: 1,
    configuration: {
      startSeed: options.startSeed,
      endSeed: options.startSeed + options.gameCount - 1,
      gameCount: options.gameCount,
      playerIds,
      candidatePlayerId: playerIds[0],
      playingPolicy: options.playingPolicy.metadata,
      passCitizenEvSource: "CriticEvBiddingAgent",
      t1MarginModel: options.t1MarginModel.metadata,
      t1DecisionRule:
        "opening: choose max legal Napoleon EV bid iff EV > 0; raise: choose max legal Napoleon EV raise iff EV > existing CriticEv PASS EV"
    },
    candidates: [current, t1],
    pairedComparison: {
      t1MinusCurrentCandidateRelativeReward: deltaMean(
        t1.rewards.candidateRelativeReward,
        current.rewards.candidateRelativeReward
      ),
      t1MinusCurrentContractSuccessRate: deltaRate(
        t1.contracts.contractSuccessRate,
        current.contracts.contractSuccessRate
      ),
      t1MinusCurrentRaiseRate: deltaRate(t1.bidding.raiseRate, current.bidding.raiseRate),
      t1MinusCurrentOpeningBidRate: deltaRate(
        t1.bidding.openingBidRate,
        current.bidding.openingBidRate
      )
    }
  };
}

async function runCandidate(args: {
  label: Issue429CandidateSummary["label"];
  agentName: string;
  options: RunIssue429T1BiddingRuntimeEvaluationOptions;
  createBiddingAgent: () => Agent;
  t1Diagnostics?: T1NapoleonEvBiddingDiagnostics;
}): Promise<Issue429CandidateSummary> {
  const records: AutomatedGameRecord[] = [];
  const failures: string[] = [];
  const decisionMetrics = createPolicyOnnxAgentDecisionMetrics();
  for (let offset = 0; offset < args.options.gameCount; offset += 1) {
    const seed = args.options.startSeed + offset;
    try {
      const biddingAgent = args.createBiddingAgent();
      const record = await runAutomatedGame({
        seed,
        playerIds,
        maxDecisionSteps: args.options.maxDecisionSteps,
        createAgent: ({ playerIndex, rng }) => playerIndex === 0
          ? new CandidateRuntimeAgent({
              playingPolicy: args.options.playingPolicy,
              biddingAgent,
              rng,
              decisionMetrics
            })
          : new RuleBasedAgent(rng)
      });
      records.push(record);
    } catch (error) {
      failures.push(error instanceof Error ? error.message : String(error));
    }
    if ((offset + 1) % 100 === 0) {
      args.options.progress?.(`${args.agentName}: completed ${offset + 1}/${args.options.gameCount}`);
    }
  }
  return summarizeCandidate({
    label: args.label,
    agentName: args.agentName,
    records,
    failures,
    t1Diagnostics: args.t1Diagnostics
  });
}

class CandidateRuntimeAgent implements Agent {
  private readonly playingAgent: PolicyOnnxAgent;
  private readonly fallbackAgent: RuleBasedAgent;
  private readonly biddingAgent: Agent;

  constructor(options: {
    playingPolicy: PolicyOnnxModel;
    biddingAgent: Agent;
    rng: () => number;
    decisionMetrics: ReturnType<typeof createPolicyOnnxAgentDecisionMetrics>;
  }) {
    this.playingAgent = new PolicyOnnxAgent({
      policy: options.playingPolicy,
      rng: options.rng,
      playerIds,
      decisionMetrics: options.decisionMetrics
    });
    this.fallbackAgent = new RuleBasedAgent(options.rng);
    this.biddingAgent = options.biddingAgent;
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    return this.selectActionWithContext(observation);
  }

  async selectActionWithContext(
    observation: PlayerObservation,
    context?: { actualState: ActualCardState; playerIds: readonly PlayerId[] }
  ): Promise<GameAction> {
    switch (observation.view.phase) {
      case "bidding":
        return this.biddingAgent.selectAction(observation);
      case "playing":
        return this.playingAgent.selectActionWithContext(observation, context);
      case "choosing-adjutant":
      case "exchanging":
      case "finished":
        return this.fallbackAgent.selectAction(observation);
    }
  }
}

function summarizeCandidate(args: {
  label: Issue429CandidateSummary["label"];
  agentName: string;
  records: readonly AutomatedGameRecord[];
  failures: readonly string[];
  t1Diagnostics?: T1NapoleonEvBiddingDiagnostics;
}): Issue429CandidateSummary {
  const candidateDecisionRecords = collectCandidateBiddingDecisions(args.records);
  const allBiddingActionCount = args.records.reduce(
    (sum, record) => sum + record.decisions.filter((decision) => decision.phase === "bidding").length,
    0
  );
  const selectedBids = candidateDecisionRecords.filter((record) => record.action.type === "bid");
  const openingDecisions = candidateDecisionRecords.filter((record) =>
    record.observation.view.bidding?.highestBid === null
  );
  const openingBids = openingDecisions.filter((record) => record.action.type === "bid");
  const raiseOpportunities = candidateDecisionRecords.filter((record) =>
    record.observation.view.bidding?.highestBid !== null &&
    record.legalActions.some((action) => action.type === "bid")
  );
  const raiseBids = raiseOpportunities.filter((record) => record.action.type === "bid");
  const standardRecords = args.records.filter((record) => record.result.resultType === "standard");
  const allPassCount = args.records.length - standardRecords.length;
  const targetDistribution = emptyTargetRecord();
  const suitDistribution = emptySuitRecord();
  const targetSuccess = new Map<number, { success: number; total: number }>();
  const finalTargets: number[] = [];
  for (const record of standardRecords) {
    if (record.result.resultType !== "standard") {
      continue;
    }
    const bid = finalBid(record);
    if (bid !== null) {
      suitDistribution[bid.suit] += 1;
    }
    targetDistribution[String(record.result.targetPointCards)] += 1;
    finalTargets.push(record.result.targetPointCards);
    const bucket = targetSuccess.get(record.result.targetPointCards) ?? { success: 0, total: 0 };
    bucket.total += 1;
    if (record.result.winner === "napoleon-team") {
      bucket.success += 1;
    }
    targetSuccess.set(record.result.targetPointCards, bucket);
  }

  return {
    label: args.label,
    agentName: args.agentName,
    games: {
      total: args.records.length + args.failures.length,
      completed: args.records.length,
      crashed: args.failures.length,
      completionRate: rate(args.records.length, args.records.length + args.failures.length)
    },
    bidding: {
      actionsPerGame: args.records.length === 0 ? null : allBiddingActionCount / args.records.length,
      candidateDecisionCount: candidateDecisionRecords.length,
      passCount: candidateDecisionRecords.length - selectedBids.length,
      passRate: rate(candidateDecisionRecords.length - selectedBids.length, candidateDecisionRecords.length),
      bidCount: selectedBids.length,
      bidRate: rate(selectedBids.length, candidateDecisionRecords.length),
      openingDecisionCount: openingDecisions.length,
      openingBidCount: openingBids.length,
      openingBidRate: rate(openingBids.length, openingDecisions.length),
      raiseOpportunityCount: raiseOpportunities.length,
      raiseBidCount: raiseBids.length,
      raiseRate: rate(raiseBids.length, raiseOpportunities.length)
    },
    contracts: {
      allPassCount,
      allPassRate: rate(allPassCount, args.records.length),
      noContractRate: rate(allPassCount, args.records.length),
      finalCandidateNapoleonCount: standardRecords.filter((record) =>
        record.result.resultType === "standard" && record.result.napoleonPlayerId === playerIds[0]
      ).length,
      finalCandidateNapoleonRate: rate(
        standardRecords.filter((record) =>
          record.result.resultType === "standard" && record.result.napoleonPlayerId === playerIds[0]
        ).length,
        args.records.length
      ),
      targetPointCards: targetDistribution,
      suits: suitDistribution,
      meanFinalTarget: mean(finalTargets),
      medianFinalTarget: median(finalTargets),
      contractSuccessCount: standardRecords.filter((record) =>
        record.result.resultType === "standard" && record.result.winner === "napoleon-team"
      ).length,
      contractSuccessRate: rate(
        standardRecords.filter((record) =>
          record.result.resultType === "standard" && record.result.winner === "napoleon-team"
        ).length,
        standardRecords.length
      ),
      contractSuccessRateByTarget: Object.fromEntries(targetValues.map((target) => {
        const bucket = targetSuccess.get(target) ?? { success: 0, total: 0 };
        return [String(target), rateSummary(bucket.success, bucket.total)];
      }))
    },
    handStrength: summarizeStrength(args.records),
    rewards: summarizeRewards(args.records),
    safety: {
      illegalBidCount: args.failures.filter((failure) => failure.includes("illegal")).length,
      invalidMaskCount: args.t1Diagnostics?.invalidMaskCount ?? 0,
      modelInferenceFailureCount: args.t1Diagnostics?.inferenceFailureCount ?? 0,
      fallbackCount: args.t1Diagnostics?.fallbackCount ?? 0,
      crashCount: args.failures.length
    },
    decisionQuality: summarizeDecisionQuality(args.t1Diagnostics?.decisionRecords ?? [])
  };
}

function collectCandidateBiddingDecisions(records: readonly AutomatedGameRecord[]) {
  return records.flatMap((record) =>
    record.decisions.filter((decision) =>
      decision.phase === "bidding" && decision.playerId === playerIds[0]
    )
  );
}

function finalBid(record: AutomatedGameRecord): BidAction | null {
  for (let index = record.decisions.length - 1; index >= 0; index -= 1) {
    const action = record.decisions[index].action;
    if (action.type === "bid") {
      return action;
    }
  }
  return null;
}

function summarizeStrength(
  records: readonly AutomatedGameRecord[]
): Readonly<Record<T1HandStrengthBucket, StrengthBucketSummary>> {
  return Object.fromEntries(strengthBuckets.map((bucket) => {
    const bucketRecords = records.filter((record) => candidateInitialStrengthBucket(record) === bucket);
    const bucketDecisions = collectCandidateBiddingDecisions(bucketRecords);
    const opening = bucketDecisions.filter((decision) => decision.observation.view.bidding?.highestBid === null);
    const raise = bucketDecisions.filter((decision) =>
      decision.observation.view.bidding?.highestBid !== null &&
      decision.legalActions.some((action) => action.type === "bid")
    );
    const standard = bucketRecords.filter((record) => record.result.resultType === "standard");
    const candidateNapoleon = standard.filter((record) =>
      record.result.resultType === "standard" && record.result.napoleonPlayerId === playerIds[0]
    );
    const candidateContracts = candidateNapoleon;
    return [bucket, {
      gameCount: bucketRecords.length,
      openingBidRate: rateSummary(
        opening.filter((decision) => decision.action.type === "bid").length,
        opening.length
      ),
      raiseRate: rateSummary(
        raise.filter((decision) => decision.action.type === "bid").length,
        raise.length
      ),
      finalCandidateNapoleonRate: rateSummary(candidateNapoleon.length, bucketRecords.length),
      contractSuccessRate: rateSummary(
        candidateContracts.filter((record) =>
          record.result.resultType === "standard" && record.result.winner === "napoleon-team"
        ).length,
        candidateContracts.length
      )
    }];
  })) as Readonly<Record<T1HandStrengthBucket, StrengthBucketSummary>>;
}

function candidateInitialStrengthBucket(record: AutomatedGameRecord): T1HandStrengthBucket {
  const cards = record.initialHands[playerIds[0]]
    .map((cardId) => cardById.get(cardId))
    .filter((card): card is Card => card !== undefined);
  const fakeObservation = {
    playerId: playerIds[0],
    view: {
      players: [{ id: playerIds[0], hand: cards }]
    }
  } as unknown as PlayerObservation;
  return handStrength(fakeObservation).bucket;
}

function summarizeRewards(records: readonly AutomatedGameRecord[]): Issue429CandidateSummary["rewards"] {
  const allRewards: number[] = [];
  const candidateRewards: number[] = [];
  const byRole = new Map<string, number[]>();
  for (const record of records) {
    for (const playerId of playerIds) {
      const role = roleForResult(record.result, playerId);
      const reward = calculateNonPlayingLearningTerminalReward(record.result, playerId, playerIds).terminalReward;
      allRewards.push(reward);
      if (playerId === playerIds[0]) {
        candidateRewards.push(reward);
      }
      const values = byRole.get(role) ?? [];
      values.push(reward);
      byRole.set(role, values);
    }
  }
  return {
    meanRelativeRewardPerPlayerGame: meanSummary(allRewards),
    candidateRelativeReward: meanSummary(candidateRewards),
    byRole: Object.fromEntries([...byRole.entries()].sort().map(([role, values]) => [
      role,
      meanSummary(values)
    ]))
  };
}

function summarizeDecisionQuality(
  records: readonly T1NapoleonEvBiddingDecisionRecord[]
): Issue429CandidateSummary["decisionQuality"] {
  const opening = records.filter((record) => record.kind === "opening" && record.evDelta !== null);
  const raise = records.filter((record) => record.kind === "raise" && record.evDelta !== null);
  const bids = records.filter((record) => record.selectedActionType === "bid");
  const passes = records.filter((record) => record.selectedActionType === "pass");
  return {
    available: records.length > 0,
    openingNearZeroRate: rateSummary(
      opening.filter((record) => Math.abs(record.evDelta ?? Infinity) <= decisionBoundaryEpsilon).length,
      opening.length
    ),
    raiseNearZeroDeltaRate: rateSummary(
      raise.filter((record) => Math.abs(record.evDelta ?? Infinity) <= decisionBoundaryEpsilon).length,
      raise.length
    ),
    lowPWinBidRate: rateSummary(
      bids.filter((record) => (record.selectedPWin ?? 1) < lowPWinBidThreshold).length,
      bids.length
    ),
    highPWinPassRate: rateSummary(
      passes.filter((record) => (record.bestBidPWin ?? 0) > highPWinPassThreshold).length,
      passes.length
    )
  };
}

function roleForResult(result: GameResult, playerId: PlayerId): string {
  if (result.resultType === "all-pass") {
    return playerId === result.starterPlayerId ? "all-pass-starter" : "all-pass-other";
  }
  if (playerId === result.napoleonPlayerId) {
    return result.adjutantPlayerId === null || result.adjutantPlayerId === playerId
      ? "napoleon-adjutant"
      : "napoleon";
  }
  if (playerId === result.adjutantPlayerId) {
    return "adjutant";
  }
  return "citizen";
}

function emptyTargetRecord(): Record<string, number> {
  return Object.fromEntries(targetValues.map((target) => [String(target), 0]));
}

function emptySuitRecord(): Record<Suit, number> {
  return Object.fromEntries(suitValues.map((suit) => [suit, 0])) as Record<Suit, number>;
}

function rate(numerator: number, denominator: number): number | null {
  return denominator === 0 ? null : numerator / denominator;
}

function rateSummary(numerator: number, denominator: number): RateSummary {
  return { numerator, denominator, rate: rate(numerator, denominator) };
}

function mean(values: readonly number[]): number | null {
  return values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: readonly number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function meanSummary(values: readonly number[]): MeanSummary {
  const average = mean(values);
  if (average === null) {
    return { count: 0, mean: null, ci95: { lower: null, upper: null } };
  }
  if (values.length < 2) {
    return { count: values.length, mean: average, ci95: { lower: average, upper: average } };
  }
  const variance = values.reduce((sum, value) => sum + (value - average) ** 2, 0) / (values.length - 1);
  const margin = 1.96 * Math.sqrt(variance / values.length);
  return {
    count: values.length,
    mean: average,
    ci95: { lower: average - margin, upper: average + margin }
  };
}

function deltaRate(left: number | null, right: number | null): number | null {
  return left === null || right === null ? null : left - right;
}

function deltaMean(left: MeanSummary, right: MeanSummary): number | null {
  return left.mean === null || right.mean === null ? null : left.mean - right.mean;
}
