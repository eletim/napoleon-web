import {
  RuleBasedAgent,
  createEvaluationReport,
  runEvaluation
} from "@napoleon/ai";
import type {
  ActualCardState,
  Agent,
  EvaluationAgentDefinition,
  EvaluationComparisonSummary,
  EvaluationConfidenceInterval,
  EvaluationFailureSummary,
  EvaluationGameCountSummary,
  EvaluationGameRecord,
  EvaluationPerformanceSummary,
  EvaluationRateSummary,
  EvaluationReport,
  EvaluationRolePerformanceSummary,
  EvaluationRunRecord,
  EvaluationSeatAssignment,
  EvaluationSeatPerformanceSummary,
  EvaluationSeatRole,
  PlayerObservation
} from "@napoleon/ai";
import type { GameAction, GameResult, PlayerId, Suit } from "@napoleon/game-core";
import { CriticEvBiddingAgent } from "./criticEvBiddingAgent.js";
import type { PolicyCriticValueModel } from "./criticEvBiddingAgent.js";
import { getNonPlayingPolicyOnnxSpec } from "./policySpecs.js";
import {
  PolicyOnnxAgent,
  createPolicyOnnxAgentDecisionMetrics
} from "./policyOnnxAgent.js";
import type {
  PolicyOnnxAgentDecisionMetrics
} from "./policyOnnxAgent.js";
import type { NonPlayingPolicyOnnxModel, PolicyOnnxModel } from "./policyOnnx.js";
import type {
  NonPlayingPolicyOnnxMetadata,
  PolicyOnnxInferenceDevice,
  NonPlayingPolicyType,
  PolicyOnnxMetadata,
  PolicyOnnxRuntimeInfo
} from "./types.js";
import {
  RL_V740_BENCHMARK_POLICY_ID,
  loadRepoManagedPlayingPolicyBenchmark,
  validatePlayingPolicyArtifactReference
} from "./benchmarkArtifacts.js";
import type {
  PlayingPolicyArtifactReference
} from "./benchmarkArtifacts.js";

type CompletedRole = Exclude<EvaluationSeatRole, "unknown">;
type AgentGroup = "policy" | "rule-based";
type BiddingBenchmarkCandidateKind = "rule-based" | "critic-ev" | "ppo";
type BiddingRole = "napoleon" | "adjutant" | "citizen" | "napoleon-adjutant";

const defaultPlayerIds: readonly PlayerId[] = [
  "player-0",
  "player-1",
  "player-2",
  "player-3",
  "player-4"
];
const defaultRotationOffsets = [0, 1, 2, 3, 4] as const;
const completedRoles: readonly CompletedRole[] = ["napoleon", "adjutant", "alliance"];
const biddingRoles: readonly BiddingRole[] = [
  "napoleon",
  "adjutant",
  "citizen",
  "napoleon-adjutant"
];
const confidenceLevel = 0.95;
const z95 = 1.959963984540054;

export interface RunPolicyVsRuleBasedEvaluationOptions {
  policy: PolicyOnnxModel;
  startSeed: number;
  gameCount: number;
  playerIds?: readonly PlayerId[];
  rotationOffsets?: readonly number[];
  maxDecisionSteps?: number;
  policyAgentName?: string;
  ruleBasedAgentName?: string;
}

export interface PlayingPolicyRuleBasedOpponent {
  type: "rule-based";
  agentName?: string;
}

export interface PlayingPolicyOnnxOpponent {
  type: "playing-onnx";
  policy: PolicyOnnxModel;
  agentName?: string;
  artifact?: PlayingPolicyArtifactReference;
}

export type PlayingPolicyEvaluationOpponent =
  | PlayingPolicyRuleBasedOpponent
  | PlayingPolicyOnnxOpponent;

export type StandardPlayingPolicyBenchmarkId =
  | "rule-based-x4"
  | "rl-v740-x4"
  | "rule-based-x2-rl-v740-x2";

export interface RunPlayingPolicyRosterEvaluationOptions {
  candidatePolicy: PolicyOnnxModel;
  opponentRoster: readonly PlayingPolicyEvaluationOpponent[];
  startSeed: number;
  gameCount: number;
  playerIds?: readonly PlayerId[];
  rotationOffsets?: readonly number[];
  opponentAgentOrders?: readonly (readonly number[])[];
  maxDecisionSteps?: number;
  candidateAgentName?: string;
}

export interface RunStandardPlayingPolicyBenchmarksOptions {
  candidatePolicy: PolicyOnnxModel;
  benchmarks?: readonly StandardPlayingPolicyBenchmarkId[];
  startSeed: number;
  gameCount: number;
  playerIds?: readonly PlayerId[];
  rotationOffsets?: readonly number[];
  maxDecisionSteps?: number;
  candidateAgentName?: string;
  inferenceDevice?: PolicyOnnxInferenceDevice;
}

export interface RunFullPolicyVsRuleBasedEvaluationOptions {
  playingPolicy: PolicyOnnxModel;
  biddingPolicy: NonPlayingPolicyOnnxModel;
  adjutantPolicy: NonPlayingPolicyOnnxModel;
  exchangePolicy: NonPlayingPolicyOnnxModel;
  startSeed: number;
  gameCount: number;
  playerIds?: readonly PlayerId[];
  rotationOffsets?: readonly number[];
  maxDecisionSteps?: number;
  policyAgentName?: string;
  ruleBasedAgentName?: string;
}

export interface RunBiddingPolicyBenchmarkOptions {
  playingPolicy: PolicyOnnxModel;
  ppoBiddingPolicy: NonPlayingPolicyOnnxModel;
  criticEvBiddingCritic: PolicyCriticValueModel;
  startSeed: number;
  gameCount: number;
  playerIds?: readonly PlayerId[];
  rotationOffsets?: readonly number[];
  maxDecisionSteps?: number;
}

export interface PolicyVsRuleBasedEvaluationConfiguration {
  startSeed: number;
  endSeed: number;
  gameCount: number;
  rotationOffsets: readonly number[];
  playerIds: readonly PlayerId[];
  policyAgentName: string;
  ruleBasedAgentName: string;
  policyMetadata: PolicyOnnxMetadata;
  policyRuntime: PolicyOnnxRuntimeInfo;
}

export interface PlayingPolicyOpponentRosterEntry {
  sourceAgentIndex: number;
  type: PlayingPolicyEvaluationOpponent["type"];
  agentName: string;
  artifact?: PlayingPolicyArtifactReference;
  runtime?: PolicyOnnxRuntimeInfo;
}

export interface PlayingPolicyRosterEvaluationConfiguration {
  startSeed: number;
  endSeed: number;
  gameCount: number;
  rotationOffsets: readonly number[];
  playerIds: readonly PlayerId[];
  candidateAgentName: string;
  candidateMetadata: PolicyOnnxMetadata;
  candidateRuntime: PolicyOnnxRuntimeInfo;
  opponentRoster: readonly PlayingPolicyOpponentRosterEntry[];
  agentOrders: readonly (readonly number[])[];
}

export interface FullPolicyVsRuleBasedEvaluationConfiguration {
  startSeed: number;
  endSeed: number;
  gameCount: number;
  rotationOffsets: readonly number[];
  playerIds: readonly PlayerId[];
  policyAgentName: string;
  ruleBasedAgentName: string;
  policyMetadata: {
    playing: PolicyOnnxMetadata;
    bidding: NonPlayingPolicyOnnxMetadata;
    adjutant: NonPlayingPolicyOnnxMetadata;
    exchange: NonPlayingPolicyOnnxMetadata;
  };
  policyRuntime: {
    playing: PolicyOnnxRuntimeInfo;
    bidding: PolicyOnnxRuntimeInfo;
    adjutant: PolicyOnnxRuntimeInfo;
    exchange: PolicyOnnxRuntimeInfo;
  };
}

export interface FailedPolicyVsRuleBasedGame {
  gameIndex: number;
  seed: number;
  rotationOffset: number;
  failureReason: string;
}

export interface PolicyVsRuleBasedAgentSummary extends EvaluationPerformanceSummary {
  agentGroup: AgentGroup;
  agentName: string;
  sourceAgentIndices: readonly number[];
  roleResults: readonly EvaluationRolePerformanceSummary[];
  seatResults: readonly EvaluationSeatPerformanceSummary[];
  comparison: EvaluationComparisonSummary;
}

export interface PolicyVsRuleBasedComparisonReport {
  schemaVersion: 1;
  illegalActionCount: number;
  failedGames: readonly FailedPolicyVsRuleBasedGame[];
  policy: PolicyVsRuleBasedAgentSummary;
  ruleBased: PolicyVsRuleBasedAgentSummary;
}

export interface PlayingPolicyRosterEvaluationResult {
  schemaVersion: 1;
  configuration: PlayingPolicyRosterEvaluationConfiguration;
  run: EvaluationRunRecord;
  report: EvaluationReport;
  comparison: PolicyVsRuleBasedComparisonReport;
}

export interface StandardPlayingPolicyBenchmarkResult {
  benchmarkId: StandardPlayingPolicyBenchmarkId;
  result: PlayingPolicyRosterEvaluationResult | PolicyVsRuleBasedEvaluationResult;
}

export interface StandardPlayingPolicyBenchmarkSuiteResult {
  schemaVersion: 1;
  candidateMetadata: PolicyOnnxMetadata;
  benchmarks: readonly StandardPlayingPolicyBenchmarkResult[];
}

export interface PolicyVsRuleBasedEvaluationResult {
  schemaVersion: 1;
  configuration: PolicyVsRuleBasedEvaluationConfiguration;
  run: EvaluationRunRecord;
  report: EvaluationReport;
  comparison: PolicyVsRuleBasedComparisonReport;
}

export interface FullPolicyVsRuleBasedDiagnostics {
  policyAgentDecisionCounts: PolicyOnnxAgentDecisionMetrics;
  adjutantSelection: AdjutantSelectionDistributionSummary;
}

export interface AdjutantSelectionDistributionSummary {
  decisionCount: number;
  cardIds: Readonly<Record<string, number>>;
}

export interface FullPolicyVsRuleBasedEvaluationResult {
  schemaVersion: 1;
  configuration: FullPolicyVsRuleBasedEvaluationConfiguration;
  run: EvaluationRunRecord;
  report: EvaluationReport;
  comparison: PolicyVsRuleBasedComparisonReport;
  diagnostics: FullPolicyVsRuleBasedDiagnostics;
}

export interface BiddingActionDistributionSummary {
  decisionCount: number;
  passCount: number;
  passRate: number | null;
  bidCount: number;
  targetPointCards: Readonly<Record<string, number>>;
  suits: Readonly<Record<Suit, number>>;
}

export interface BiddingContractSummary {
  completedGameCount: number;
  napoleonFormationCount: number;
  napoleonFormationRate: number | null;
  declarationSuccessCount: number;
  declarationSuccessRate: number | null;
  averageTargetPointCards: number | null;
  targetPointCards: Readonly<Record<string, number>>;
}

export interface BiddingRoleRewardSummary {
  role: BiddingRole;
  sampleCount: number;
  averageReward: number | null;
}

export interface BiddingPolicyBenchmarkCandidateResult {
  kind: BiddingBenchmarkCandidateKind;
  agentName: string;
  run: EvaluationRunRecord;
  report: EvaluationReport;
  comparison: PolicyVsRuleBasedComparisonReport;
  bidding: BiddingActionDistributionSummary;
  contracts: BiddingContractSummary;
  roleRewards: readonly BiddingRoleRewardSummary[];
  illegalActionCount: number;
  biddingOnnxDecisionCount: number;
}

export interface BiddingPolicyBenchmarkResult {
  schemaVersion: 1;
  configuration: {
    startSeed: number;
    endSeed: number;
    gameCount: number;
    rotationOffsets: readonly number[];
    playerIds: readonly PlayerId[];
    playingPolicyMetadata: PolicyOnnxMetadata;
    ppoBiddingPolicyMetadata: NonPlayingPolicyOnnxMetadata;
  };
  candidates: readonly BiddingPolicyBenchmarkCandidateResult[];
}

interface MutableStats {
  games: {
    total: number;
    completed: number;
    failed: number;
  };
  wins: number;
  losses: number;
  contractSuccesses: number;
  pointCardTotal: number;
  pointCardSquareTotal: number;
  failuresByReason: Map<string, number>;
}

interface AgentGroupStats {
  stats: MutableStats;
  sourceAgentIndices: Set<number>;
  roleStats: Map<CompletedRole, MutableStats>;
  seatStats: Map<number, MutableStats>;
}

interface BiddingActionMetrics {
  decisionCount: number;
  passCount: number;
  bidCount: number;
  targetPointCards: Map<number, number>;
  suits: Map<Suit, number>;
}

interface AdjutantSelectionMetrics {
  decisionCount: number;
  cardIds: Map<string, number>;
}

export async function runPolicyVsRuleBasedEvaluation(
  options: RunPolicyVsRuleBasedEvaluationOptions
): Promise<PolicyVsRuleBasedEvaluationResult> {
  const playerIds = options.playerIds ?? defaultPlayerIds;
  const rotationOffsets = options.rotationOffsets ?? defaultRotationOffsets;
  const policyAgentName = options.policyAgentName ?? "PolicyOnnxAgent";
  const ruleBasedAgentName = options.ruleBasedAgentName ?? "RuleBasedAgent";
  const agents = createPolicyVsRuleBasedAgents({
    policy: options.policy,
    playerIds,
    policyAgentName,
    ruleBasedAgentName
  });
  const run = await runEvaluation({
    startSeed: options.startSeed,
    gameCount: options.gameCount,
    playerIds,
    rotationOffsets,
    maxDecisionSteps: options.maxDecisionSteps,
    agents
  });
  const report = createEvaluationReport(run);

  return {
    schemaVersion: 1,
    configuration: {
      startSeed: run.startSeed,
      endSeed: run.endSeed,
      gameCount: run.gameCount,
      rotationOffsets: run.rotationOffsets,
      playerIds: run.playerIds,
      policyAgentName,
      ruleBasedAgentName,
      policyMetadata: options.policy.metadata,
      policyRuntime: options.policy.runtime
    },
    run,
    report,
    comparison: createPolicyVsRuleBasedComparison(run, {
      policyAgentName,
      ruleBasedAgentName
    })
  };
}

export async function runPlayingPolicyRosterEvaluation(
  options: RunPlayingPolicyRosterEvaluationOptions
): Promise<PlayingPolicyRosterEvaluationResult> {
  assertPlayingPolicyType("candidatePolicy", options.candidatePolicy);
  validateOpponentRoster(options.opponentRoster);
  await validateOpponentRosterArtifacts(options.opponentRoster);

  const playerIds = options.playerIds ?? defaultPlayerIds;
  const rotationOffsets = options.rotationOffsets ?? defaultRotationOffsets;
  const candidateAgentName = options.candidateAgentName ?? "PolicyOnnxAgent";
  const agents = createPlayingPolicyRosterAgents({
    candidatePolicy: options.candidatePolicy,
    opponentRoster: options.opponentRoster,
    playerIds,
    candidateAgentName
  });
  const agentOrders = options.opponentAgentOrders ?? createDefaultRosterAgentOrders(
    options.opponentRoster
  );
  assertCandidateFirstAgentOrders(agentOrders);
  const run = await runEvaluation({
    startSeed: options.startSeed,
    gameCount: options.gameCount,
    playerIds,
    rotationOffsets,
    agentOrders,
    maxDecisionSteps: options.maxDecisionSteps,
    agents
  });
  const report = createEvaluationReport(run);

  return {
    schemaVersion: 1,
    configuration: {
      startSeed: run.startSeed,
      endSeed: run.endSeed,
      gameCount: run.gameCount,
      rotationOffsets: run.rotationOffsets,
      playerIds: run.playerIds,
      candidateAgentName,
      candidateMetadata: options.candidatePolicy.metadata,
      candidateRuntime: options.candidatePolicy.runtime,
      opponentRoster: options.opponentRoster.map((opponent, index) =>
        toOpponentRosterEntry(opponent, index + 1)
      ),
      agentOrders
    },
    run,
    report,
    comparison: createPolicyVsRuleBasedComparison(run, {
      policyAgentName: candidateAgentName,
      ruleBasedAgentName: "OpponentRoster"
    })
  };
}

export async function runStandardPlayingPolicyBenchmarks(
  options: RunStandardPlayingPolicyBenchmarksOptions
): Promise<StandardPlayingPolicyBenchmarkSuiteResult> {
  const benchmarkIds = options.benchmarks ?? [
    "rule-based-x4",
    "rl-v740-x4",
    "rule-based-x2-rl-v740-x2"
  ];
  const rlV740 = benchmarkIds.some((benchmarkId) => benchmarkId.includes("rl-v740"))
    ? await loadRepoManagedPlayingPolicyBenchmark(RL_V740_BENCHMARK_POLICY_ID, {
        inferenceDevice: options.inferenceDevice
      })
    : null;
  const benchmarks: StandardPlayingPolicyBenchmarkResult[] = [];

  for (const benchmarkId of benchmarkIds) {
    if (benchmarkId === "rule-based-x4") {
      benchmarks.push({
        benchmarkId,
        result: await runPolicyVsRuleBasedEvaluation({
          policy: options.candidatePolicy,
          startSeed: options.startSeed,
          gameCount: options.gameCount,
          playerIds: options.playerIds,
          rotationOffsets: options.rotationOffsets,
          maxDecisionSteps: options.maxDecisionSteps,
          policyAgentName: options.candidateAgentName
        })
      });
      continue;
    }

    if (rlV740 === null) {
      throw new Error("RL v740 benchmark artifact was not loaded.");
    }

    benchmarks.push({
      benchmarkId,
      result: await runPlayingPolicyRosterEvaluation({
        candidatePolicy: options.candidatePolicy,
        opponentRoster: createStandardOpponentRoster(benchmarkId, rlV740),
        startSeed: options.startSeed,
        gameCount: options.gameCount,
        playerIds: options.playerIds,
        rotationOffsets: options.rotationOffsets,
        maxDecisionSteps: options.maxDecisionSteps,
        candidateAgentName: options.candidateAgentName
      })
    });
  }

  return {
    schemaVersion: 1,
    candidateMetadata: options.candidatePolicy.metadata,
    benchmarks
  };
}

export async function runBiddingPolicyBenchmark(
  options: RunBiddingPolicyBenchmarkOptions
): Promise<BiddingPolicyBenchmarkResult> {
  assertPlayingPolicyType("playingPolicy", options.playingPolicy);
  assertNonPlayingPolicyModelType("ppoBiddingPolicy", options.ppoBiddingPolicy, "bidding");

  const playerIds = options.playerIds ?? defaultPlayerIds;
  const rotationOffsets = options.rotationOffsets ?? defaultRotationOffsets;
  const candidates = [
    {
      kind: "rule-based" as const,
      agentName: "RuleBasedBidding",
      createBiddingAgent: (rng: () => number) => new RuleBasedAgent(rng)
    },
    {
      kind: "critic-ev" as const,
      agentName: "CriticEvBidding",
      createBiddingAgent: (rng: () => number) => new CriticEvBiddingAgent({
        critic: options.criticEvBiddingCritic,
        delegateAgent: new RuleBasedAgent(rng),
        playerIds
      })
    },
    {
      kind: "ppo" as const,
      agentName: "PpoBidding",
      createBiddingAgent: (
        rng: () => number,
        decisionMetrics: PolicyOnnxAgentDecisionMetrics
      ) => new PolicyOnnxAgent({
        policy: options.playingPolicy,
        biddingPolicy: options.ppoBiddingPolicy,
        rng,
        playerIds,
        decisionMetrics
      })
    }
  ];
  const results: BiddingPolicyBenchmarkCandidateResult[] = [];

  for (const candidate of candidates) {
    const biddingMetrics = createBiddingActionMetrics();
    const decisionMetrics = createPolicyOnnxAgentDecisionMetrics();
    const run = await runEvaluation({
      startSeed: options.startSeed,
      gameCount: options.gameCount,
      playerIds,
      rotationOffsets,
      maxDecisionSteps: options.maxDecisionSteps,
      agents: createBiddingBenchmarkAgents({
        playingPolicy: options.playingPolicy,
        playerIds,
        candidateAgentName: candidate.agentName,
        biddingMetrics,
        decisionMetrics,
        createBiddingAgent: candidate.createBiddingAgent
      })
    });
    const report = createEvaluationReport(run);
    const comparison = createPolicyVsRuleBasedComparison(run, {
      policyAgentName: candidate.agentName,
      ruleBasedAgentName: "RuleBasedAgent"
    });

    results.push({
      kind: candidate.kind,
      agentName: candidate.agentName,
      run,
      report,
      comparison,
      bidding: summarizeBiddingActions(biddingMetrics),
      contracts: summarizeBiddingContracts(run),
      roleRewards: summarizeBiddingRoleRewards(run),
      illegalActionCount: comparison.illegalActionCount,
      biddingOnnxDecisionCount: decisionMetrics.biddingOnnxCallCount
    });
  }

  return {
    schemaVersion: 1,
    configuration: {
      startSeed: options.startSeed,
      endSeed: options.startSeed + options.gameCount - 1,
      gameCount: options.gameCount,
      rotationOffsets,
      playerIds,
      playingPolicyMetadata: options.playingPolicy.metadata,
      ppoBiddingPolicyMetadata: options.ppoBiddingPolicy.metadata
    },
    candidates: results
  };
}

export async function runFullPolicyVsRuleBasedEvaluation(
  options: RunFullPolicyVsRuleBasedEvaluationOptions
): Promise<FullPolicyVsRuleBasedEvaluationResult> {
  assertFullPolicyTypes(options);

  const playerIds = options.playerIds ?? defaultPlayerIds;
  const rotationOffsets = options.rotationOffsets ?? defaultRotationOffsets;
  const policyAgentName = options.policyAgentName ?? "FullPolicyOnnxAgent";
  const ruleBasedAgentName = options.ruleBasedAgentName ?? "RuleBasedAgent";
  const decisionMetrics = createPolicyOnnxAgentDecisionMetrics();
  const adjutantSelectionMetrics = createAdjutantSelectionMetrics();
  const agents = createFullPolicyVsRuleBasedAgents({
    playingPolicy: options.playingPolicy,
    biddingPolicy: options.biddingPolicy,
    adjutantPolicy: options.adjutantPolicy,
    exchangePolicy: options.exchangePolicy,
    playerIds,
    policyAgentName,
    ruleBasedAgentName,
    decisionMetrics,
    adjutantSelectionMetrics
  });
  const run = await runEvaluation({
    startSeed: options.startSeed,
    gameCount: options.gameCount,
    playerIds,
    rotationOffsets,
    maxDecisionSteps: options.maxDecisionSteps,
    agents
  });
  const report = createEvaluationReport(run);

  return {
    schemaVersion: 1,
    configuration: {
      startSeed: run.startSeed,
      endSeed: run.endSeed,
      gameCount: run.gameCount,
      rotationOffsets: run.rotationOffsets,
      playerIds: run.playerIds,
      policyAgentName,
      ruleBasedAgentName,
      policyMetadata: {
        playing: options.playingPolicy.metadata,
        bidding: options.biddingPolicy.metadata,
        adjutant: options.adjutantPolicy.metadata,
        exchange: options.exchangePolicy.metadata
      },
      policyRuntime: {
        playing: options.playingPolicy.runtime,
        bidding: options.biddingPolicy.runtime,
        adjutant: options.adjutantPolicy.runtime,
        exchange: options.exchangePolicy.runtime
      }
    },
    run,
    report,
    comparison: createPolicyVsRuleBasedComparison(run, {
      policyAgentName,
      ruleBasedAgentName
    }),
    diagnostics: {
      policyAgentDecisionCounts: copyDecisionMetrics(decisionMetrics),
      adjutantSelection: summarizeAdjutantSelections(adjutantSelectionMetrics)
    }
  };
}

function createBiddingBenchmarkAgents(args: {
  playingPolicy: PolicyOnnxModel;
  playerIds: readonly PlayerId[];
  candidateAgentName: string;
  biddingMetrics: BiddingActionMetrics;
  decisionMetrics: PolicyOnnxAgentDecisionMetrics;
  createBiddingAgent: (
    rng: () => number,
    decisionMetrics: PolicyOnnxAgentDecisionMetrics
  ) => Agent;
}): readonly EvaluationAgentDefinition[] {
  return [
    {
      name: args.candidateAgentName,
      createAgent: ({ rng }) => new BiddingBenchmarkAgent({
        playingPolicy: args.playingPolicy,
        biddingAgent: args.createBiddingAgent(rng, args.decisionMetrics),
        rng,
        playerIds: args.playerIds,
        biddingMetrics: args.biddingMetrics,
        decisionMetrics: args.decisionMetrics
      })
    },
    ...args.playerIds.slice(1).map((): EvaluationAgentDefinition => ({
      name: "RuleBasedAgent",
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    }))
  ];
}

class BiddingBenchmarkAgent implements Agent {
  private readonly playingAgent: PolicyOnnxAgent;
  private readonly fallbackAgent: RuleBasedAgent;
  private readonly biddingAgent: Agent;
  private readonly biddingMetrics: BiddingActionMetrics;

  constructor(options: {
    playingPolicy: PolicyOnnxModel;
    biddingAgent: Agent;
    rng: () => number;
    playerIds: readonly PlayerId[];
    biddingMetrics: BiddingActionMetrics;
    decisionMetrics: PolicyOnnxAgentDecisionMetrics;
  }) {
    this.playingAgent = new PolicyOnnxAgent({
      policy: options.playingPolicy,
      rng: options.rng,
      playerIds: options.playerIds,
      decisionMetrics: options.decisionMetrics
    });
    this.fallbackAgent = new RuleBasedAgent(options.rng);
    this.biddingAgent = options.biddingAgent;
    this.biddingMetrics = options.biddingMetrics;
  }

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    return this.selectActionWithContext(observation);
  }

  async selectActionWithContext(
    observation: PlayerObservation,
    context?: { actualState: ActualCardState; playerIds: readonly PlayerId[] }
  ): Promise<GameAction> {
    switch (observation.view.phase) {
      case "bidding": {
        const action = await this.biddingAgent.selectAction(observation);
        recordBiddingAction(this.biddingMetrics, action);
        return action;
      }
      case "playing":
        return this.playingAgent.selectActionWithContext(observation, context);
      case "choosing-adjutant":
      case "exchanging":
      case "finished":
        return this.fallbackAgent.selectAction(observation);
    }
  }
}

function createBiddingActionMetrics(): BiddingActionMetrics {
  return {
    decisionCount: 0,
    passCount: 0,
    bidCount: 0,
    targetPointCards: new Map(),
    suits: new Map()
  };
}

function recordBiddingAction(metrics: BiddingActionMetrics, action: GameAction): void {
  if (action.type !== "bid" && action.type !== "pass") {
    throw new Error(`Bidding benchmark expected bid/pass action, got ${action.type}.`);
  }

  metrics.decisionCount += 1;
  if (action.type === "pass") {
    metrics.passCount += 1;
    return;
  }

  metrics.bidCount += 1;
  incrementNumberMap(metrics.targetPointCards, action.targetPointCards);
  incrementSuitMap(metrics.suits, action.suit);
}

function summarizeBiddingActions(
  metrics: BiddingActionMetrics
): BiddingActionDistributionSummary {
  return {
    decisionCount: metrics.decisionCount,
    passCount: metrics.passCount,
    passRate: metrics.decisionCount === 0 ? null : metrics.passCount / metrics.decisionCount,
    bidCount: metrics.bidCount,
    targetPointCards: targetDistributionRecord(metrics.targetPointCards),
    suits: suitDistributionRecord(metrics.suits)
  };
}

function summarizeBiddingContracts(run: EvaluationRunRecord): BiddingContractSummary {
  let completedGameCount = 0;
  let napoleonFormationCount = 0;
  let declarationSuccessCount = 0;
  let targetTotal = 0;
  const targetPointCards = new Map<number, number>();

  for (const game of run.games) {
    if (game.status !== "completed") {
      continue;
    }
    completedGameCount += 1;
    targetTotal += game.contract.targetPointCards;
    incrementNumberMap(targetPointCards, game.contract.targetPointCards);
    if (game.result.targetPointCards >= 13) {
      napoleonFormationCount += 1;
    }
    if (game.contractSucceeded) {
      declarationSuccessCount += 1;
    }
  }

  return {
    completedGameCount,
    napoleonFormationCount,
    napoleonFormationRate: completedGameCount === 0
      ? null
      : napoleonFormationCount / completedGameCount,
    declarationSuccessCount,
    declarationSuccessRate: completedGameCount === 0
      ? null
      : declarationSuccessCount / completedGameCount,
    averageTargetPointCards: completedGameCount === 0 ? null : targetTotal / completedGameCount,
    targetPointCards: targetDistributionRecord(targetPointCards)
  };
}

function summarizeBiddingRoleRewards(
  run: EvaluationRunRecord
): readonly BiddingRoleRewardSummary[] {
  const accumulators = new Map<BiddingRole, { sampleCount: number; rewardTotal: number }>();
  for (const role of biddingRoles) {
    accumulators.set(role, { sampleCount: 0, rewardTotal: 0 });
  }

  for (const game of run.games) {
    if (game.status !== "completed") {
      continue;
    }
    const candidateSeat = game.seats.find((seat) => seat.sourceAgentIndex === 0);
    if (candidateSeat === undefined) {
      continue;
    }
    const role = biddingRoleForSeat(game.result, candidateSeat.playerId);
    const accumulator = accumulators.get(role);
    if (accumulator === undefined) {
      continue;
    }
    accumulator.sampleCount += 1;
    accumulator.rewardTotal += calculateBiddingRoleReward(game.result, role);
  }

  return biddingRoles.map((role) => {
    const accumulator = accumulators.get(role)!;
    return {
      role,
      sampleCount: accumulator.sampleCount,
      averageReward: accumulator.sampleCount === 0
        ? null
        : accumulator.rewardTotal / accumulator.sampleCount
    };
  });
}

function biddingRoleForSeat(result: GameResult, playerId: PlayerId): BiddingRole {
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

function calculateBiddingRoleReward(result: GameResult, role: BiddingRole): number {
  const d = result.targetPointCards;
  const napoleonWon = result.winner === "napoleon-team";

  switch (role) {
    case "napoleon":
      return napoleonWon ? d : -3;
    case "adjutant":
      return napoleonWon ? d - 7 : 0;
    case "citizen":
      return napoleonWon ? 7 : 0;
    case "napoleon-adjutant":
      return napoleonWon ? 2 * d - 7 : -3;
  }
}

function targetDistributionRecord(values: ReadonlyMap<number, number>): Readonly<Record<string, number>> {
  return Object.fromEntries(
    [12, 13, 14, 15, 16, 17, 18, 19].map((target) => [
      String(target),
      values.get(target) ?? 0
    ])
  );
}

function suitDistributionRecord(values: ReadonlyMap<Suit, number>): Readonly<Record<Suit, number>> {
  return {
    spades: values.get("spades") ?? 0,
    hearts: values.get("hearts") ?? 0,
    diamonds: values.get("diamonds") ?? 0,
    clubs: values.get("clubs") ?? 0
  };
}

function incrementNumberMap(map: Map<number, number>, key: number): void {
  map.set(key, (map.get(key) ?? 0) + 1);
}

function incrementSuitMap(map: Map<Suit, number>, key: Suit): void {
  map.set(key, (map.get(key) ?? 0) + 1);
}

function createAdjutantSelectionMetrics(): AdjutantSelectionMetrics {
  return {
    decisionCount: 0,
    cardIds: new Map()
  };
}

function recordAdjutantSelection(metrics: AdjutantSelectionMetrics, cardId: string): void {
  metrics.decisionCount += 1;
  metrics.cardIds.set(cardId, (metrics.cardIds.get(cardId) ?? 0) + 1);
}

function summarizeAdjutantSelections(
  metrics: AdjutantSelectionMetrics
): AdjutantSelectionDistributionSummary {
  return {
    decisionCount: metrics.decisionCount,
    cardIds: Object.fromEntries([...metrics.cardIds.entries()].sort(([left], [right]) =>
      left.localeCompare(right)
    ))
  };
}

function createPolicyVsRuleBasedAgents(args: {
  policy: PolicyOnnxModel;
  playerIds: readonly PlayerId[];
  policyAgentName: string;
  ruleBasedAgentName: string;
}): readonly EvaluationAgentDefinition[] {
  return [
    {
      name: args.policyAgentName,
      createAgent: ({ rng }) => new PolicyOnnxAgent({
        policy: args.policy,
        rng,
        playerIds: args.playerIds
      })
    },
    ...args.playerIds.slice(1).map((): EvaluationAgentDefinition => ({
      name: args.ruleBasedAgentName,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    }))
  ];
}

function createPlayingPolicyRosterAgents(args: {
  candidatePolicy: PolicyOnnxModel;
  opponentRoster: readonly PlayingPolicyEvaluationOpponent[];
  playerIds: readonly PlayerId[];
  candidateAgentName: string;
}): readonly EvaluationAgentDefinition[] {
  return [
    {
      name: args.candidateAgentName,
      createAgent: ({ rng }) => new PolicyOnnxAgent({
        policy: args.candidatePolicy,
        rng,
        playerIds: args.playerIds
      })
    },
    ...args.opponentRoster.map((opponent): EvaluationAgentDefinition => {
      if (opponent.type === "rule-based") {
        return {
          name: opponent.agentName ?? "RuleBasedAgent",
          createAgent: ({ rng }) => new RuleBasedAgent(rng)
        };
      }

      return {
        name: opponent.agentName ?? opponent.artifact?.displayName ?? "FrozenPolicyOnnxAgent",
        createAgent: ({ rng }) => new PolicyOnnxAgent({
          policy: opponent.policy,
          rng,
          playerIds: args.playerIds
        })
      };
    })
  ];
}

function createStandardOpponentRoster(
  benchmarkId: Exclude<StandardPlayingPolicyBenchmarkId, "rule-based-x4">,
  rlV740: {
    artifact: PlayingPolicyArtifactReference;
    policy: PolicyOnnxModel;
  }
): readonly PlayingPolicyEvaluationOpponent[] {
  const rlOpponent = (): PlayingPolicyOnnxOpponent => ({
    type: "playing-onnx",
    policy: rlV740.policy,
    agentName: rlV740.artifact.displayName,
    artifact: rlV740.artifact
  });
  const ruleBasedOpponent = (): PlayingPolicyRuleBasedOpponent => ({
    type: "rule-based",
    agentName: "RuleBasedAgent"
  });

  switch (benchmarkId) {
    case "rl-v740-x4":
      return [rlOpponent(), rlOpponent(), rlOpponent(), rlOpponent()];
    case "rule-based-x2-rl-v740-x2":
      return [ruleBasedOpponent(), ruleBasedOpponent(), rlOpponent(), rlOpponent()];
  }
}

function toOpponentRosterEntry(
  opponent: PlayingPolicyEvaluationOpponent,
  sourceAgentIndex: number
): PlayingPolicyOpponentRosterEntry {
  if (opponent.type === "rule-based") {
    return {
      sourceAgentIndex,
      type: opponent.type,
      agentName: opponent.agentName ?? "RuleBasedAgent"
    };
  }

  return {
    sourceAgentIndex,
    type: opponent.type,
    agentName: opponent.agentName ?? opponent.artifact?.displayName ?? "FrozenPolicyOnnxAgent",
    artifact: opponent.artifact,
    runtime: opponent.policy.runtime
  };
}

function createDefaultRosterAgentOrders(
  opponentRoster: readonly PlayingPolicyEvaluationOpponent[]
): readonly (readonly number[])[] {
  const baseOpponentSourceIndices = [1, 2, 3, 4] as const;
  const signatures = opponentRoster.map((opponent, index) =>
    opponentSignature(opponent, index + 1)
  );

  if (new Set(signatures).size === 1) {
    return [[0, ...baseOpponentSourceIndices]];
  }

  return baseOpponentSourceIndices.map((_, offset) => [
    0,
    ...baseOpponentSourceIndices.slice(offset),
    ...baseOpponentSourceIndices.slice(0, offset)
  ]);
}

function opponentSignature(
  opponent: PlayingPolicyEvaluationOpponent,
  sourceAgentIndex: number
): string {
  if (opponent.type === "rule-based") {
    return "rule-based";
  }

  return `playing-onnx:${opponent.artifact?.id ?? opponent.agentName ?? `source-${sourceAgentIndex}`}`;
}

function validateOpponentRoster(roster: readonly PlayingPolicyEvaluationOpponent[]): void {
  if (roster.length !== 4) {
    throw new Error(`opponentRoster must contain exactly 4 entries, got ${roster.length}.`);
  }

  for (const [index, opponent] of roster.entries()) {
    if (opponent.type === "playing-onnx") {
      assertPlayingPolicyType(`opponentRoster[${index}].policy`, opponent.policy);
    }
  }
}

async function validateOpponentRosterArtifacts(
  roster: readonly PlayingPolicyEvaluationOpponent[]
): Promise<void> {
  await Promise.all(roster.map(async (opponent) => {
    if (opponent.type === "playing-onnx" && opponent.artifact !== undefined) {
      await validatePlayingPolicyArtifactReference(opponent.artifact);
    }
  }));
}

function assertCandidateFirstAgentOrders(agentOrders: readonly (readonly number[])[]): void {
  for (const [index, agentOrder] of agentOrders.entries()) {
    if (agentOrder[0] !== 0) {
      throw new Error(`agentOrders[${index}] must keep candidate source agent index 0 first.`);
    }
  }
}

function createFullPolicyVsRuleBasedAgents(args: {
  playingPolicy: PolicyOnnxModel;
  biddingPolicy: NonPlayingPolicyOnnxModel;
  adjutantPolicy: NonPlayingPolicyOnnxModel;
  exchangePolicy: NonPlayingPolicyOnnxModel;
  playerIds: readonly PlayerId[];
  policyAgentName: string;
  ruleBasedAgentName: string;
  decisionMetrics: PolicyOnnxAgentDecisionMetrics;
  adjutantSelectionMetrics: AdjutantSelectionMetrics;
}): readonly EvaluationAgentDefinition[] {
  return [
    {
      name: args.policyAgentName,
      createAgent: ({ rng }) => new RecordingFullPolicyOnnxAgent({
        delegate: new PolicyOnnxAgent({
          policy: args.playingPolicy,
          biddingPolicy: args.biddingPolicy,
          adjutantPolicy: args.adjutantPolicy,
          exchangePolicy: args.exchangePolicy,
          rng,
          playerIds: args.playerIds,
          decisionMetrics: args.decisionMetrics
        }),
        adjutantSelectionMetrics: args.adjutantSelectionMetrics
      })
    },
    ...args.playerIds.slice(1).map((): EvaluationAgentDefinition => ({
      name: args.ruleBasedAgentName,
      createAgent: ({ rng }) => new RuleBasedAgent(rng)
    }))
  ];
}

class RecordingFullPolicyOnnxAgent implements Agent {
  constructor(
    private readonly options: {
      delegate: PolicyOnnxAgent;
      adjutantSelectionMetrics: AdjutantSelectionMetrics;
    }
  ) {}

  async selectAction(observation: PlayerObservation): Promise<GameAction> {
    return this.selectActionWithContext(observation);
  }

  async selectActionWithContext(
    observation: PlayerObservation,
    context?: { actualState: ActualCardState; playerIds: readonly PlayerId[] }
  ): Promise<GameAction> {
    const action = await this.options.delegate.selectActionWithContext(observation, context);
    if (observation.view.phase === "choosing-adjutant" && action.type === "choose-adjutant") {
      recordAdjutantSelection(this.options.adjutantSelectionMetrics, action.cardId);
    }
    return action;
  }
}

function createPolicyVsRuleBasedComparison(
  run: EvaluationRunRecord,
  names: {
    policyAgentName: string;
    ruleBasedAgentName: string;
  }
): PolicyVsRuleBasedComparisonReport {
  const groups: Record<AgentGroup, AgentGroupStats> = {
    policy: createAgentGroupStats(),
    "rule-based": createAgentGroupStats()
  };
  const failedGames = run.games
    .filter((game) => game.status === "failed")
    .map((game) => ({
      gameIndex: game.gameIndex,
      seed: game.seed,
      rotationOffset: game.rotationOffset,
      failureReason: game.failureReason
    }));

  for (const game of [...run.games].sort(compareGames)) {
    for (const seat of [...game.seats].sort(compareSeats)) {
      const group = seat.sourceAgentIndex === 0 ? groups.policy : groups["rule-based"];
      group.sourceAgentIndices.add(seat.sourceAgentIndex);
      countGame(group.stats, game, seat);

      countGame(getOrCreate(group.seatStats, seat.seatIndex), game, seat);

      if (isCompletedRole(seat.role)) {
        countGame(getOrCreate(group.roleStats, seat.role), game, seat);
      }
    }
  }

  const policy = toAgentGroupSummary({
    agentGroup: "policy",
    agentName: names.policyAgentName,
    group: groups.policy,
    baseline: groups["rule-based"]
  });
  const ruleBased = toAgentGroupSummary({
    agentGroup: "rule-based",
    agentName: names.ruleBasedAgentName,
    group: groups["rule-based"],
    baseline: groups.policy
  });

  return {
    schemaVersion: 1,
    illegalActionCount: failedGames.filter((game) =>
      isIllegalActionFailureReason(game.failureReason)
    ).length,
    failedGames,
    policy,
    ruleBased
  };
}

function toAgentGroupSummary(args: {
  agentGroup: AgentGroup;
  agentName: string;
  group: AgentGroupStats;
  baseline: AgentGroupStats;
}): PolicyVsRuleBasedAgentSummary {
  const summary = toPerformanceSummary(args.group.stats);
  const baselineSummary = toPerformanceSummary(args.baseline.stats);

  return {
    agentGroup: args.agentGroup,
    agentName: args.agentName,
    sourceAgentIndices: [...args.group.sourceAgentIndices].sort((left, right) => left - right),
    ...summary,
    roleResults: completedRoles.map((role) =>
      toRoleSummary(role, args.group.roleStats.get(role) ?? createStats())
    ),
    seatResults: [...args.group.seatStats.entries()]
      .sort(([left], [right]) => left - right)
      .map(([seatIndex, stats]) => toSeatSummary(seatIndex, stats)),
    comparison: createComparison(
      summary,
      args.group.stats,
      baselineSummary,
      args.baseline.stats
    )
  };
}

function createAgentGroupStats(): AgentGroupStats {
  return {
    stats: createStats(),
    sourceAgentIndices: new Set(),
    roleStats: new Map(),
    seatStats: new Map()
  };
}

function countGame(
  stats: MutableStats,
  game: EvaluationGameRecord,
  seat: EvaluationSeatAssignment
): void {
  stats.games.total += 1;

  if (game.status === "failed") {
    stats.games.failed += 1;
    incrementReason(stats.failuresByReason, game.failureReason);
    return;
  }

  stats.games.completed += 1;
  stats.contractSuccesses += game.contractSucceeded ? 1 : 0;

  if (!isCompletedRole(seat.role)) {
    return;
  }

  const won = didSeatWin(game, seat.role);
  const pointCards = seat.role === "alliance"
    ? game.pointCards.alliance
    : game.pointCards.napoleonTeam;
  stats.wins += won ? 1 : 0;
  stats.losses += won ? 0 : 1;
  stats.pointCardTotal += pointCards;
  stats.pointCardSquareTotal += pointCards * pointCards;
}

function didSeatWin(
  game: Extract<EvaluationGameRecord, { status: "completed" }>,
  role: CompletedRole
): boolean {
  return role === "alliance"
    ? game.winner === "alliance"
    : game.winner === "napoleon-team";
}

function toPerformanceSummary(stats: MutableStats): EvaluationPerformanceSummary {
  return {
    games: toGameCountSummary(stats.games),
    sampleCount: stats.games.completed,
    wins: stats.wins,
    losses: stats.losses,
    winRate: toRate(stats.wins, stats.games.completed),
    contractSuccesses: stats.contractSuccesses,
    contractSuccessRate: toRate(stats.contractSuccesses, stats.games.completed),
    averagePointCards: stats.games.completed === 0
      ? null
      : stats.pointCardTotal / stats.games.completed,
    failures: toFailureSummary(stats.failuresByReason)
  };
}

function toSeatSummary(
  seatIndex: number,
  stats: MutableStats
): EvaluationSeatPerformanceSummary {
  return {
    seatIndex,
    ...toPerformanceSummary(stats)
  };
}

function toRoleSummary(
  role: CompletedRole,
  stats: MutableStats
): EvaluationRolePerformanceSummary {
  return {
    role,
    ...toPerformanceSummary(stats)
  };
}

function createComparison(
  summary: EvaluationPerformanceSummary,
  stats: MutableStats,
  baseline: EvaluationPerformanceSummary,
  baselineStats: MutableStats
): EvaluationComparisonSummary {
  return {
    winRateDelta: subtractNullable(summary.winRate.rate, baseline.winRate.rate),
    winRateDeltaConfidenceInterval: toProportionDeltaConfidenceInterval(
      summary.winRate,
      baseline.winRate
    ),
    contractSuccessRateDelta: subtractNullable(
      summary.contractSuccessRate.rate,
      baseline.contractSuccessRate.rate
    ),
    contractSuccessRateDeltaConfidenceInterval: toProportionDeltaConfidenceInterval(
      summary.contractSuccessRate,
      baseline.contractSuccessRate
    ),
    averagePointCardsDelta: subtractNullable(
      summary.averagePointCards,
      baseline.averagePointCards
    ),
    averagePointCardsDeltaConfidenceInterval: toMeanDeltaConfidenceInterval(
      summary,
      stats,
      baseline,
      baselineStats
    )
  };
}

function createStats(): MutableStats {
  return {
    games: {
      total: 0,
      completed: 0,
      failed: 0
    },
    wins: 0,
    losses: 0,
    contractSuccesses: 0,
    pointCardTotal: 0,
    pointCardSquareTotal: 0,
    failuresByReason: new Map()
  };
}

function toRate(numerator: number, denominator: number): EvaluationRateSummary {
  return {
    numerator,
    denominator,
    rate: denominator === 0 ? null : numerator / denominator,
    confidenceInterval: toWilsonConfidenceInterval(numerator, denominator)
  };
}

function toWilsonConfidenceInterval(
  numerator: number,
  denominator: number
): EvaluationConfidenceInterval {
  if (denominator === 0) {
    return emptyConfidenceInterval("wilson");
  }

  const proportion = numerator / denominator;
  const zSquared = z95 * z95;
  const scale = 1 + zSquared / denominator;
  const center = (proportion + zSquared / (2 * denominator)) / scale;
  const margin = (
    z95
    * Math.sqrt((proportion * (1 - proportion) + zSquared / (4 * denominator)) / denominator)
  ) / scale;

  return {
    level: confidenceLevel,
    method: "wilson",
    lower: clamp(center - margin, 0, 1),
    upper: clamp(center + margin, 0, 1)
  };
}

function toProportionDeltaConfidenceInterval(
  summary: EvaluationRateSummary,
  baseline: EvaluationRateSummary
): EvaluationConfidenceInterval {
  if (summary.rate === null || baseline.rate === null) {
    return emptyConfidenceInterval("newcombe-wilson");
  }

  const summaryInterval = summary.confidenceInterval;
  const baselineInterval = baseline.confidenceInterval;

  if (
    summaryInterval.lower === null
    || summaryInterval.upper === null
    || baselineInterval.lower === null
    || baselineInterval.upper === null
  ) {
    return emptyConfidenceInterval("newcombe-wilson");
  }

  const delta = summary.rate - baseline.rate;
  const lowerMargin = Math.sqrt(
    ((summary.rate - summaryInterval.lower) ** 2)
    + ((baselineInterval.upper - baseline.rate) ** 2)
  );
  const upperMargin = Math.sqrt(
    ((summaryInterval.upper - summary.rate) ** 2)
    + ((baseline.rate - baselineInterval.lower) ** 2)
  );

  return {
    level: confidenceLevel,
    method: "newcombe-wilson",
    lower: clamp(delta - lowerMargin, -1, 1),
    upper: clamp(delta + upperMargin, -1, 1)
  };
}

function toMeanDeltaConfidenceInterval(
  summary: EvaluationPerformanceSummary,
  stats: MutableStats,
  baseline: EvaluationPerformanceSummary,
  baselineStats: MutableStats
): EvaluationConfidenceInterval {
  if (summary.averagePointCards === null || baseline.averagePointCards === null) {
    return emptyConfidenceInterval("normal");
  }

  const delta = summary.averagePointCards - baseline.averagePointCards;
  const summaryVariance = pointCardMeanVariance(stats);
  const baselineVariance = pointCardMeanVariance(baselineStats);

  if (summaryVariance === null || baselineVariance === null) {
    return emptyConfidenceInterval("normal");
  }

  const standardError = Math.sqrt(
    summaryVariance / summary.sampleCount
    + baselineVariance / baseline.sampleCount
  );

  if (!Number.isFinite(standardError)) {
    return emptyConfidenceInterval("normal");
  }

  const margin = z95 * standardError;

  return {
    level: confidenceLevel,
    method: "normal",
    lower: delta - margin,
    upper: delta + margin
  };
}

function pointCardMeanVariance(stats: MutableStats): number | null {
  if (stats.games.completed < 2) {
    return null;
  }

  const mean = stats.pointCardTotal / stats.games.completed;
  const numerator = stats.pointCardSquareTotal - stats.games.completed * mean ** 2;

  return Math.max(0, numerator / (stats.games.completed - 1));
}

function emptyConfidenceInterval(
  method: EvaluationConfidenceInterval["method"]
): EvaluationConfidenceInterval {
  return {
    level: confidenceLevel,
    method,
    lower: null,
    upper: null
  };
}

function toGameCountSummary(games: MutableStats["games"]): EvaluationGameCountSummary {
  return {
    total: games.total,
    completed: games.completed,
    failed: games.failed
  };
}

function toFailureSummary(failuresByReason: ReadonlyMap<string, number>): EvaluationFailureSummary {
  return {
    total: [...failuresByReason.values()].reduce((sum, count) => sum + count, 0),
    byReason: Object.fromEntries(
      [...failuresByReason.entries()].sort(([left], [right]) => left.localeCompare(right))
    )
  };
}

function getOrCreate<TKey>(map: Map<TKey, MutableStats>, key: TKey): MutableStats {
  const existing = map.get(key);

  if (existing !== undefined) {
    return existing;
  }

  const stats = createStats();
  map.set(key, stats);
  return stats;
}

function incrementReason(failuresByReason: Map<string, number>, reason: string): void {
  failuresByReason.set(reason, (failuresByReason.get(reason) ?? 0) + 1);
}

function subtractNullable(left: number | null, right: number | null): number | null {
  return left === null || right === null ? null : left - right;
}

function isCompletedRole(role: EvaluationSeatRole): role is CompletedRole {
  return role !== "unknown";
}

function isIllegalActionFailureReason(reason: string): boolean {
  return reason.includes("Automated agent selected an illegal action.")
    || reason.includes("outside legal actions");
}

function assertFullPolicyTypes(options: RunFullPolicyVsRuleBasedEvaluationOptions): void {
  assertPlayingPolicyType("playingPolicy", options.playingPolicy);
  assertNonPlayingPolicyModelType("biddingPolicy", options.biddingPolicy, "bidding");
  assertNonPlayingPolicyModelType("adjutantPolicy", options.adjutantPolicy, "adjutant");
  assertNonPlayingPolicyModelType("exchangePolicy", options.exchangePolicy, "exchange");
}

function assertPlayingPolicyType(optionName: string, policy: PolicyOnnxModel): void {
  const metadata = policy.metadata as unknown as Record<string, unknown>;
  if (metadata.policyType !== undefined) {
    throw new Error(
      `${optionName} policy type mismatch: expected playing, got ${String(metadata.policyType)}.`
    );
  }
  if (metadata.artifactType !== undefined) {
    throw new Error(`${optionName} artifact type mismatch: expected playing policy metadata.`);
  }
  if (metadata.playingEncoderSchemaVersion === undefined) {
    throw new Error(`${optionName} policy type mismatch: expected playing metadata.`);
  }
}

function assertNonPlayingPolicyModelType(
  optionName: string,
  policy: NonPlayingPolicyOnnxModel,
  expectedPolicyType: NonPlayingPolicyType
): void {
  if (policy.policyType !== expectedPolicyType) {
    throw new Error(
      `${optionName} policy type mismatch: expected ${expectedPolicyType}, got ${policy.policyType}.`
    );
  }

  const metadata = policy.metadata;
  if (metadata.policyType !== expectedPolicyType) {
    throw new Error(
      `${optionName} metadata policy type mismatch: expected ${expectedPolicyType}, got ${metadata.policyType}.`
    );
  }

  const expectedArtifactType = getNonPlayingPolicyOnnxSpec(expectedPolicyType).artifactType;
  if (metadata.artifactType !== expectedArtifactType) {
    throw new Error(
      `${optionName} artifact type mismatch: expected ${expectedArtifactType}, got ${metadata.artifactType}.`
    );
  }
}

function copyDecisionMetrics(
  metrics: PolicyOnnxAgentDecisionMetrics
): PolicyOnnxAgentDecisionMetrics {
  return {
    biddingOnnxCallCount: metrics.biddingOnnxCallCount,
    adjutantOnnxCallCount: metrics.adjutantOnnxCallCount,
    exchangeOnnxCallCount: metrics.exchangeOnnxCallCount,
    playingOnnxCallCount: metrics.playingOnnxCallCount,
    ruleBasedFallbackDecisionCount: metrics.ruleBasedFallbackDecisionCount
  };
}

function compareGames(left: EvaluationGameRecord, right: EvaluationGameRecord): number {
  return left.gameIndex - right.gameIndex
    || left.seed - right.seed
    || left.rotationOffset - right.rotationOffset
    || left.status.localeCompare(right.status);
}

function compareSeats(
  left: EvaluationSeatAssignment,
  right: EvaluationSeatAssignment
): number {
  return left.seatIndex - right.seatIndex
    || left.agentName.localeCompare(right.agentName)
    || left.playerId.localeCompare(right.playerId);
}

function clamp(value: number, lower: number, upper: number): number {
  return Math.min(Math.max(value, lower), upper);
}
