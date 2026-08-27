import { createReadStream } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { evaluateHandForTrump } from "@napoleon/ai";
import { decodeBiddingAction } from "@napoleon/ai-observation";
import { createDeck } from "@napoleon/game-core";
import type { Card, Suit } from "@napoleon/game-core";
import {
  optionalValue,
  parseOptionMap,
  requireValue
} from "./cliArgs.js";

const suits = ["spades", "hearts", "diamonds", "clubs"] as const;
const targets = [13, 14, 15, 16, 17, 18, 19] as const;
const selectedColumns = ["spades", "hearts", "diamonds", "clubs", "PASS"] as const;
const scoreBins = [
  { id: "<200", min: Number.NEGATIVE_INFINITY, max: 200 },
  { id: "200-249", min: 200, max: 250 },
  { id: "250-299", min: 250, max: 300 },
  { id: "300-349", min: 300, max: 350 },
  { id: ">=350", min: 350, max: Number.POSITIVE_INFINITY }
] as const;

type SelectedColumn = typeof selectedColumns[number];
type TargetKey = "PASS" | `${typeof targets[number]}`;

export interface NonPlayingBiddingDiagnosticsOptions {
  rolloutDataset: string;
  evaluation?: string;
  label?: string;
  outputJson?: string;
}

export interface BiddingSampleLike {
  actingPlayerId: string;
  selectedActionIndex: number;
  modelInput: readonly number[];
}

export interface BiddingActionSummary {
  decisionCount: number;
  passCount: number;
  passRate: number | null;
  bidCount: number;
  bidRate: number | null;
  targets: Record<TargetKey, CountRate>;
  suits: Record<Suit, CountRate>;
  targetSuit: Record<`${number}-${Suit}`, CountRate>;
}

export interface StrongestSuitSummary {
  strongestCounts: Record<Suit, number>;
  passRateByStrongest: Record<Suit, number | null>;
  matchCount: number;
  matchRate: number | null;
  selectedByStrongestCounts: Record<Suit, Record<SelectedColumn, number>>;
  selectedByStrongestRates: Record<Suit, Record<SelectedColumn, number | null>>;
  scoreBins: Record<Suit, Record<string, Record<TargetKey, CountRate>>>;
}

export interface GameResultSummary {
  gameCount: number;
  completedGames: number;
  failedGames: number;
  allPassCount: number;
  allPassRate: number | null;
  candidateNapoleonCount: number;
  candidateNapoleonRate: number | null;
  candidateAdjutantCount: number;
  candidateAdjutantRate: number | null;
  candidateCitizenCount: number;
  candidateCitizenRate: number | null;
  candidateWinCount: number;
  candidateWinRate: number | null;
  napoleonContractSuccessCount: number;
  napoleonContractSuccessRate: number | null;
  napoleonMeanTarget: number | null;
  napoleonMeanPointCards: number | null;
}

export interface CountRate {
  count: number;
  rate: number | null;
}

export interface BiddingDiagnosticsReport {
  schemaVersion: 1;
  label: string;
  rollout: {
    dataset: string;
    manifest: unknown;
  };
  evaluation?: {
    path: string;
    configuration: unknown;
    comparison?: unknown;
  };
  bidding: BiddingActionSummary;
  strongestSuit: StrongestSuitSummary;
  gameResult?: GameResultSummary;
}

export interface BiddingSampleSummary {
  bidding: BiddingActionSummary;
  strongestSuit: StrongestSuitSummary;
}

interface MutableBiddingSummary {
  decisionCount: number;
  passCount: number;
  bidCount: number;
  targetCounts: Record<TargetKey, number>;
  suitCounts: Record<Suit, number>;
  targetSuitCounts: Record<`${number}-${Suit}`, number>;
  strongestCounts: Record<Suit, number>;
  passByStrongest: Record<Suit, number>;
  matchCount: number;
  selectedByStrongest: Record<Suit, Record<SelectedColumn, number>>;
  scoreBins: Record<Suit, Record<string, Record<TargetKey, number>>>;
}

const cardById = new Map(createDeck().map((card) => [card.id, card]));
const cardIds = createDeck().map((card) => card.id);

const optionNames = new Set([
  "--rollout-dataset",
  "--evaluation",
  "--label",
  "--output-json"
]);

export async function runNonPlayingBiddingDiagnosticsCli(
  argv: readonly string[],
  io: {
    stdout: { write: (chunk: string) => void };
    stderr: { write: (chunk: string) => void };
  }
): Promise<number> {
  try {
    const options = parseDiagnosticsArgs(argv);
    const report = await createBiddingDiagnosticsReport(options);
    const markdown = formatBiddingDiagnosticsMarkdown(report);
    io.stdout.write(`${markdown}\n`);
    if (options.outputJson !== undefined) {
      await writeFile(options.outputJson, `${JSON.stringify(report, null, 2)}\n`, "utf8");
    }
    return 0;
  } catch (error) {
    io.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    return 1;
  }
}

export function parseDiagnosticsArgs(argv: readonly string[]): NonPlayingBiddingDiagnosticsOptions {
  const values = parseOptionMap(argv, optionNames);
  return {
    rolloutDataset: requireValue(values, "--rollout-dataset"),
    evaluation: optionalValue(values, "--evaluation"),
    label: optionalValue(values, "--label"),
    outputJson: optionalValue(values, "--output-json")
  };
}

export async function createBiddingDiagnosticsReport(
  options: NonPlayingBiddingDiagnosticsOptions
): Promise<BiddingDiagnosticsReport> {
  const manifestPath = join(options.rolloutDataset, "manifest.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as {
    shards?: readonly { file: string }[];
  };
  const accumulator = createMutableBiddingSummary();

  for (const shard of manifest.shards ?? []) {
    await readSamples(join(options.rolloutDataset, shard.file), (sample) => {
      recordBiddingSample(accumulator, sample);
    });
  }

  const evaluation = options.evaluation === undefined
    ? undefined
    : JSON.parse(await readFile(options.evaluation, "utf8"));

  return {
    schemaVersion: 1,
    label: options.label ?? "candidate",
    rollout: {
      dataset: options.rolloutDataset,
      manifest
    },
    ...(options.evaluation === undefined
      ? {}
      : {
          evaluation: {
            path: options.evaluation,
            configuration: evaluation.configuration,
            comparison: evaluation.comparison
          }
        }),
    bidding: finalizeBiddingSummary(accumulator),
    strongestSuit: finalizeStrongestSuitSummary(accumulator),
    ...(evaluation === undefined ? {} : { gameResult: summarizeEvaluationGames(evaluation) })
  };
}

export function recordBiddingSample(
  accumulator: MutableBiddingSummary,
  sample: BiddingSampleLike
): void {
  const action = decodeBiddingAction(sample.selectedActionIndex, sample.actingPlayerId);
  const hand = handFromModelInput(sample.modelInput);
  const strongest = selectStrongestSuit(hand);
  const selected: SelectedColumn = action.type === "pass" ? "PASS" : action.suit;
  const targetKey: TargetKey = action.type === "pass" ? "PASS" : `${action.targetPointCards}` as TargetKey;

  accumulator.decisionCount += 1;
  accumulator.targetCounts[targetKey] += 1;
  accumulator.strongestCounts[strongest] += 1;
  accumulator.selectedByStrongest[strongest][selected] += 1;

  if (action.type === "pass") {
    accumulator.passCount += 1;
    accumulator.passByStrongest[strongest] += 1;
  } else {
    accumulator.bidCount += 1;
    accumulator.suitCounts[action.suit] += 1;
    accumulator.targetSuitCounts[`${action.targetPointCards}-${action.suit}`] += 1;
    accumulator.matchCount += action.suit === strongest ? 1 : 0;
  }

  const scoreBin = scoreBinFor(evaluateHandForTrump(hand, strongest));
  accumulator.scoreBins[strongest][scoreBin][targetKey] += 1;
}

export function summarizeBiddingSamples(
  samples: readonly BiddingSampleLike[]
): BiddingSampleSummary {
  const accumulator = createMutableBiddingSummary();
  for (const sample of samples) {
    recordBiddingSample(accumulator, sample);
  }
  return {
    bidding: finalizeBiddingSummary(accumulator),
    strongestSuit: finalizeStrongestSuitSummary(accumulator)
  };
}

export function selectStrongestSuit(hand: readonly Card[]): Suit {
  return suits.reduce((bestSuit, candidateSuit) => {
    const bestScore = evaluateHandForTrump(hand, bestSuit);
    const candidateScore = evaluateHandForTrump(hand, candidateSuit);
    return candidateScore > bestScore ? candidateSuit : bestSuit;
  }, suits[0]);
}

export function summarizeEvaluationGames(evaluation: {
  configuration?: { policyAgentName?: string };
  run: { games: readonly EvaluationGameLike[] };
}): GameResultSummary {
  const policyAgentName = evaluation.configuration?.policyAgentName ?? "FullPolicyOnnxAgent";
  const games = evaluation.run.games;
  let completedGames = 0;
  let failedGames = 0;
  let allPassCount = 0;
  let candidateNapoleonCount = 0;
  let candidateAdjutantCount = 0;
  let candidateCitizenCount = 0;
  let candidateWinCount = 0;
  let napoleonContractSuccessCount = 0;
  let napoleonTargetTotal = 0;
  let napoleonPointCardTotal = 0;

  for (const game of games) {
    if (game.status !== "completed") {
      failedGames += 1;
      continue;
    }

    completedGames += 1;
    if (game.resultType === "all-pass") {
      allPassCount += 1;
      continue;
    }

    const candidateSeat = game.seats.find((seat) => seat.agentName === policyAgentName);
    if (candidateSeat === undefined) {
      continue;
    }

    const candidateWon = candidateSeat.role === "alliance"
      ? game.winner === "alliance"
      : game.winner === "napoleon-team";
    candidateWinCount += candidateWon ? 1 : 0;

    if (candidateSeat.role === "napoleon") {
      candidateNapoleonCount += 1;
      napoleonContractSuccessCount += game.contractSucceeded === true ? 1 : 0;
      napoleonTargetTotal += game.contract?.targetPointCards ?? 0;
      napoleonPointCardTotal += game.pointCards?.napoleonTeam ?? 0;
    } else if (candidateSeat.role === "adjutant") {
      candidateAdjutantCount += 1;
    } else if (candidateSeat.role === "alliance") {
      candidateCitizenCount += 1;
    }
  }

  const standardCandidateGames = candidateNapoleonCount + candidateAdjutantCount + candidateCitizenCount;
  return {
    gameCount: games.length,
    completedGames,
    failedGames,
    allPassCount,
    allPassRate: rate(allPassCount, completedGames),
    candidateNapoleonCount,
    candidateNapoleonRate: rate(candidateNapoleonCount, standardCandidateGames),
    candidateAdjutantCount,
    candidateAdjutantRate: rate(candidateAdjutantCount, standardCandidateGames),
    candidateCitizenCount,
    candidateCitizenRate: rate(candidateCitizenCount, standardCandidateGames),
    candidateWinCount,
    candidateWinRate: rate(candidateWinCount, standardCandidateGames),
    napoleonContractSuccessCount,
    napoleonContractSuccessRate: rate(napoleonContractSuccessCount, candidateNapoleonCount),
    napoleonMeanTarget: mean(napoleonTargetTotal, candidateNapoleonCount),
    napoleonMeanPointCards: mean(napoleonPointCardTotal, candidateNapoleonCount)
  };
}

export function formatBiddingDiagnosticsMarkdown(report: BiddingDiagnosticsReport): string {
  const lines: string[] = [];
  lines.push(`## ${report.label}`);
  lines.push("");
  lines.push(`- rollout dataset: \`${report.rollout.dataset}\``);
  if (report.evaluation !== undefined) {
    lines.push(`- evaluation: \`${report.evaluation.path}\``);
  }
  lines.push(`- bidding decisions: ${report.bidding.decisionCount}`);
  lines.push(`- pass rate: ${formatRate(report.bidding.passRate)}`);
  lines.push(`- bid rate: ${formatRate(report.bidding.bidRate)}`);
  lines.push(`- strongest suit match rate among bids: ${formatRate(report.strongestSuit.matchRate)}`);
  const rolloutBidding = getRolloutBiddingDiagnostics(report.rollout.manifest);
  if (rolloutBidding !== undefined) {
    lines.push(`- rollout candidate Napoleon formation rate: ${formatRate(rolloutBidding.candidateNapoleonFormationRate)}`);
    lines.push(`- rollout declaration success rate: ${formatRate(rolloutBidding.declarationSuccessRate)}`);
  }
  lines.push("");
  lines.push("### Target Distribution");
  lines.push("");
  lines.push("| action | count | rate |");
  lines.push("|---|---:|---:|");
  for (const target of ["PASS", ...targets.map(String)] as TargetKey[]) {
    const entry = report.bidding.targets[target];
    lines.push(`| ${target} | ${entry.count} | ${formatRate(entry.rate)} |`);
  }
  lines.push("");
  lines.push("### Suit Distribution");
  lines.push("");
  lines.push("| suit | count | rate |");
  lines.push("|---|---:|---:|");
  for (const suit of suits) {
    const entry = report.bidding.suits[suit];
    lines.push(`| ${shortSuit(suit)} | ${entry.count} | ${formatRate(entry.rate)} |`);
  }
  lines.push("");
  lines.push("### Target x Suit Distribution");
  lines.push("");
  lines.push("| target | S | H | D | C |");
  lines.push("|---|---:|---:|---:|---:|");
  for (const target of targets) {
    const row = suits.map((suit) => {
      const entry = report.bidding.targetSuit[`${target}-${suit}`];
      return `${entry.count} (${formatRate(entry.rate)})`;
    });
    lines.push(`| ${target} | ${row.join(" | ")} |`);
  }
  lines.push("");
  lines.push("### Strongest Suit Summary");
  lines.push("");
  lines.push("| strongest | count | pass rate | same-suit bid rate |");
  lines.push("|---|---:|---:|---:|");
  for (const suit of suits) {
    const strongestCount = report.strongestSuit.strongestCounts[suit];
    const sameSuitRate = report.strongestSuit.selectedByStrongestRates[suit][suit];
    lines.push(`| ${shortSuit(suit)} | ${strongestCount} | ${formatRate(report.strongestSuit.passRateByStrongest[suit])} | ${formatRate(sameSuitRate)} |`);
  }
  lines.push("");
  lines.push("### Strongest Suit x Selected Action Counts");
  lines.push("");
  lines.push("| strongest | S | H | D | C | PASS |");
  lines.push("|---|---:|---:|---:|---:|---:|");
  for (const strongest of suits) {
    const row = report.strongestSuit.selectedByStrongestCounts[strongest];
    lines.push(`| ${shortSuit(strongest)} | ${row.spades} | ${row.hearts} | ${row.diamonds} | ${row.clubs} | ${row.PASS} |`);
  }
  lines.push("");
  lines.push("### Strongest Suit x Selected Action Rates");
  lines.push("");
  lines.push("| strongest | S | H | D | C | PASS |");
  lines.push("|---|---:|---:|---:|---:|---:|");
  for (const strongest of suits) {
    const row = report.strongestSuit.selectedByStrongestRates[strongest];
    lines.push(`| ${shortSuit(strongest)} | ${formatRate(row.spades)} | ${formatRate(row.hearts)} | ${formatRate(row.diamonds)} | ${formatRate(row.clubs)} | ${formatRate(row.PASS)} |`);
  }
  if (report.gameResult !== undefined) {
    const game = report.gameResult;
    lines.push("");
    lines.push("### Game Result");
    lines.push("");
    lines.push(`- completed games: ${game.completedGames}/${game.gameCount}`);
    lines.push(`- all-pass rate: ${formatRate(game.allPassRate)}`);
    lines.push(`- candidate Napoleon rate: ${formatRate(game.candidateNapoleonRate)}`);
    lines.push(`- candidate Adjutant rate: ${formatRate(game.candidateAdjutantRate)}`);
    lines.push(`- candidate Citizen rate: ${formatRate(game.candidateCitizenRate)}`);
    lines.push(`- Napoleon contract success: ${formatRate(game.napoleonContractSuccessRate)}`);
    lines.push(`- Napoleon mean target: ${formatNumber(game.napoleonMeanTarget)}`);
    lines.push(`- Napoleon mean point cards: ${formatNumber(game.napoleonMeanPointCards)}`);
    lines.push(`- candidate win rate: ${formatRate(game.candidateWinRate)}`);
  }
  if (rolloutBidding !== undefined) {
    const actualGameCount = getNumber(report.rollout.manifest, ["actualGameCount"]);
    lines.push("");
    lines.push("### Rollout Game/Formation Diagnostics");
    lines.push("");
    lines.push(`- all-pass immediate end rate: ${formatRate(actualGameCount === undefined ? null : rate(rolloutBidding.allPassImmediateEndCount, actualGameCount))}`);
    lines.push(`- candidate Napoleon formation rate: ${formatRate(rolloutBidding.candidateNapoleonFormationRate)}`);
    lines.push(`- declaration success rate: ${formatRate(rolloutBidding.declarationSuccessRate)}`);
    lines.push(`- candidate role counts: ${formatRoleDistribution(rolloutBidding.candidateRoleDistribution)}`);
  }
  const comparison = getRuleBasedComparison(report.evaluation?.comparison);
  if (comparison !== undefined) {
    lines.push("");
    lines.push("### RuleBased Comparison");
    lines.push("");
    lines.push("| agent | win rate | contract success | average point cards |");
    lines.push("|---|---:|---:|---:|");
    lines.push(`| policy | ${formatRate(comparison.policy.winRate)} | ${formatRate(comparison.policy.contractSuccessRate)} | ${formatNumber(comparison.policy.averagePointCards)} |`);
    lines.push(`| ruleBased | ${formatRate(comparison.ruleBased.winRate)} | ${formatRate(comparison.ruleBased.contractSuccessRate)} | ${formatNumber(comparison.ruleBased.averagePointCards)} |`);
    lines.push(`| delta | ${formatSignedRate(comparison.winRateDelta)} | ${formatSignedRate(comparison.contractSuccessRateDelta)} | ${formatSignedNumber(comparison.averagePointCardsDelta)} |`);
  }
  return lines.join("\n");
}

async function readSamples(
  path: string,
  onSample: (sample: BiddingSampleLike) => void
): Promise<void> {
  const stream = createReadStream(path, { encoding: "utf8" });
  const lines = createInterface({ input: stream, crlfDelay: Infinity });
  for await (const line of lines) {
    if (line.trim().length > 0) {
      onSample(JSON.parse(line) as BiddingSampleLike);
    }
  }
}

function createMutableBiddingSummary(): MutableBiddingSummary {
  const targetCounts = Object.fromEntries(
    (["PASS", ...targets.map(String)] as TargetKey[]).map((target) => [target, 0])
  ) as Record<TargetKey, number>;
  const suitCounts = Object.fromEntries(suits.map((suit) => [suit, 0])) as Record<Suit, number>;
  const targetSuitCounts = Object.fromEntries(
    targets.flatMap((target) => suits.map((suit) => [`${target}-${suit}`, 0]))
  ) as Record<`${number}-${Suit}`, number>;
  const selectedByStrongest = Object.fromEntries(
    suits.map((suit) => [
      suit,
      Object.fromEntries(selectedColumns.map((column) => [column, 0]))
    ])
  ) as Record<Suit, Record<SelectedColumn, number>>;
  const strongestCounts = Object.fromEntries(suits.map((suit) => [suit, 0])) as Record<Suit, number>;
  const passByStrongest = Object.fromEntries(suits.map((suit) => [suit, 0])) as Record<Suit, number>;
  const scoreBinTargets = () => Object.fromEntries(
    (["PASS", ...targets.map(String)] as TargetKey[]).map((target) => [target, 0])
  ) as Record<TargetKey, number>;
  const scoreBinsBySuit = Object.fromEntries(
    suits.map((suit) => [
      suit,
      Object.fromEntries(scoreBins.map((bin) => [bin.id, scoreBinTargets()]))
    ])
  ) as Record<Suit, Record<string, Record<TargetKey, number>>>;
  return {
    decisionCount: 0,
    passCount: 0,
    bidCount: 0,
    targetCounts,
    suitCounts,
    targetSuitCounts,
    strongestCounts,
    passByStrongest,
    matchCount: 0,
    selectedByStrongest,
    scoreBins: scoreBinsBySuit
  };
}

function finalizeBiddingSummary(accumulator: MutableBiddingSummary): BiddingActionSummary {
  return {
    decisionCount: accumulator.decisionCount,
    passCount: accumulator.passCount,
    passRate: rate(accumulator.passCount, accumulator.decisionCount),
    bidCount: accumulator.bidCount,
    bidRate: rate(accumulator.bidCount, accumulator.decisionCount),
    targets: mapCounts(accumulator.targetCounts, accumulator.decisionCount),
    suits: mapCounts(accumulator.suitCounts, accumulator.bidCount),
    targetSuit: mapCounts(accumulator.targetSuitCounts, accumulator.bidCount)
  };
}

function finalizeStrongestSuitSummary(accumulator: MutableBiddingSummary): StrongestSuitSummary {
  return {
    strongestCounts: accumulator.strongestCounts,
    passRateByStrongest: Object.fromEntries(
      suits.map((suit) => [suit, rate(accumulator.passByStrongest[suit], accumulator.strongestCounts[suit])])
    ) as Record<Suit, number | null>,
    matchCount: accumulator.matchCount,
    matchRate: rate(accumulator.matchCount, accumulator.bidCount),
    selectedByStrongestCounts: accumulator.selectedByStrongest,
    selectedByStrongestRates: Object.fromEntries(
      suits.map((suit) => [
        suit,
        Object.fromEntries(
          selectedColumns.map((column) => [
            column,
            rate(accumulator.selectedByStrongest[suit][column], accumulator.strongestCounts[suit])
          ])
        )
      ])
    ) as Record<Suit, Record<SelectedColumn, number | null>>,
    scoreBins: Object.fromEntries(
      suits.map((suit) => [
        suit,
        Object.fromEntries(
          Object.entries(accumulator.scoreBins[suit]).map(([bin, counts]) => [
            bin,
            mapCounts(counts, Object.values(counts).reduce((sum, count) => sum + count, 0))
          ])
        )
      ])
    ) as Record<Suit, Record<string, Record<TargetKey, CountRate>>>
  };
}

function handFromModelInput(modelInput: readonly number[]): readonly Card[] {
  return modelInput.slice(0, cardIds.length).flatMap((value, index) => {
    if (value !== 1) {
      return [];
    }
    const card = cardById.get(cardIds[index]);
    if (card === undefined) {
      throw new Error(`No card for modelInput index ${index}.`);
    }
    return [card];
  });
}

function scoreBinFor(score: number): string {
  const bin = scoreBins.find((candidate) => score >= candidate.min && score < candidate.max);
  if (bin === undefined) {
    throw new Error(`No score bin for ${score}.`);
  }
  return bin.id;
}

function mapCounts<K extends string>(
  counts: Record<K, number>,
  denominator: number
): Record<K, CountRate> {
  return Object.fromEntries(
    Object.entries(counts).map(([key, count]) => [
      key,
      { count, rate: rate(count as number, denominator) }
    ])
  ) as Record<K, CountRate>;
}

function rate(numerator: number, denominator: number): number | null {
  return denominator === 0 ? null : numerator / denominator;
}

function mean(sum: number, count: number): number | null {
  return count === 0 ? null : sum / count;
}

function shortSuit(suit: Suit): string {
  return suit[0].toUpperCase();
}

function formatRate(value: number | null): string {
  return value === null ? "n/a" : `${(value * 100).toFixed(2)}%`;
}

function formatNumber(value: number | null): string {
  return value === null ? "n/a" : value.toFixed(3);
}

function formatSignedRate(value: number | null): string {
  return value === null ? "n/a" : `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}pp`;
}

function formatSignedNumber(value: number | null): string {
  return value === null ? "n/a" : `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}

interface RolloutBiddingDiagnostics {
  candidateNapoleonFormationRate: number | null;
  declarationSuccessRate: number | null;
  allPassImmediateEndCount: number;
  candidateRoleDistribution?: Record<string, number>;
}

function getRolloutBiddingDiagnostics(manifest: unknown): RolloutBiddingDiagnostics | undefined {
  const bidding = getPath(manifest, ["diagnostics", "bidding"]);
  if (!isRecord(bidding)) {
    return undefined;
  }

  const allPassImmediateEndCount = typeof bidding.allPassImmediateEndCount === "number"
    ? bidding.allPassImmediateEndCount
    : 0;
  return {
    candidateNapoleonFormationRate: nullableNumber(bidding.candidateNapoleonFormationRate),
    declarationSuccessRate: nullableNumber(bidding.declarationSuccessRate),
    allPassImmediateEndCount,
    candidateRoleDistribution: isNumberRecord(bidding.candidateRoleDistribution)
      ? bidding.candidateRoleDistribution
      : undefined
  };
}

function formatRoleDistribution(value: Record<string, number> | undefined): string {
  if (value === undefined) {
    return "n/a";
  }
  return Object.entries(value)
    .map(([role, count]) => `${role}=${count}`)
    .join(", ");
}

interface RuleBasedComparisonSummary {
  policy: {
    winRate: number | null;
    contractSuccessRate: number | null;
    averagePointCards: number | null;
  };
  ruleBased: {
    winRate: number | null;
    contractSuccessRate: number | null;
    averagePointCards: number | null;
  };
  winRateDelta: number | null;
  contractSuccessRateDelta: number | null;
  averagePointCardsDelta: number | null;
}

function getRuleBasedComparison(comparison: unknown): RuleBasedComparisonSummary | undefined {
  if (!isRecord(comparison) || !isRecord(comparison.policy) || !isRecord(comparison.ruleBased)) {
    return undefined;
  }

  return {
    policy: {
      winRate: nullableNumber(getPath(comparison.policy, ["winRate", "rate"])),
      contractSuccessRate: nullableNumber(getPath(comparison.policy, ["contractSuccessRate", "rate"])),
      averagePointCards: nullableNumber(comparison.policy.averagePointCards)
    },
    ruleBased: {
      winRate: nullableNumber(getPath(comparison.ruleBased, ["winRate", "rate"])),
      contractSuccessRate: nullableNumber(getPath(comparison.ruleBased, ["contractSuccessRate", "rate"])),
      averagePointCards: nullableNumber(comparison.ruleBased.averagePointCards)
    },
    winRateDelta: nullableNumber(getPath(comparison.policy, ["comparison", "winRateDelta"])),
    contractSuccessRateDelta: nullableNumber(getPath(comparison.policy, ["comparison", "contractSuccessRateDelta"])),
    averagePointCardsDelta: nullableNumber(getPath(comparison.policy, ["comparison", "averagePointCardsDelta"]))
  };
}

function getNumber(value: unknown, path: readonly string[]): number | undefined {
  const result = getPath(value, path);
  return typeof result === "number" ? result : undefined;
}

function getPath(value: unknown, path: readonly string[]): unknown {
  return path.reduce<unknown>((current, key) => {
    if (!isRecord(current)) {
      return undefined;
    }
    return current[key];
  }, value);
}

function nullableNumber(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isNumberRecord(value: unknown): value is Record<string, number> {
  return isRecord(value) && Object.values(value).every((entry) => typeof entry === "number");
}

interface EvaluationGameLike {
  status: "completed" | "failed";
  resultType?: "standard" | "all-pass";
  seats: readonly {
    agentName: string;
    role: "napoleon" | "adjutant" | "alliance" | "unknown";
  }[];
  winner?: "napoleon-team" | "alliance" | null;
  contractSucceeded?: boolean | null;
  contract?: { targetPointCards: number } | null;
  pointCards?: { napoleonTeam: number; alliance: number } | null;
}
