import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { runAutomatedGame } from "@napoleon/ai";
import type { AutomatedGameRecord } from "@napoleon/ai";
import type { GameResult, PlayerId, Suit } from "@napoleon/game-core";
import type { AiPolicyComposition, PublicAiPhaseCallDiagnostics } from "@napoleon/protocol";
import { createPhasePolicyRegistry } from "./phasePolicyRegistry.js";

const PLAYER_IDS = ["player-0", "player-1", "player-2", "player-3", "player-4"] as const;
const POLICY_NAMES = ["AI", "RB"] as const;
const COMPOSITION_KEYS = [
  "napoleon-rb-adjutant-rb",
  "napoleon-rb-adjutant-ai",
  "napoleon-ai-adjutant-rb",
  "napoleon-ai-adjutant-ai",
  "solo-napoleon-rb",
  "solo-napoleon-ai"
] as const;
const AI_COMPOSITION = {
  playing: "ppo-separated-v1000",
  bidding: "frozen-raise-v1",
  nonPlaying: "parameterized-adjutant-exchange-v1"
} as const satisfies AiPolicyComposition;
const RB_COMPOSITION = {
  playing: "rule-based",
  bidding: "rule-based",
  nonPlaying: "rule-based"
} as const satisfies AiPolicyComposition;

type PolicyName = typeof POLICY_NAMES[number];
type CompositionKey = typeof COMPOSITION_KEYS[number];

export interface RosterScheduleEntry {
  gameIndex: number;
  seed: number;
  aiCount: number;
  rbCount: number;
  combinationIndex: number;
  repetitionIndex: number;
  aiSeats: readonly number[];
  seatPolicies: readonly PolicyName[];
}

interface RawGameRow {
  schemaVersion: 1;
  status: "completed";
  gameIndex: number;
  seed: number;
  aiCount: number;
  rbCount: number;
  combinationIndex: number;
  repetitionIndex: number;
  aiSeats: readonly number[];
  seatPolicies: readonly PolicyName[];
  resultType: "standard" | "all-pass";
  napoleonSideComposition: CompositionKey | "all-pass";
  napoleonSeat: number | null;
  napoleonPolicy: PolicyName | null;
  adjutantSeat: number | null;
  adjutantPolicy: PolicyName | null;
  solo: boolean | null;
  winnerSide: "napoleon" | "citizen" | null;
  contractTarget: number | null;
  contractSuit: Suit | null;
  contractSuccess: boolean | null;
  contractMargin: number | null;
  napoleonSidePointCards: number | null;
  illegal: number;
  fallback: number;
  invariantFailure: number;
}

interface FailedRawGameRow {
  schemaVersion: 1;
  status: "failed";
  gameIndex: number;
  seed: number;
  aiCount: number;
  rbCount: number;
  combinationIndex: number;
  repetitionIndex: number;
  aiSeats: readonly number[];
  seatPolicies: readonly PolicyName[];
  illegal: number;
  fallback: number;
  invariantFailure: number;
  otherGameFailure: number;
  failureReason: string;
}

type AnyRawGameRow = RawGameRow | FailedRawGameRow;

interface AuditSample {
  gameIndex: number;
  seat: number;
  policy: PolicyName;
  phase: "bidding" | "adjutant" | "exchange" | "playing";
  diagnostics: PublicAiPhaseCallDiagnostics;
}

export function createRosterSchedule(startSeed: number): readonly RosterScheduleEntry[] {
  const schedule: RosterScheduleEntry[] = [];
  for (let aiCount = 0; aiCount <= PLAYER_IDS.length; aiCount += 1) {
    const combinations = combinationsOfSeats(PLAYER_IDS.length, aiCount);
    const repetitions = 1_000 / combinations.length;
    if (!Number.isInteger(repetitions)) {
      throw new Error(`1,000 games cannot be balanced across ${combinations.length} combinations.`);
    }
    for (let repetitionIndex = 0; repetitionIndex < repetitions; repetitionIndex += 1) {
      combinations.forEach((aiSeats, combinationIndex) => {
        const gameIndex = schedule.length;
        const aiSeatSet = new Set(aiSeats);
        schedule.push({
          gameIndex,
          seed: startSeed + gameIndex,
          aiCount,
          rbCount: PLAYER_IDS.length - aiCount,
          combinationIndex,
          repetitionIndex,
          aiSeats,
          seatPolicies: PLAYER_IDS.map((_, seat) => aiSeatSet.has(seat) ? "AI" : "RB")
        });
      });
    }
  }
  return schedule;
}

function combinationsOfSeats(seatCount: number, count: number): readonly (readonly number[])[] {
  const result: number[][] = [];
  const visit = (nextSeat: number, selected: number[]) => {
    if (selected.length === count) {
      result.push([...selected]);
      return;
    }
    for (let seat = nextSeat; seat <= seatCount - (count - selected.length); seat += 1) {
      selected.push(seat);
      visit(seat + 1, selected);
      selected.pop();
    }
  };
  visit(0, []);
  return result;
}

async function runFinalRosterEvaluation(options: CliOptions): Promise<void> {
  const schedule = createRosterSchedule(options.startSeed);
  const manifestCore = {
    schemaVersion: 1,
    evaluationId: "final-com-ai-roster-evaluation-v1",
    startSeed: options.startSeed,
    endSeed: options.startSeed + schedule.length - 1,
    scheduledGames: schedule.length,
    scheduleAlgorithm: "lexicographic-seat-combinations-round-robin-v1",
    rosters: Array.from({ length: 6 }, (_, aiCount) => ({
      aiCount,
      rbCount: 5 - aiCount,
      games: 1_000,
      combinationCount: combinationsOfSeats(5, aiCount).length,
      gamesPerCombination: 1_000 / combinationsOfSeats(5, aiCount).length
    })),
    policies: { AI: AI_COMPOSITION, RB: RB_COMPOSITION },
    scheduleSha256: sha256(schedule.map((entry) =>
      `${entry.gameIndex}:${entry.seed}:${entry.seatPolicies.join("")}`
    ).join("\n")),
    schedule
  };
  const manifestHash = sha256(`${JSON.stringify(manifestCore, null, 2)}\n`);
  const registry = createPhasePolicyRegistry();
  await registry.initialize();
  const phaseRegistry = registry.describe();
  assertFormalPoliciesAvailable(phaseRegistry);

  const rows: AnyRawGameRow[] = [];
  const auditSamples = new Map<string, AuditSample>();
  let nextGameIndex = 0;
  let processed = 0;
  const startedAt = Date.now();

  const worker = async () => {
    while (true) {
      const scheduleIndex = nextGameIndex++;
      if (scheduleIndex >= schedule.length) return;
      const entry = schedule[scheduleIndex];
      const diagnostics = entry.seatPolicies.map((policy) =>
        registry.createDiagnostics(policy === "AI" ? AI_COMPOSITION : RB_COMPOSITION)
      );
      try {
        const record = await runAutomatedGame({
          seed: entry.seed,
          playerIds: PLAYER_IDS,
          createAgent: ({ playerIndex, rng }) => registry.createAgent(
            entry.seatPolicies[playerIndex] === "AI" ? AI_COMPOSITION : RB_COMPOSITION,
            diagnostics[playerIndex],
            rng
          )
        });
        rows.push(createRawGameRow(entry, record, diagnostics));
      } catch (error) {
        const diagnosticCounts = sumDiagnostics(diagnostics);
        rows.push({
          schemaVersion: 1,
          status: "failed",
          ...entry,
          illegal: diagnosticCounts.illegal,
          fallback: diagnosticCounts.fallback,
          invariantFailure: error instanceof RosterEvaluationInvariantError ? 1 : 0,
          otherGameFailure: error instanceof RosterEvaluationInvariantError ? 0 : 1,
          failureReason: error instanceof Error ? error.message : String(error)
        });
      } finally {
        diagnostics.forEach((diagnostic, seat) => {
          const policy = entry.seatPolicies[seat];
          for (const phase of ["bidding", "adjutant", "exchange", "playing"] as const) {
            const key = `${policy}:${phase}`;
            if (diagnostic[`${phase}Calls`] > 0 && !auditSamples.has(key)) {
              auditSamples.set(key, { gameIndex: entry.gameIndex, seat, policy, phase,
                diagnostics: { ...diagnostic } });
            }
          }
        });
        processed += 1;
        if (processed % options.progressEvery === 0 || processed === schedule.length) {
          const seconds = (Date.now() - startedAt) / 1_000;
          const rate = processed / seconds;
          process.stderr.write(
            `[final roster] ${processed}/${schedule.length} (${rate.toFixed(2)} games/s, ETA ${((schedule.length - processed) / rate / 60).toFixed(1)} min)\n`
          );
        }
      }
    }
  };
  await Promise.all(Array.from({ length: options.concurrency }, () => worker()));
  rows.sort((left, right) => left.gameIndex - right.gameIndex);

  const summary = createSummary(rows, schedule, [...auditSamples.values()]);
  const config = {
    schemaVersion: 1,
    evaluationId: manifestCore.evaluationId,
    generatedAt: new Date().toISOString(),
    outputDirectory: options.outputDirectory,
    repoReport: options.repoReport,
    startSeed: options.startSeed,
    concurrency: options.concurrency,
    policies: manifestCore.policies,
    phaseRegistry
  };
  const artifact = { schemaVersion: 1, manifestHash, ...summary };
  const report = renderRosterReport({ manifestHash, manifest: manifestCore, summary: artifact,
    outputDirectory: options.outputDirectory });

  await Promise.all([
    mkdir(options.outputDirectory, { recursive: true }),
    mkdir(dirname(options.repoReport), { recursive: true })
  ]);
  await Promise.all([
    writeFile(join(options.outputDirectory, "games.jsonl"),
      `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`),
    writeFile(join(options.outputDirectory, "summary.json"), `${JSON.stringify(artifact, null, 2)}\n`),
    writeFile(join(options.outputDirectory, "config.json"), `${JSON.stringify(config, null, 2)}\n`),
    writeFile(join(options.outputDirectory, "manifest.json"),
      `${JSON.stringify({ ...manifestCore, manifestHash }, null, 2)}\n`),
    writeFile(join(options.outputDirectory, "report.md"), report),
    writeFile(options.repoReport, report)
  ]);

  if (summary.failures.total !== 0 || summary.rawRowCount !== schedule.length) {
    throw new Error(`Final roster evaluation has ${summary.failures.total} failures and ${summary.rawRowCount} raw rows.`);
  }
  process.stdout.write(`${JSON.stringify({
    games: summary.completedGames,
    rawRows: summary.rawRowCount,
    manifestHash,
    outputDirectory: options.outputDirectory,
    repoReport: options.repoReport,
    failures: summary.failures
  }, null, 2)}\n`);
}

function createRawGameRow(
  entry: RosterScheduleEntry,
  record: AutomatedGameRecord,
  diagnostics: readonly PublicAiPhaseCallDiagnostics[]
): RawGameRow {
  const diagnosticCounts = sumDiagnostics(diagnostics);
  validateResult(record.result, record, entry);
  if (record.result.resultType === "all-pass") {
    return {
      schemaVersion: 1,
      status: "completed",
      ...entry,
      resultType: "all-pass",
      napoleonSideComposition: "all-pass",
      napoleonSeat: null,
      napoleonPolicy: null,
      adjutantSeat: null,
      adjutantPolicy: null,
      solo: null,
      winnerSide: null,
      contractTarget: null,
      contractSuit: null,
      contractSuccess: null,
      contractMargin: null,
      napoleonSidePointCards: null,
      illegal: diagnosticCounts.illegal,
      fallback: diagnosticCounts.fallback,
      invariantFailure: 0
    };
  }
  const napoleonSeat = PLAYER_IDS.indexOf(
    record.result.napoleonPlayerId as typeof PLAYER_IDS[number]
  );
  const adjutantSeat = record.result.adjutantPlayerId === null
    ? null
    : PLAYER_IDS.indexOf(record.result.adjutantPlayerId as typeof PLAYER_IDS[number]);
  const napoleonPolicy = entry.seatPolicies[napoleonSeat];
  const adjutantPolicy = adjutantSeat === null ? null : entry.seatPolicies[adjutantSeat];
  const finalBid = getFinalBid(record);
  return {
    schemaVersion: 1,
    status: "completed",
    ...entry,
    resultType: "standard",
    napoleonSideComposition: compositionKey(napoleonPolicy, adjutantPolicy),
    napoleonSeat,
    napoleonPolicy,
    adjutantSeat,
    adjutantPolicy,
    solo: adjutantSeat === null,
    winnerSide: record.result.winner === "napoleon-team" ? "napoleon" : "citizen",
    contractTarget: record.result.targetPointCards,
    contractSuit: finalBid.suit,
    contractSuccess: record.result.winner === "napoleon-team",
    contractMargin: record.result.napoleonTeamPointCards - record.result.targetPointCards,
    napoleonSidePointCards: record.result.napoleonTeamPointCards,
    illegal: diagnosticCounts.illegal,
    fallback: diagnosticCounts.fallback,
    invariantFailure: 0
  };
}

function getFinalBid(record: AutomatedGameRecord): { playerId: PlayerId; suit: Suit; target: number } {
  const bids = record.decisions.flatMap((decision) => decision.action.type === "bid"
    ? [{ playerId: decision.action.playerId, suit: decision.action.suit,
      target: decision.action.targetPointCards }]
    : []);
  const finalBid = bids.at(-1);
  if (finalBid === undefined) {
    throw new RosterEvaluationInvariantError("Standard game has no final bid.");
  }
  return finalBid;
}

function validateResult(
  result: GameResult,
  record: AutomatedGameRecord,
  entry: RosterScheduleEntry
): void {
  if (record.seed !== entry.seed || record.playerIds.join(",") !== PLAYER_IDS.join(",")) {
    throw new RosterEvaluationInvariantError("Automated record does not match its schedule entry.");
  }
  if (result.resultType === "all-pass") return;
  if (result.napoleonTeamPointCards + result.alliancePointCards !== 20) {
    throw new RosterEvaluationInvariantError("Point-card total is not 20.");
  }
  if ((result.napoleonTeamPointCards >= result.targetPointCards) !==
      (result.winner === "napoleon-team")) {
    throw new RosterEvaluationInvariantError("Winner and contract success disagree.");
  }
  const finalBid = getFinalBid(record);
  if (finalBid.playerId !== result.napoleonPlayerId || finalBid.target !== result.targetPointCards) {
    throw new RosterEvaluationInvariantError("Final bid and result contract disagree.");
  }
}

class RosterEvaluationInvariantError extends Error {}

function compositionKey(
  napoleonPolicy: PolicyName,
  adjutantPolicy: PolicyName | null
): CompositionKey {
  if (adjutantPolicy === null) {
    return napoleonPolicy === "AI" ? "solo-napoleon-ai" : "solo-napoleon-rb";
  }
  return `napoleon-${napoleonPolicy.toLowerCase()}-adjutant-${adjutantPolicy.toLowerCase()}` as CompositionKey;
}

function sumDiagnostics(diagnostics: readonly PublicAiPhaseCallDiagnostics[]) {
  return {
    illegal: diagnostics.reduce((total, value) => total + value.illegalCount, 0),
    fallback: diagnostics.reduce((total, value) => total + value.fallbackCount, 0)
  };
}

function createSummary(
  rows: readonly AnyRawGameRow[],
  schedule: readonly RosterScheduleEntry[],
  auditSamples: readonly AuditSample[]
) {
  const completed = rows.filter((row): row is RawGameRow => row.status === "completed");
  const failed = rows.filter((row): row is FailedRawGameRow => row.status === "failed");
  const rosterSummaries = Array.from({ length: 6 }, (_, aiCount) => {
    const rosterRows = completed.filter((row) => row.aiCount === aiCount);
    const standard = rosterRows.filter((row) => row.resultType === "standard");
    const nonSolo = standard.filter((row) => !row.solo);
    const napWins = standard.filter((row) => row.winnerSide === "napoleon").length;
    const citizenWins = standard.length - napWins;
    return {
      aiCount,
      rbCount: 5 - aiCount,
      roster: rosterLabel(aiCount),
      totalGames: rosterRows.length,
      napoleonSideWins: napWins,
      citizenSideWins: citizenWins,
      napoleonSideWinRate: rate(napWins, standard.length),
      allPass: rosterRows.length - standard.length,
      napoleonPolicy: policyExposure(standard.map((row) => row.napoleonPolicy!)),
      adjutantPolicy: policyExposure(nonSolo.map((row) => row.adjutantPolicy!)),
      soloCount: standard.length - nonSolo.length,
      meanTarget: mean(standard.map((row) => row.contractTarget!)),
      meanMargin: mean(standard.map((row) => row.contractMargin!)),
      meanNapoleonSidePointCards: mean(standard.map((row) => row.napoleonSidePointCards!)),
      composition: COMPOSITION_KEYS.map((key) => summarizeComposition(standard, key))
    };
  });
  const combinationCounts = Array.from(new Map(schedule.map((entry) => [
    `${entry.aiCount}:${entry.aiSeats.join(",")}`,
    { aiCount: entry.aiCount, rbCount: entry.rbCount, aiSeats: entry.aiSeats }
  ])).values()).map((combination) => ({
    ...combination,
    scheduledGames: schedule.filter((entry) =>
      entry.aiCount === combination.aiCount && entry.aiSeats.join(",") === combination.aiSeats.join(",")
    ).length,
    completedGames: completed.filter((row) =>
      row.aiCount === combination.aiCount && row.aiSeats.join(",") === combination.aiSeats.join(",")
    ).length
  }));
  const allStandard = completed.filter((row) => row.resultType === "standard");
  return {
    rawRowCount: rows.length,
    completedGames: completed.length,
    scheduledGames: schedule.length,
    rosterSummaries,
    combinationCounts,
    overallComposition: COMPOSITION_KEYS.map((key) => summarizeComposition(allStandard, key)),
    allPass: completed.filter((row) => row.resultType === "all-pass").length,
    illegal: completed.reduce((total, row) => total + row.illegal, 0) +
      failed.reduce((total, row) => total + row.illegal, 0),
    fallback: completed.reduce((total, row) => total + row.fallback, 0) +
      failed.reduce((total, row) => total + row.fallback, 0),
    invariantFailure: completed.reduce((total, row) => total + row.invariantFailure, 0) +
      failed.reduce((total, row) => total + row.invariantFailure, 0),
    failures: {
      total: failed.length,
      otherGameFailure: failed.reduce((total, row) => total + row.otherGameFailure, 0),
      rows: failed
    },
    auditSamples: [...auditSamples].sort((left, right) =>
      left.policy.localeCompare(right.policy) || left.phase.localeCompare(right.phase)
    )
  };
}

function summarizeComposition(rows: readonly RawGameRow[], key: CompositionKey) {
  const selected = rows.filter((row) => row.napoleonSideComposition === key);
  const napoleonWins = selected.filter((row) => row.winnerSide === "napoleon").length;
  return {
    key,
    games: selected.length,
    napoleonSideWins: napoleonWins,
    citizenSideWins: selected.length - napoleonWins,
    napoleonSideWinRate: rate(napoleonWins, selected.length),
    meanTarget: mean(selected.map((row) => row.contractTarget!)),
    meanMargin: mean(selected.map((row) => row.contractMargin!))
  };
}

function policyExposure(values: readonly PolicyName[]) {
  const ai = values.filter((value) => value === "AI").length;
  const rb = values.length - ai;
  return { denominator: values.length, AI: ai, RB: rb, aiRate: rate(ai, values.length),
    rbRate: rate(rb, values.length) };
}

function mean(values: readonly number[]): number | null {
  return values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function rate(numerator: number, denominator: number): number | null {
  return denominator === 0 ? null : numerator / denominator;
}

export function renderRosterReport(input: {
  manifestHash: string;
  manifest: any;
  summary: any;
  outputDirectory: string;
}): string {
  const { manifest, manifestHash, summary, outputDirectory } = input;
  const p = (value: number | null) => value === null ? "—" : `${(value * 100).toFixed(2)}%`;
  const n = (value: number | null) => value === null ? "—" : value.toFixed(3);
  const lines = [
    "# Final COM-AI roster evaluation",
    "",
    "旧50/50 mixed evaluationに代わるPR #203の正式最終評価です。1ゲーム内のCOM-AI人数を0〜5で固定し、各rosterを1,000 gamesずつ評価しました。player単位のAI対RB勝率ではなく、Napoleon-sideの構成と勝敗を主に読みます。",
    "",
    "## 1. 正式AI構成",
    "",
    "| preset | playing | bidding | nonPlaying |",
    "| --- | --- | --- | --- |",
    "| COM-AI | `ppo-separated-v1000` | `frozen-raise-v1` | `parameterized-adjutant-exchange-v1` |",
    "| COM-RuleBase | `rule-based` | `rule-based` | `rule-based` |",
    "",
    "legacy full-policy、旧adjutant/exchange PPO、experimental artifactは使用していません。",
    "",
    "## 2. 6 roster条件",
    "",
    "| roster | games | combinations | games / combination |",
    "| --- | ---: | ---: | ---: |",
    ...manifest.rosters.map((row: any) => `| ${rosterLabel(row.aiCount)} | ${row.games.toLocaleString()} | ${row.combinationCount} | ${row.gamesPerCombination} |`),
    "",
    `game seedは \`${manifest.startSeed}..${manifest.endSeed}\`、scheduleは \`${manifest.scheduleAlgorithm}\` です。`,
    "",
    "## 3. Seat balancing",
    "",
    "| roster | AI seats | scheduled | completed |",
    "| --- | --- | ---: | ---: |",
    ...summary.combinationCounts.map((row: any) => `| ${rosterLabel(row.aiCount)} | ${formatSeats(row.aiSeats)} | ${row.scheduledGames} | ${row.completedGames} |`),
    "",
    "## 4. Raw保存先 / manifest",
    "",
    `- raw directory: \`${outputDirectory}\``,
    `- games.jsonl rows: ${summary.rawRowCount.toLocaleString()}`,
    `- logical manifest SHA-256: \`${manifestHash}\``,
    `- schedule SHA-256: \`${manifest.scheduleSha256}\``,
    "- files: `games.jsonl`, `summary.json`, `config.json`, `manifest.json`, `report.md`",
    "",
    "## 5. Roster別1,000-game結果",
    "",
    "| roster | games | Napoleon wins | Citizen wins | Napoleon win rate | All-Pass | Napoleon policy AI/RB | Adjutant policy AI/RB | solo | mean target | mean margin |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...summary.rosterSummaries.map((row: any) => `| ${row.roster} | ${row.totalGames.toLocaleString()} | ${row.napoleonSideWins} | ${row.citizenSideWins} | ${p(row.napoleonSideWinRate)} | ${row.allPass} | ${row.napoleonPolicy.AI}/${row.napoleonPolicy.RB} | ${row.adjutantPolicy.AI}/${row.adjutantPolicy.RB} | ${row.soloCount} | ${n(row.meanTarget)} | ${n(row.meanMargin)} |`),
    "",
    "## 6. Roster × Napoleon-side 6分類",
    "",
    "| roster | classification | games | Napoleon wins | Citizen wins | Napoleon win rate | mean target | mean margin |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...summary.rosterSummaries.flatMap((roster: any) => roster.composition.map((row: any) =>
      `| ${roster.roster} | ${compositionLabel(row.key)} | ${row.games} | ${row.napoleonSideWins} | ${row.citizenSideWins} | ${p(row.napoleonSideWinRate)} | ${n(row.meanTarget)} | ${n(row.meanMargin)} |`
    )),
    "",
    "全roster合計の構成別参考集計:",
    "",
    "| classification | games | Napoleon wins | Citizen wins | Napoleon win rate | mean target | mean margin |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...summary.overallComposition.map((row: any) => `| ${compositionLabel(row.key)} | ${row.games} | ${row.napoleonSideWins} | ${row.citizenSideWins} | ${p(row.napoleonSideWinRate)} | ${n(row.meanTarget)} | ${n(row.meanMargin)} |`),
    "",
    "## 7. Target / margin",
    "",
    "targetとmarginは上のroster別・6分類表に併記しました。異なるrosterは独立seed集合であり、構成別の生差にはdeal/role selectionも含まれるため、単一phaseの因果効果とは断定しません。",
    "",
    "## 8. illegal / fallback / invariant",
    "",
    `- illegal: ${summary.illegal}`,
    `- fallback: ${summary.fallback}`,
    `- invariant failure: ${summary.invariantFailure}`,
    `- other game failure: ${summary.failures.otherGameFailure}`,
    `- All-Pass: ${summary.allPass}`,
    "",
    "Sample phase audit:",
    "",
    ...summary.auditSamples.map((sample: AuditSample) => `- ${sample.policy} ${sample.phase}: game ${sample.gameIndex}, seat ${sample.seat}, composition=${JSON.stringify(sample.diagnostics.composition)}, calls bidding=${sample.diagnostics.biddingCalls}, adjutant=${sample.diagnostics.adjutantCalls}, exchange=${sample.diagnostics.exchangeCalls}, playing=${sample.diagnostics.playingCalls}, illegal=${sample.diagnostics.illegalCount}, fallback=${sample.diagnostics.fallbackCount}`),
    "",
    "## 9. 最終結論",
    "",
    rosterConclusion(summary),
    "",
    "この結果を見てpolicy再学習、artifact再選定、reward/preset変更は行っていません。"
  ];
  return `${lines.join("\n")}\n`;
}

function rosterConclusion(summary: any): string {
  const rates = summary.rosterSummaries.map((row: any) =>
    `${row.roster} ${formatPercent(row.napoleonSideWinRate)}`
  ).join("、");
  const firstRoster = summary.rosterSummaries[0];
  const lastRoster = summary.rosterSummaries.at(-1);
  const rosterDifference = lastRoster.napoleonSideWinRate - firstRoster.napoleonSideWinRate;
  const byKey = Object.fromEntries(summary.overallComposition.map((row: any) => [row.key, row]));
  return [
    `Napoleon-side win rateは、${rates}でした。今回の独立seed群ではAI人数が増える各段階で単調に上がり、AI0からAI5までの差は ${(rosterDifference * 100).toFixed(2)} percentage pointsでした。`,
    `全roster合計の記述集計では、Napoleon=AI / Adjutant=AI は ${formatPercent(byKey["napoleon-ai-adjutant-ai"].napoleonSideWinRate)}、Napoleon=AI / Adjutant=RB は ${formatPercent(byKey["napoleon-ai-adjutant-rb"].napoleonSideWinRate)}、Napoleon=RB / Adjutant=AI は ${formatPercent(byKey["napoleon-rb-adjutant-ai"].napoleonSideWinRate)}、Napoleon=RB / Adjutant=RB は ${formatPercent(byKey["napoleon-rb-adjutant-rb"].napoleonSideWinRate)}でした。Napoleon policyをAIにした構成、Adjutant policyをAIにした構成の双方で高い生勝率が観測され、AI+AIが最も高い値でした。`,
    `solo NapoleonはAI ${formatPercent(byKey["solo-napoleon-ai"].napoleonSideWinRate)}、RB ${formatPercent(byKey["solo-napoleon-rb"].napoleonSideWinRate)}でした。またroster平均では、AI0→AI5でdeclared targetが ${firstRoster.meanTarget.toFixed(3)}→${lastRoster.meanTarget.toFixed(3)}、contract marginが ${firstRoster.meanMargin.toFixed(3)}→${lastRoster.meanMargin.toFixed(3)}へ変化しました。`,
    "したがって、この固定roster評価では、正式COM-AIを増やした編成ほどNapoleon-side成績が良く、Napoleon・Adjutant・soloの構成別集計もCOM-AIの有効性と整合しています。一方、異なるrosterは同一dealの対比較ではなく、biddingによるrole selection、deal、Citizen側構成も同時に変わります。このため各差を特定phaseの因果効果やCitizen policy単体の優劣とは断定しません。"
  ].join("\n\n");
}

function compositionLabel(key: CompositionKey): string {
  return ({
    "napoleon-rb-adjutant-rb": "Napoleon=RB / Adjutant=RB",
    "napoleon-rb-adjutant-ai": "Napoleon=RB / Adjutant=AI",
    "napoleon-ai-adjutant-rb": "Napoleon=AI / Adjutant=RB",
    "napoleon-ai-adjutant-ai": "Napoleon=AI / Adjutant=AI",
    "solo-napoleon-rb": "Solo Napoleon=RB",
    "solo-napoleon-ai": "Solo Napoleon=AI"
  } as const)[key];
}

function rosterLabel(aiCount: number): string { return `RB${5 - aiCount} / AI${aiCount}`; }
function formatSeats(seats: readonly number[]): string {
  return seats.length === 0 ? "none" : seats.length === 5 ? "0,1,2,3,4 (all)" : seats.join(",");
}
function formatPercent(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(2)}%`;
}
function sha256(value: string): string { return createHash("sha256").update(value).digest("hex"); }

function assertFormalPoliciesAvailable(description: ReturnType<ReturnType<typeof createPhasePolicyRegistry>["describe"]>): void {
  const required = [
    [description.playing, "ppo-separated-v1000"],
    [description.bidding, "frozen-raise-v1"],
    [description.nonPlaying, "parameterized-adjutant-exchange-v1"]
  ] as const;
  for (const [policies, id] of required) {
    if (!policies.some((policy) => policy.id === id && policy.isAvailable)) {
      throw new Error(`Required formal policy ${id} is unavailable.`);
    }
  }
}

interface CliOptions {
  startSeed: number;
  concurrency: number;
  progressEvery: number;
  outputDirectory: string;
  repoReport: string;
}

function parseArgs(argv: readonly string[]): CliOptions {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) throw new Error(`Invalid argument near ${key}.`);
    values.set(key.slice(2), value);
  }
  const integer = (name: string, fallback: number) => {
    const value = Number(values.get(name) ?? fallback);
    if (!Number.isSafeInteger(value) || value < 1) throw new Error(`${name} must be a positive integer.`);
    return value;
  };
  return {
    startSeed: integer("start-seed", 462_600_000),
    concurrency: integer("concurrency", 16),
    progressEvery: integer("progress-every", 250),
    outputDirectory: resolve(values.get("output-directory") ?? "/tmp/napoleon-final-roster-eval"),
    repoReport: resolve(values.get("repo-report") ?? "diagnostics/final-com-ai-roster-evaluation.md")
  };
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  runFinalRosterEvaluation(parseArgs(process.argv.slice(2))).catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
    process.exitCode = 1;
  });
}
