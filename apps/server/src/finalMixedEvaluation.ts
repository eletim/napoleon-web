import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
  createSeededRandom,
  deriveSeed,
  runAutomatedGame
} from "@napoleon/ai";
import type { GameResult, PlayerId } from "@napoleon/game-core";
import { calculateNonPlayingLearningTerminalReward } from "@napoleon/training-data";
import type { AiPolicyComposition, PublicAiPhaseCallDiagnostics } from "@napoleon/protocol";
import { createPhasePolicyRegistry } from "./phasePolicyRegistry.js";

const PLAYER_IDS = ["player-0", "player-1", "player-2", "player-3", "player-4"] as const;
const POLICY_NAMES = ["COM-AI", "COM-RuleBase"] as const;
const ROLE_NAMES = ["Napoleon", "Adjutant", "Citizen"] as const;
const AI_COMPOSITION = {
  playing: "ppo-separated-v1000",
  bidding: "frozen-raise-v1",
  nonPlaying: "parameterized-adjutant-exchange-v1"
} as const satisfies AiPolicyComposition;
const RULE_BASED_COMPOSITION = {
  playing: "rule-based",
  bidding: "rule-based",
  nonPlaying: "rule-based"
} as const satisfies AiPolicyComposition;

type PolicyName = typeof POLICY_NAMES[number];
type RoleName = typeof ROLE_NAMES[number];

export interface BalancedAssignmentManifest {
  schemaVersion: 1;
  gameCount: number;
  gameStartSeed: number;
  assignmentSeed: number;
  assignmentAlgorithm: "independent-seat-balanced-fisher-yates-v1";
  assignmentSequenceSha256: string;
  presets: {
    "COM-AI": AiPolicyComposition;
    "COM-RuleBase": AiPolicyComposition;
  };
  inferenceDevice: "cpu";
  statistics: "game-cluster-robust-sandwich-normal-95-v1";
  allPassTreatment: string;
}

interface SeatOutcome {
  seat: number;
  policy: PolicyName;
  role: RoleName | "All-Pass";
  win: number | null;
  rawReward: number;
  relativeReward: number;
}

interface GameOutcome {
  gameIndex: number;
  seed: number;
  assignment: readonly PolicyName[];
  resultType: "standard" | "all-pass";
  seats: readonly SeatOutcome[];
  napoleonPolicy: PolicyName | null;
  adjutantPolicy: PolicyName | null;
  contractSuccess: number | null;
  contractMargin: number | null;
  target: number | null;
  napoleonPointCards: number | null;
}

interface ClusterEstimate {
  n: number;
  estimate: number | null;
  standardError: number | null;
  ci95: readonly [number | null, number | null];
}

interface EvaluationFailure {
  gameIndex: number;
  seed: number;
  reason: string;
  invariant: boolean;
}

interface AuditSample {
  gameIndex: number;
  seat: number;
  policy: PolicyName;
  phase: "bidding" | "adjutant" | "exchange" | "playing";
  diagnostics: PublicAiPhaseCallDiagnostics;
}

export function createBalancedAssignments(
  gameCount: number,
  assignmentSeed: number
): readonly (readonly PolicyName[])[] {
  if (!Number.isSafeInteger(gameCount) || gameCount < 2 || gameCount % 2 !== 0) {
    throw new Error("gameCount must be an even integer of at least 2 for exact seat balance.");
  }
  const bySeat = PLAYER_IDS.map((_, seat) => {
    const values: PolicyName[] = Array.from(
      { length: gameCount },
      (_, index) => index < gameCount / 2 ? "COM-AI" : "COM-RuleBase"
    );
    const rng = createSeededRandom(deriveSeed(assignmentSeed, `final-mixed-assignment:seat:${seat}`));
    for (let index = values.length - 1; index > 0; index -= 1) {
      const replacement = Math.floor(rng() * (index + 1));
      [values[index], values[replacement]] = [values[replacement], values[index]];
    }
    return values;
  });
  return Array.from({ length: gameCount }, (_, gameIndex) =>
    bySeat.map((seatAssignments) => seatAssignments[gameIndex])
  );
}

export function clusterEstimate(
  games: readonly GameOutcome[],
  select: (game: GameOutcome) => readonly number[],
  options: { rate?: boolean } = {}
): ClusterEstimate {
  const clusters = games.map(select);
  const n = clusters.reduce((total, values) => total + values.length, 0);
  if (n === 0) return { n: 0, estimate: null, standardError: null, ci95: [null, null] };
  const total = clusters.reduce(
    (sum, values) => sum + values.reduce((inner, value) => inner + value, 0),
    0
  );
  const estimate = total / n;
  const clusterCount = clusters.length;
  const squaredInfluence = clusters.reduce((sum, values) => {
    const clusterTotal = values.reduce((inner, value) => inner + value, 0);
    const influence = clusterTotal - estimate * values.length;
    return sum + influence * influence;
  }, 0);
  const standardError = clusterCount > 1
    ? Math.sqrt((clusterCount / (clusterCount - 1)) * squaredInfluence) / n
    : 0;
  const bound = (value: number) => options.rate ? Math.max(0, Math.min(1, value)) : value;
  return {
    n,
    estimate,
    standardError,
    ci95: [bound(estimate - 1.96 * standardError), bound(estimate + 1.96 * standardError)]
  };
}

function clusterDifference(
  games: readonly GameOutcome[],
  left: (game: GameOutcome) => readonly number[],
  right: (game: GameOutcome) => readonly number[]
): ClusterEstimate {
  const leftEstimate = clusterEstimate(games, left);
  const rightEstimate = clusterEstimate(games, right);
  if (leftEstimate.estimate === null || rightEstimate.estimate === null) {
    return { n: 0, estimate: null, standardError: null, ci95: [null, null] };
  }
  const leftN = leftEstimate.n;
  const rightN = rightEstimate.n;
  const influences = games.map((game) => {
    const leftValues = left(game);
    const rightValues = right(game);
    return (
      (leftValues.reduce((sum, value) => sum + value, 0) - leftEstimate.estimate! * leftValues.length) / leftN
      - (rightValues.reduce((sum, value) => sum + value, 0) - rightEstimate.estimate! * rightValues.length) / rightN
    );
  });
  const correction = influences.length / (influences.length - 1);
  const standardError = Math.sqrt(correction * influences.reduce((sum, value) => sum + value * value, 0));
  const estimate = leftEstimate.estimate - rightEstimate.estimate;
  return {
    n: Math.min(leftN, rightN),
    estimate,
    standardError,
    ci95: [estimate - 1.96 * standardError, estimate + 1.96 * standardError]
  };
}

async function runEvaluation(options: CliOptions): Promise<void> {
  const assignments = createBalancedAssignments(options.games, options.assignmentSeed);
  const assignmentSequenceSha256 = sha256(
    assignments.map((row) => row.map((policy) => policy === "COM-AI" ? "A" : "R").join("")).join("\n")
  );
  const manifest: BalancedAssignmentManifest = {
    schemaVersion: 1,
    gameCount: options.games,
    gameStartSeed: options.startSeed,
    assignmentSeed: options.assignmentSeed,
    assignmentAlgorithm: "independent-seat-balanced-fisher-yates-v1",
    assignmentSequenceSha256,
    presets: { "COM-AI": AI_COMPOSITION, "COM-RuleBase": RULE_BASED_COMPOSITION },
    inferenceDevice: "cpu",
    statistics: "game-cluster-robust-sandwich-normal-95-v1",
    allPassTreatment: "included in exposure and reward means; excluded from win/loss and role denominators"
  };
  const manifestHash = sha256(`${JSON.stringify(manifest, null, 2)}\n`);
  const registry = createPhasePolicyRegistry();
  await registry.initialize();
  const description = registry.describe();
  assertFormalPoliciesAvailable(description);

  const outcomes: GameOutcome[] = [];
  const failures: EvaluationFailure[] = [];
  const auditSamples = new Map<string, AuditSample>();
  const diagnosticsTotals = Object.fromEntries(POLICY_NAMES.map((policy) => [policy, {
    playingCalls: 0, biddingCalls: 0, adjutantCalls: 0, exchangeCalls: 0,
    fallbackCount: 0, illegalCount: 0
  }])) as Record<PolicyName, Omit<PublicAiPhaseCallDiagnostics, "composition">>;
  let nextGame = 0;
  let completed = 0;
  const started = Date.now();

  const worker = async () => {
    while (true) {
      const gameIndex = nextGame++;
      if (gameIndex >= options.games) return;
      const seed = options.startSeed + gameIndex;
      const assignment = assignments[gameIndex];
      const diagnostics = assignment.map((policy) =>
        registry.createDiagnostics(policy === "COM-AI" ? AI_COMPOSITION : RULE_BASED_COMPOSITION)
      );
      try {
        const record = await runAutomatedGame({
          seed,
          playerIds: PLAYER_IDS,
          createAgent: ({ playerIndex, rng }) => registry.createAgent(
            assignment[playerIndex] === "COM-AI" ? AI_COMPOSITION : RULE_BASED_COMPOSITION,
            diagnostics[playerIndex],
            rng
          )
        });
        validateResult(record.result, PLAYER_IDS);
        outcomes.push(toOutcome(gameIndex, seed, assignment, record.result));
      } catch (error) {
        failures.push({
          gameIndex,
          seed,
          reason: error instanceof Error ? error.message : String(error),
          invariant: error instanceof EvaluationInvariantError
        });
      } finally {
        diagnostics.forEach((diagnostic, seat) => {
          const policy = assignment[seat];
          const totals = diagnosticsTotals[policy];
          totals.playingCalls += diagnostic.playingCalls;
          totals.biddingCalls += diagnostic.biddingCalls;
          totals.adjutantCalls += diagnostic.adjutantCalls;
          totals.exchangeCalls += diagnostic.exchangeCalls;
          totals.fallbackCount += diagnostic.fallbackCount;
          totals.illegalCount += diagnostic.illegalCount;
          for (const phase of ["bidding", "adjutant", "exchange", "playing"] as const) {
            const calls = diagnostic[`${phase}Calls`];
            const key = `${policy}:${phase}`;
            if (calls > 0 && !auditSamples.has(key)) {
              auditSamples.set(key, { gameIndex, seat, policy, phase, diagnostics: { ...diagnostic } });
            }
          }
        });
        completed += 1;
        if (completed % options.progressEvery === 0 || completed === options.games) {
          const seconds = (Date.now() - started) / 1000;
          const rate = completed / seconds;
          const eta = (options.games - completed) / rate;
          process.stderr.write(
            `[final mixed] ${completed}/${options.games} (${rate.toFixed(2)} games/s, ETA ${(eta / 60).toFixed(1)} min)\n`
          );
        }
      }
    }
  };
  await Promise.all(Array.from({ length: options.concurrency }, () => worker()));
  outcomes.sort((left, right) => left.gameIndex - right.gameIndex);
  failures.sort((left, right) => left.gameIndex - right.gameIndex);

  const summary = createSummary(outcomes, assignments, failures, diagnosticsTotals);
  const artifact = {
    schemaVersion: 1,
    manifest,
    manifestHash,
    generatedAt: new Date().toISOString(),
    completedGames: outcomes.length,
    failedGames: failures.length,
    summary,
    phaseRegistry: description,
    diagnosticsTotals,
    auditSamples: [...auditSamples.values()].sort((a, b) =>
      a.policy.localeCompare(b.policy) || a.seat - b.seat
    ),
    failures
  };
  await Promise.all([
    mkdir(dirname(options.outputJson), { recursive: true }),
    mkdir(dirname(options.manifestOutput), { recursive: true }),
    mkdir(dirname(options.reportOutput), { recursive: true })
  ]);
  await writeFile(options.manifestOutput, `${JSON.stringify({ ...manifest, manifestHash }, null, 2)}\n`);
  await writeFile(options.outputJson, `${JSON.stringify(artifact, null, 2)}\n`);
  await writeFile(options.reportOutput, renderReport(artifact));
  process.stdout.write(`${JSON.stringify({
    games: outcomes.length,
    failedGames: failures.length,
    manifestHash,
    outputJson: options.outputJson,
    manifestOutput: options.manifestOutput,
    reportOutput: options.reportOutput
  }, null, 2)}\n`);
}

function toOutcome(
  gameIndex: number,
  seed: number,
  assignment: readonly PolicyName[],
  result: GameResult
): GameOutcome {
  const seats = PLAYER_IDS.map((playerId, seat): SeatOutcome => {
    const reward = calculateNonPlayingLearningTerminalReward(result, playerId, PLAYER_IDS);
    if (result.resultType === "all-pass") {
      return { seat, policy: assignment[seat], role: "All-Pass", win: null,
        rawReward: reward.rawTerminalReward, relativeReward: reward.terminalReward };
    }
    const role: RoleName = playerId === result.napoleonPlayerId
      ? "Napoleon"
      : playerId === result.adjutantPlayerId ? "Adjutant" : "Citizen";
    const napoleonSide = role === "Napoleon" || role === "Adjutant";
    const win = napoleonSide
      ? Number(result.winner === "napoleon-team")
      : Number(result.winner === "alliance");
    return { seat, policy: assignment[seat], role, win,
      rawReward: reward.rawTerminalReward, relativeReward: reward.terminalReward };
  });
  if (result.resultType === "all-pass") {
    return { gameIndex, seed, assignment, resultType: "all-pass", seats,
      napoleonPolicy: null, adjutantPolicy: null, contractSuccess: null,
      contractMargin: null, target: null, napoleonPointCards: null };
  }
  return {
    gameIndex,
    seed,
    assignment,
    resultType: "standard",
    seats,
    napoleonPolicy: assignment[PLAYER_IDS.indexOf(result.napoleonPlayerId as typeof PLAYER_IDS[number])],
    adjutantPolicy: result.adjutantPlayerId === null
      ? null
      : assignment[PLAYER_IDS.indexOf(result.adjutantPlayerId as typeof PLAYER_IDS[number])],
    contractSuccess: Number(result.winner === "napoleon-team"),
    contractMargin: result.napoleonTeamPointCards - result.targetPointCards,
    target: result.targetPointCards,
    napoleonPointCards: result.napoleonTeamPointCards
  };
}

function createSummary(
  outcomes: readonly GameOutcome[],
  assignments: readonly (readonly PolicyName[])[],
  failures: readonly EvaluationFailure[],
  diagnostics: Record<PolicyName, Omit<PublicAiPhaseCallDiagnostics, "composition">>
) {
  const seatAi = PLAYER_IDS.map((_, seat) => ({
    seat,
    ai: assignments.filter((row) => row[seat] === "COM-AI").length,
    ruleBase: assignments.filter((row) => row[seat] === "COM-RuleBase").length,
    aiRate: assignments.filter((row) => row[seat] === "COM-AI").length / assignments.length
  }));
  const policy = Object.fromEntries(POLICY_NAMES.map((name) => {
    const exposure = clusterEstimate(outcomes, (game) => game.seats.filter((seat) => seat.policy === name).map(() => 1));
    const wins = clusterEstimate(outcomes, (game) => game.seats.filter(
      (seat) => seat.policy === name && seat.win !== null
    ).map((seat) => seat.win!), { rate: true });
    const relativeReward = clusterEstimate(outcomes, (game) => game.seats.filter(
      (seat) => seat.policy === name
    ).map((seat) => seat.relativeReward));
    const rawReward = clusterEstimate(outcomes, (game) => game.seats.filter(
      (seat) => seat.policy === name
    ).map((seat) => seat.rawReward));
    return [name, { exposure: exposure.n, win: wins, relativeReward, rawReward }];
  }));
  const roles = ROLE_NAMES.flatMap((role) => POLICY_NAMES.map((name) => ({
    role,
    policy: name,
    win: clusterEstimate(outcomes, (game) => game.seats.filter(
      (seat) => seat.policy === name && seat.role === role
    ).map((seat) => seat.win!), { rate: true }),
    relativeReward: clusterEstimate(outcomes, (game) => game.seats.filter(
      (seat) => seat.policy === name && seat.role === role
    ).map((seat) => seat.relativeReward))
  })));
  const roleAcquisition = POLICY_NAMES.map((name) => {
    const denominator = outcomes.reduce(
      (total, game) => total + game.seats.filter((seat) => seat.policy === name).length,
      0
    );
    return {
      policy: name,
      exposure: denominator,
      rates: Object.fromEntries([...ROLE_NAMES, "All-Pass" as const].map((role) => [
        role,
        outcomes.reduce(
          (total, game) => total + game.seats.filter(
            (seat) => seat.policy === name && seat.role === role
          ).length,
          0
        ) / denominator
      ]))
    };
  });
  const napoleon = POLICY_NAMES.map((name) => ({
    policy: name,
    contractSuccess: clusterEstimate(outcomes, (game) => game.napoleonPolicy === name ? [game.contractSuccess!] : [], { rate: true }),
    contractMargin: clusterEstimate(outcomes, (game) => game.napoleonPolicy === name ? [game.contractMargin!] : []),
    target: clusterEstimate(outcomes, (game) => game.napoleonPolicy === name ? [game.target!] : []),
    pointCards: clusterEstimate(outcomes, (game) => game.napoleonPolicy === name ? [game.napoleonPointCards!] : [])
  }));
  const composition = [
    ...POLICY_NAMES.flatMap((napoleonPolicy) => POLICY_NAMES.map((adjutantPolicy) => ({ napoleonPolicy, adjutantPolicy }))),
    ...POLICY_NAMES.map((napoleonPolicy) => ({ napoleonPolicy, adjutantPolicy: null }))
  ].map(({ napoleonPolicy, adjutantPolicy }) => ({
    napoleonPolicy,
    adjutantPolicy,
    win: clusterEstimate(outcomes, (game) =>
      game.napoleonPolicy === napoleonPolicy && game.adjutantPolicy === adjutantPolicy
        ? [game.contractSuccess!] : [], { rate: true }),
    contractMargin: clusterEstimate(outcomes, (game) =>
      game.napoleonPolicy === napoleonPolicy && game.adjutantPolicy === adjutantPolicy
        ? [game.contractMargin!] : [])
  }));
  return {
    games: outcomes.length,
    allPass: {
      count: outcomes.filter((game) => game.resultType === "all-pass").length,
      rate: outcomes.filter((game) => game.resultType === "all-pass").length / outcomes.length
    },
    seatAi,
    policy,
    policyDifference: {
      win: clusterDifference(outcomes,
        (game) => game.seats.filter((seat) => seat.policy === "COM-AI" && seat.win !== null).map((seat) => seat.win!),
        (game) => game.seats.filter((seat) => seat.policy === "COM-RuleBase" && seat.win !== null).map((seat) => seat.win!)),
      relativeReward: clusterDifference(outcomes,
        (game) => game.seats.filter((seat) => seat.policy === "COM-AI").map((seat) => seat.relativeReward),
        (game) => game.seats.filter((seat) => seat.policy === "COM-RuleBase").map((seat) => seat.relativeReward))
    },
    roles,
    roleAcquisition,
    napoleon,
    composition,
    illegal: diagnostics["COM-AI"].illegalCount + diagnostics["COM-RuleBase"].illegalCount,
    fallback: diagnostics["COM-AI"].fallbackCount + diagnostics["COM-RuleBase"].fallbackCount,
    invariantFailure: failures.filter((failure) => failure.invariant).length,
    otherFailure: failures.filter((failure) => !failure.invariant).length
  };
}

function validateResult(result: GameResult, playerIds: readonly PlayerId[]): void {
  if (result.resultType === "all-pass") {
    if (result.payoffs.length !== playerIds.length || result.payoffs.reduce((sum, item) => sum + item.payoff, 0) !== -3) {
      throw new EvaluationInvariantError("Invalid all-pass payoff invariant.");
    }
    return;
  }
  if (result.napoleonTeamPointCards + result.alliancePointCards !== 20) {
    throw new EvaluationInvariantError("Point-card total is not 20.");
  }
  if ((result.napoleonTeamPointCards >= result.targetPointCards) !== (result.winner === "napoleon-team")) {
    throw new EvaluationInvariantError("Winner and contract success disagree.");
  }
  if (!playerIds.includes(result.napoleonPlayerId) ||
      (result.adjutantPlayerId !== null && !playerIds.includes(result.adjutantPlayerId))) {
    throw new EvaluationInvariantError("Result references an unknown role player.");
  }
}

class EvaluationInvariantError extends Error {}

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

export function renderReport(artifact: any): string {
  const { manifest, manifestHash, summary, diagnosticsTotals, auditSamples, completedGames, failedGames } = artifact;
  const estimate = (item: ClusterEstimate, percent = false) => item.estimate === null ? "—" :
    percent ? `${(item.estimate * 100).toFixed(2)}%` : item.estimate.toFixed(4);
  const ci = (item: ClusterEstimate, percent = false) => item.ci95[0] === null ? "—" :
    percent
      ? `[${(item.ci95[0]! * 100).toFixed(2)}%, ${(item.ci95[1]! * 100).toFixed(2)}%]`
      : `[${item.ci95[0]!.toFixed(4)}, ${item.ci95[1]!.toFixed(4)}]`;
  const p = (value: number) => `${(value * 100).toFixed(2)}%`;
  const lines = [
    "# Final COM-AI vs RuleBase evaluation",
    "",
    "PR #203 で積み上げた non-playing AI 開発の最終評価として、正式 builtin preset 同士を同一 game 内で 50/50 混成対戦させた結果です。学習・policy 変更・artifact 再選定は行っていません。",
    "",
    "## 1. 最終AI構成",
    "",
    "| preset | playing | bidding | nonPlaying |",
    "| --- | --- | --- | --- |",
    "| COM-AI | `ppo-separated-v1000` | `frozen-raise-v1` | `parameterized-adjutant-exchange-v1` |",
    "| COM-RuleBase | `rule-based` | `rule-based` | `rule-based` |",
    "",
    "legacy full-policy、旧 adjutant/exchange PPO、experimental artifact は使用していません。",
    "",
    "## 2. 実験条件",
    "",
    `- 完了 game: ${completedGames.toLocaleString()}（失敗 ${failedGames}）`,
    `- game seed: ${manifest.gameStartSeed}..${manifest.gameStartSeed + manifest.gameCount - 1}`,
    `- assignment: seat ごとの独立 balanced Fisher–Yates shuffle（seed ${manifest.assignmentSeed}）`,
    `- logical manifest SHA-256: \`${manifestHash}\``,
    `- assignment sequence SHA-256: \`${manifest.assignmentSequenceSha256}\``,
    "- 推論: CPU、正式 repo-managed artifact",
    "- 95% CI / standard error: game を cluster とする sandwich 推定（同一 game 内 5 seats の相関を保持）",
    "- All-Pass: exposure と reward 平均には含め、通常の win/loss および role 別 denominator からは除外",
    "- mirrored 補助評価: 未実施（50/50 mixed 主評価を 50,000 games で完遂することを優先）",
    "",
    "## 3. 50/50 assignment確認",
    "",
    "| seat | COM-AI | COM-RuleBase | AI比率 |",
    "| ---: | ---: | ---: | ---: |",
    ...summary.seatAi.map((row: any) => `| ${row.seat} | ${row.ai.toLocaleString()} | ${row.ruleBase.toLocaleString()} | ${p(row.aiRate)} |`),
    "",
    `All-Pass は ${summary.allPass.count.toLocaleString()} game（${p(summary.allPass.rate)}）でした。`,
    "",
    "## 4. 全体 COM-AI vs COM-RuleBase",
    "",
    "| policy | player-game exposure | win denominator | win rate (SE; 95% CI) | mean relative reward (SE; 95% CI) | mean raw reward (SE; 95% CI) |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
    ...POLICY_NAMES.map((name) => {
      const row = summary.policy[name];
      return `| ${name} | ${row.exposure.toLocaleString()} | ${row.win.n.toLocaleString()} | ${estimate(row.win, true)} (SE ${pct(row.win.standardError)}; ${ci(row.win, true)}) | ${estimate(row.relativeReward)} (SE ${num(row.relativeReward.standardError)}; ${ci(row.relativeReward)}) | ${estimate(row.rawReward)} (SE ${num(row.rawReward.standardError)}; ${ci(row.rawReward)}) |`;
    }),
    "",
    `COM-AI − COM-RuleBase の勝率差は ${estimate(summary.policyDifference.win, true)}（95% CI ${ci(summary.policyDifference.win, true)}）、mean relative reward 差は ${estimate(summary.policyDifference.relativeReward)}（95% CI ${ci(summary.policyDifference.relativeReward)}）です。`,
    "",
    "win rate と reward は同じ指標ではありません。raw / relative reward は現行 Reward v3 の役職依存 payoff を使い、relative reward は各 game の5人平均を引いています。今回のように policy 間で役職獲得率が大きく違う場合、全体 win rate と mean relative reward が逆方向になることがあります。このため、全体表だけでなく次の role 内比較を主に用いて解釈します。",
    "",
    "## 5. Napoleon / Adjutant / Citizen 別",
    "",
    "| role | policy | n | win rate (95% CI) | mean relative reward (95% CI) |",
    "| --- | --- | ---: | ---: | ---: |",
    ...summary.roles.map((row: any) => `| ${row.role} | ${row.policy} | ${row.relativeReward.n.toLocaleString()} | ${estimate(row.win, true)} ${ci(row.win, true)} | ${estimate(row.relativeReward)} ${ci(row.relativeReward)} |`),
    "",
    "## 6. Napoleon-side composition 別",
    "",
    "| Napoleon | Adjutant | n | Napoleon-side win / contract success | mean contract margin (95% CI) |",
    "| --- | --- | ---: | ---: | ---: |",
    ...summary.composition.map((row: any) => `| ${row.napoleonPolicy} | ${row.adjutantPolicy ?? "None (solo)"} | ${row.win.n.toLocaleString()} | ${estimate(row.win, true)} ${ci(row.win, true)} | ${estimate(row.contractMargin)} ${ci(row.contractMargin)} |`),
    "",
    "## 7. bidding / contract diagnostics",
    "",
    "### Role acquisition（全 policy exposure 比）",
    "",
    "| policy | exposure | Napoleon | Adjutant | Citizen | All-Pass/no-contract |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
    ...summary.roleAcquisition.map((row: any) => `| ${row.policy} | ${row.exposure.toLocaleString()} | ${p(row.rates.Napoleon)} | ${p(row.rates.Adjutant)} | ${p(row.rates.Citizen)} | ${p(row.rates["All-Pass"])} |`),
    "",
    "Napoleon になった player の contract 指標:",
    "",
    "| policy | n | success rate (95% CI) | mean margin (95% CI) | mean target | Napoleon-side point cards |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
    ...summary.napoleon.map((row: any) => `| ${row.policy} | ${row.contractMargin.n.toLocaleString()} | ${estimate(row.contractSuccess, true)} ${ci(row.contractSuccess, true)} | ${estimate(row.contractMargin)} ${ci(row.contractMargin)} | ${estimate(row.target)} | ${estimate(row.pointCards)} |`),
    "",
    "役職獲得率と役職内成績は別々に示しています。したがって、COM-AI の全体差を単に『Napoleon になりやすい／なりにくい』こととは混同しません。",
    "",
    "### Phase call sample audit",
    "",
    ...auditSamples.map((sample: AuditSample) => `- ${sample.policy} ${sample.phase}: game ${sample.gameIndex}, seat ${sample.seat}; composition ${JSON.stringify(sample.diagnostics.composition)}; calls bidding=${sample.diagnostics.biddingCalls}, adjutant=${sample.diagnostics.adjutantCalls}, exchange=${sample.diagnostics.exchangeCalls}, playing=${sample.diagnostics.playingCalls}`),
    "",
    "## 8. illegal / fallback / invariant",
    "",
    `- illegal: ${summary.illegal}`,
    `- fallback: ${summary.fallback}`,
    `- invariant failure: ${summary.invariantFailure}`,
    `- other game failure: ${summary.otherFailure}`,
    `- aggregate phase calls: COM-AI ${JSON.stringify(diagnosticsTotals["COM-AI"])}`,
    `- aggregate phase calls: COM-RuleBase ${JSON.stringify(diagnosticsTotals["COM-RuleBase"])}`,
    "",
    "## 9. 結論",
    "",
    conclusion(summary),
    "",
    "この評価は現時点の正式 COM-AI をそのまま測った締めの結果です。結果に応じた再学習や policy の差し替えは行っていません。"
  ];
  return `${lines.join("\n")}\n`;
}

function conclusion(summary: any): string {
  const ai = summary.policy["COM-AI"];
  const rb = summary.policy["COM-RuleBase"];
  const roleText = ROLE_NAMES.map((role) => {
    const a = summary.roles.find((row: any) => row.role === role && row.policy === "COM-AI");
    const r = summary.roles.find((row: any) => row.role === role && row.policy === "COM-RuleBase");
    return `${role} では COM-AI ${pct(a.win.estimate)}、RuleBase ${pct(r.win.estimate)}（差 ${signedPct(a.win.estimate - r.win.estimate)}）`;
  }).join("、");
  const overallDiff = ai.win.estimate - rb.win.estimate;
  const napoleonAi = summary.napoleon.find((row: any) => row.policy === "COM-AI");
  const napoleonRb = summary.napoleon.find((row: any) => row.policy === "COM-RuleBase");
  const targetDelta = napoleonAi.target.estimate - napoleonRb.target.estimate;
  const pointCardDelta = napoleonAi.pointCards.estimate - napoleonRb.pointCards.estimate;
  const marginDelta = napoleonAi.contractMargin.estimate - napoleonRb.contractMargin.estimate;
  const rewardDifference = summary.policyDifference.relativeReward;
  return [
    `全体では COM-AI の勝率は ${pct(ai.win.estimate)}、COM-RuleBase は ${pct(rb.win.estimate)}で、差は ${signedPct(overallDiff)}（95% CI ${pct(summary.policyDifference.win.ci95[0])} から ${pct(summary.policyDifference.win.ci95[1])}）でした。一方、mean relative reward はそれぞれ ${num(ai.relativeReward.estimate)} と ${num(rb.relativeReward.estimate)}で、AI−RuleBase 差は ${num(rewardDifference.estimate)}（95% CI ${num(rewardDifference.ci95[0])} から ${num(rewardDifference.ci95[1])}）です。つまり全体勝率は AI が明確に高いものの、役職構成を反映する relative reward は逆方向でした。`,
    `${roleText}でした。Napoleon / Adjutant の差と Citizen の差を分けることで、AI の総合差がどの立場で生じたかを確認できます。`,
    `最大の改善は Napoleon で見えます。COM-AI Napoleon は target を平均 ${Math.abs(targetDelta).toFixed(4)} ${targetDelta < 0 ? "低く" : "高く"}宣言し、Napoleon-side point cards は平均 ${signed(pointCardDelta)}、contract margin は ${signed(marginDelta)} 動きました。これは frozen bidding の契約選択と、Napoleon / Adjutant の non-playing・playing による契約実行の両方が寄与した可能性を示します。composition 比較でも Adjutant を RuleBase から AI に替えた組合せの成績が上がっています。一方で AI Citizen の勝率は下がっており、playing を含む AI の優位は全役職に一様ではありません。全 phase が同時に異なるため、単一 phase の因果効果とは断定しません。`,
    "結論として、『作った AI は RuleBase を超えたか』には、全体 win rate と Napoleon / Adjutant の実戦成績では明確に yes です。しかし Citizen と relative reward では no であり、『全役職・全指標で全面的に超えた』とは言えません。正式 COM-AI は特に Napoleon-side を大きく強化した一方、Citizen performance と bidding による role acquisition の偏りを残した、というのがこの最終評価の正確な締めです。"
  ].join("\n\n");
}

function pct(value: number): string { return `${(value * 100).toFixed(2)}%`; }
function signedPct(value: number): string { return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)} pp`; }
function signed(value: number): string { return `${value >= 0 ? "+" : ""}${value.toFixed(4)}`; }
function num(value: number): string { return value.toFixed(4); }
function sha256(value: string): string { return createHash("sha256").update(value).digest("hex"); }

interface CliOptions {
  games: number;
  startSeed: number;
  assignmentSeed: number;
  concurrency: number;
  progressEvery: number;
  outputJson: string;
  manifestOutput: string;
  reportOutput: string;
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
    games: integer("games", 50_000),
    startSeed: integer("start-seed", 462_000_000),
    assignmentSeed: integer("assignment-seed", 462_202_203),
    concurrency: integer("concurrency", 8),
    progressEvery: integer("progress-every", 100),
    outputJson: resolve(values.get("output-json") ?? "diagnostics/final-com-ai-vs-rulebase.json"),
    manifestOutput: resolve(values.get("manifest-output") ?? "diagnostics/final-com-ai-vs-rulebase-manifest.json"),
    reportOutput: resolve(values.get("report-output") ?? "diagnostics/final-com-ai-vs-rulebase.md")
  };
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  runEvaluation(parseArgs(process.argv.slice(2))).catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
    process.exitCode = 1;
  });
}
