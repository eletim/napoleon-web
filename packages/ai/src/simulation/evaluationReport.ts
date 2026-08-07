import type {
  CompletedEvaluationGameRecord,
  EvaluationAgentPerformanceSummary,
  EvaluationComparisonSummary,
  EvaluationFailureSummary,
  EvaluationGameCountSummary,
  EvaluationGameRecord,
  EvaluationPerformanceSummary,
  EvaluationRateSummary,
  EvaluationReport,
  EvaluationReportSummary,
  EvaluationRolePerformanceSummary,
  EvaluationRunRecord,
  EvaluationSeatAssignment,
  EvaluationSeatPerformanceSummary,
  EvaluationSeatRole
} from "./types.js";

type CompletedRole = Exclude<EvaluationSeatRole, "unknown">;

const completedRoles: readonly CompletedRole[] = ["napoleon", "adjutant", "alliance"];

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
  failuresByReason: Map<string, number>;
}

export function createEvaluationReport(record: EvaluationRunRecord): EvaluationReport {
  const summaryStats = createStats();
  const participantStats = createStats();
  const agentNames = new Map<number, string>();
  const agentStats = new Map<number, MutableStats>();
  const agentRoleStats = new Map<number, Map<CompletedRole, MutableStats>>();
  const agentSeatStats = new Map<number, Map<number, MutableStats>>();
  const seatStats = new Map<number, MutableStats>();
  const roleStats = new Map<CompletedRole, MutableStats>();

  for (const game of [...record.games].sort(compareGames)) {
    countGame(summaryStats, game, undefined);

    for (const seat of [...game.seats].sort(compareSeats)) {
      agentNames.set(seat.sourceAgentIndex, seat.agentName);
      countGame(participantStats, game, seat);
      countGame(getOrCreate(agentStats, seat.sourceAgentIndex), game, seat);
      countGame(getOrCreate(seatStats, seat.seatIndex), game, seat);

      const perAgentSeat = getOrCreateMap(agentSeatStats, seat.sourceAgentIndex);
      countGame(getOrCreate(perAgentSeat, seat.seatIndex), game, seat);

      if (isCompletedRole(seat.role)) {
        countGame(getOrCreate(roleStats, seat.role), game, seat);

        const perAgentRole = getOrCreateMap(agentRoleStats, seat.sourceAgentIndex);
        countGame(getOrCreate(perAgentRole, seat.role), game, seat);
      }
    }
  }

  const summary = toReportSummary(
    summaryStats,
    record.gameCount * record.rotationOffsets.length
  );
  const participantSummary = toPerformanceSummary(participantStats);
  const seats = [...seatStats.entries()]
    .sort(([left], [right]) => left - right)
    .map(([seatIndex, stats]) => toSeatSummary(seatIndex, stats));
  const roles = completedRoles.map((role) =>
    toRoleSummary(role, roleStats.get(role) ?? createStats())
  );
  const agents = [...agentStats.entries()]
    .sort(([left], [right]) => compareAgents(left, right, agentNames))
    .map(([sourceAgentIndex, stats]) =>
      toAgentSummary(
        sourceAgentIndex,
        getAgentName(agentNames, sourceAgentIndex),
        stats,
        agentRoleStats.get(sourceAgentIndex),
        agentSeatStats.get(sourceAgentIndex),
        participantSummary
      )
    );

  return {
    schemaVersion: 1,
    sourceSchemaVersion: record.schemaVersion,
    summary,
    agents,
    seats,
    roles
  };
}

function countGame(
  stats: MutableStats,
  game: EvaluationGameRecord,
  seat: EvaluationSeatAssignment | undefined
): void {
  stats.games.total += 1;

  if (game.status === "failed") {
    stats.games.failed += 1;
    incrementReason(stats.failuresByReason, game.failureReason);
    return;
  }

  stats.games.completed += 1;
  stats.contractSuccesses += game.contractSucceeded ? 1 : 0;

  if (seat === undefined) {
    stats.wins += game.winner === "napoleon-team" ? 1 : 0;
    stats.losses += game.winner === "alliance" ? 1 : 0;
    stats.pointCardTotal += game.pointCards.napoleonTeam;
    return;
  }

  if (!isCompletedRole(seat.role)) {
    return;
  }

  const won = didSeatWin(game, seat.role);
  stats.wins += won ? 1 : 0;
  stats.losses += won ? 0 : 1;
  stats.pointCardTotal += pointCardsForRole(game, seat.role);
}

function didSeatWin(game: CompletedEvaluationGameRecord, role: CompletedRole): boolean {
  return role === "alliance"
    ? game.winner === "alliance"
    : game.winner === "napoleon-team";
}

function pointCardsForRole(game: CompletedEvaluationGameRecord, role: CompletedRole): number {
  return role === "alliance" ? game.pointCards.alliance : game.pointCards.napoleonTeam;
}

function toAgentSummary(
  sourceAgentIndex: number,
  agentName: string,
  stats: MutableStats,
  roleStats: ReadonlyMap<CompletedRole, MutableStats> | undefined,
  seatStats: ReadonlyMap<number, MutableStats> | undefined,
  baseline: EvaluationPerformanceSummary
): EvaluationAgentPerformanceSummary {
  const summary = toPerformanceSummary(stats);

  return {
    sourceAgentIndex,
    agentName,
    ...summary,
    roleResults: completedRoles.map((role) =>
      toRoleSummary(role, roleStats?.get(role) ?? createStats())
    ),
    seatResults: [...(seatStats?.entries() ?? [])]
      .sort(([left], [right]) => left - right)
      .map(([seatIndex, seatSummaryStats]) => toSeatSummary(seatIndex, seatSummaryStats)),
    comparison: createComparison(summary, baseline)
  };
}

function toReportSummary(stats: MutableStats, expectedGameCount: number): EvaluationReportSummary {
  return {
    expectedGameCount,
    ...toPerformanceSummary(stats)
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

function toPerformanceSummary(stats: MutableStats): EvaluationPerformanceSummary {
  return {
    games: toGameCountSummary(stats.games),
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

function createComparison(
  summary: EvaluationPerformanceSummary,
  baseline: EvaluationPerformanceSummary
): EvaluationComparisonSummary {
  return {
    winRateDelta: subtractNullable(summary.winRate.rate, baseline.winRate.rate),
    contractSuccessRateDelta: subtractNullable(
      summary.contractSuccessRate.rate,
      baseline.contractSuccessRate.rate
    ),
    averagePointCardsDelta: subtractNullable(
      summary.averagePointCards,
      baseline.averagePointCards
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
    failuresByReason: new Map()
  };
}

function toRate(numerator: number, denominator: number): EvaluationRateSummary {
  return {
    numerator,
    denominator,
    rate: denominator === 0 ? null : numerator / denominator
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

function getOrCreateMap<TKey, TNestedKey>(
  map: Map<TKey, Map<TNestedKey, MutableStats>>,
  key: TKey
): Map<TNestedKey, MutableStats> {
  const existing = map.get(key);

  if (existing !== undefined) {
    return existing;
  }

  const nested = new Map<TNestedKey, MutableStats>();
  map.set(key, nested);
  return nested;
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

function compareAgents(
  left: number,
  right: number,
  agentNames: ReadonlyMap<number, string>
): number {
  return getAgentName(agentNames, left).localeCompare(getAgentName(agentNames, right))
    || left - right;
}

function getAgentName(agentNames: ReadonlyMap<number, string>, sourceAgentIndex: number): string {
  const agentName = agentNames.get(sourceAgentIndex);

  if (agentName === undefined) {
    throw new Error(`Missing agent name for sourceAgentIndex=${sourceAgentIndex}.`);
  }

  return agentName;
}
