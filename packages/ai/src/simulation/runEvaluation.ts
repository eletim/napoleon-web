import type { GameResult, PlayerId } from "@napoleon/game-core";
import { runAutomatedGame } from "./runAutomatedGame.js";
import { normalizeSeed } from "./seededRandom.js";
import type {
  CompletedEvaluationGameRecord,
  EvaluationAgentDefinition,
  EvaluationGameRecord,
  EvaluationRunRecord,
  EvaluationSeatAssignment,
  EvaluationSeatRole,
  FailedEvaluationGameRecord,
  RunEvaluationOptions
} from "./types.js";

const defaultPlayerIds: readonly PlayerId[] = [
  "player-0",
  "player-1",
  "player-2",
  "player-3",
  "player-4"
];

const defaultRotationOffsets = [0] as const;

export async function runEvaluation(
  options: RunEvaluationOptions
): Promise<EvaluationRunRecord> {
  const playerIds = options.playerIds ?? defaultPlayerIds;
  const rotationOffsets = options.rotationOffsets ?? defaultRotationOffsets;

  validateOptions(options, playerIds, rotationOffsets);

  const startSeed = normalizeSeed(options.startSeed);
  const endSeed = normalizeSeed(startSeed + options.gameCount - 1);
  const games: EvaluationGameRecord[] = [];
  let evaluationGameIndex = 0;

  for (let seedOffset = 0; seedOffset < options.gameCount; seedOffset += 1) {
    const seed = normalizeSeed(startSeed + seedOffset);

    for (const rotationOffset of rotationOffsets) {
      const normalizedRotationOffset = normalizeRotationOffset(rotationOffset, playerIds.length);
      const assignments = createSeatAssignments(
        playerIds,
        options.agents,
        normalizedRotationOffset,
        "unknown"
      );

      try {
        const record = await runAutomatedGame({
          seed,
          playerIds,
          maxDecisionSteps: options.maxDecisionSteps,
          createAgent: (context) => {
            const assignment = assignments[context.playerIndex];
            const definition = options.agents[assignment.sourceAgentIndex];

            return definition.createAgent({
              ...context,
              agentName: definition.name,
              sourceAgentIndex: assignment.sourceAgentIndex,
              seatIndex: assignment.seatIndex,
              rotationOffset: normalizedRotationOffset,
              evaluationSeed: seed,
              evaluationGameIndex
            });
          }
        });

        games.push(createCompletedGameRecord({
          gameIndex: evaluationGameIndex,
          seed,
          rotationOffset: normalizedRotationOffset,
          playerIds,
          assignments,
          result: record.result
        }));
      } catch (error) {
        games.push(createFailedGameRecord({
          gameIndex: evaluationGameIndex,
          seed,
          rotationOffset: normalizedRotationOffset,
          playerIds,
          assignments,
          failureReason: formatFailureReason(error)
        }));
      }

      evaluationGameIndex += 1;
    }
  }

  const completedCount = games.filter((game) => game.status === "completed").length;

  return {
    schemaVersion: 1,
    startSeed,
    endSeed,
    gameCount: options.gameCount,
    rotationOffsets: rotationOffsets.map((offset) =>
      normalizeRotationOffset(offset, playerIds.length)
    ),
    playerIds: [...playerIds],
    games,
    completedCount,
    failedCount: games.length - completedCount
  };
}

function validateOptions(
  options: RunEvaluationOptions,
  playerIds: readonly PlayerId[],
  rotationOffsets: readonly number[]
): void {
  normalizeSeed(options.startSeed);

  if (!Number.isSafeInteger(options.gameCount) || options.gameCount < 1) {
    throw new Error("gameCount must be a positive integer.");
  }

  normalizeSeed(options.startSeed + options.gameCount - 1);

  if (new Set(playerIds).size !== playerIds.length) {
    throw new Error("playerIds must be unique.");
  }

  if (playerIds.length !== 5) {
    throw new Error("evaluation runner requires exactly 5 playerIds.");
  }

  if (options.agents.length !== playerIds.length) {
    throw new Error("agents length must match playerIds length.");
  }

  if (rotationOffsets.length === 0) {
    throw new Error("rotationOffsets must not be empty.");
  }

  for (const offset of rotationOffsets) {
    if (!Number.isSafeInteger(offset)) {
      throw new Error("rotationOffsets must contain only integers.");
    }
  }
}

function createCompletedGameRecord(args: {
  gameIndex: number;
  seed: number;
  rotationOffset: number;
  playerIds: readonly PlayerId[];
  assignments: readonly EvaluationSeatAssignment[];
  result: GameResult;
}): CompletedEvaluationGameRecord {
  const seats = args.assignments.map((assignment) => ({
    ...assignment,
    role: getSeatRole(assignment.playerId, args.result)
  }));

  return {
    schemaVersion: 1,
    status: "completed",
    gameIndex: args.gameIndex,
    seed: args.seed,
    rotationOffset: args.rotationOffset,
    playerIds: [...args.playerIds],
    seats,
    contract: {
      napoleonPlayerId: args.result.napoleonPlayerId,
      targetPointCards: args.result.targetPointCards,
      adjutantPlayerId: args.result.adjutantPlayerId
    },
    pointCards: {
      napoleonTeam: args.result.napoleonTeamPointCards,
      alliance: args.result.alliancePointCards
    },
    winner: args.result.winner,
    contractSucceeded: args.result.winner === "napoleon-team",
    result: args.result
  };
}

function createFailedGameRecord(args: {
  gameIndex: number;
  seed: number;
  rotationOffset: number;
  playerIds: readonly PlayerId[];
  assignments: readonly EvaluationSeatAssignment[];
  failureReason: string;
}): FailedEvaluationGameRecord {
  return {
    schemaVersion: 1,
    status: "failed",
    gameIndex: args.gameIndex,
    seed: args.seed,
    rotationOffset: args.rotationOffset,
    playerIds: [...args.playerIds],
    seats: args.assignments.map((assignment) => ({ ...assignment, role: "unknown" })),
    failureReason: args.failureReason
  };
}

function createSeatAssignments(
  playerIds: readonly PlayerId[],
  agents: readonly EvaluationAgentDefinition[],
  rotationOffset: number,
  role: EvaluationSeatRole
): readonly EvaluationSeatAssignment[] {
  return playerIds.map((playerId, seatIndex) => {
    const sourceAgentIndex = modulo(seatIndex - rotationOffset, agents.length);

    return {
      playerId,
      seatIndex,
      agentName: agents[sourceAgentIndex].name,
      sourceAgentIndex,
      role
    };
  });
}

function getSeatRole(playerId: PlayerId, result: GameResult): EvaluationSeatRole {
  if (playerId === result.napoleonPlayerId) {
    return "napoleon";
  }

  if (playerId === result.adjutantPlayerId) {
    return "adjutant";
  }

  return "alliance";
}

function normalizeRotationOffset(offset: number, playerCount: number): number {
  return modulo(offset, playerCount);
}

function modulo(value: number, divisor: number): number {
  return ((value % divisor) + divisor) % divisor;
}

function formatFailureReason(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
