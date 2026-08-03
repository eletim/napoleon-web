import { PLAYER_COUNT } from "./schema.js";

export function createRelativePlayerOrder(
  playerIds: readonly string[],
  selfPlayerId: string
): readonly string[] {
  validatePlayerIds(playerIds);
  const selfIndex = playerIds.indexOf(selfPlayerId);

  if (selfIndex === -1) {
    throw new Error(`selfPlayerId is not included in playerIds: ${selfPlayerId}`);
  }

  return [...playerIds.slice(selfIndex), ...playerIds.slice(0, selfIndex)];
}

export function getRelativePlayerIndex(
  relativePlayerIds: readonly string[],
  playerId: string
): number {
  validatePlayerIds(relativePlayerIds);
  const index = relativePlayerIds.indexOf(playerId);

  if (index === -1) {
    throw new Error(`playerId is not included in relativePlayerIds: ${playerId}`);
  }

  return index;
}

function validatePlayerIds(playerIds: readonly string[]): void {
  if (playerIds.length !== PLAYER_COUNT) {
    throw new Error(`Expected exactly ${PLAYER_COUNT} player ids, got ${playerIds.length}.`);
  }

  const uniquePlayerIds = new Set(playerIds);

  if (uniquePlayerIds.size !== playerIds.length) {
    throw new Error("Player ids must be unique.");
  }
}
