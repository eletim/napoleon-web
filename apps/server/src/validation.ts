import type { GameAction } from "@napoleon/game-core";

export function isGameAction(value: unknown): value is GameAction {
  if (!isRecord(value)) {
    return false;
  }

  if (value.type === "play-card") {
    return typeof value.playerId === "string" && typeof value.cardId === "string";
  }

  if (value.type === "next-trick") {
    return typeof value.playerId === "string";
  }

  return false;
}

export function readActionBody(value: unknown): GameAction | undefined {
  if (!isRecord(value) || !("action" in value)) {
    return undefined;
  }

  return isGameAction(value.action) ? value.action : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
