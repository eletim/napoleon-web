import type { PlayCardRequest } from "@napoleon/protocol";

export function isPlayCardRequest(value: unknown): value is PlayCardRequest {
  if (!isRecord(value)) {
    return false;
  }

  const keys = Object.keys(value);
  return (
    keys.length === 2 &&
    keys.includes("type") &&
    keys.includes("cardId") &&
    value.type === "play-card" &&
    typeof value.cardId === "string"
  );
}

export function readActionBody(value: unknown): PlayCardRequest | undefined {
  if (!isRecord(value) || !("action" in value)) {
    return undefined;
  }

  const keys = Object.keys(value);

  if (keys.length !== 1) {
    return undefined;
  }

  return isPlayCardRequest(value.action) ? value.action : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
