import type { BidRequest, PassRequest, PlayCardRequest, PublicGameAction } from "@napoleon/protocol";

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

export function isBidRequest(value: unknown): value is BidRequest {
  if (!isRecord(value)) {
    return false;
  }

  const keys = Object.keys(value);
  return (
    keys.length === 3 &&
    keys.includes("type") &&
    keys.includes("suit") &&
    keys.includes("targetPointCards") &&
    value.type === "bid" &&
    isPublicSuit(value.suit) &&
    typeof value.targetPointCards === "number"
  );
}

export function isPassRequest(value: unknown): value is PassRequest {
  if (!isRecord(value)) {
    return false;
  }

  const keys = Object.keys(value);
  return keys.length === 1 && keys.includes("type") && value.type === "pass";
}

export function readActionBody(value: unknown): PublicGameAction | undefined {
  if (!isRecord(value) || !("action" in value)) {
    return undefined;
  }

  const keys = Object.keys(value);

  if (keys.length !== 1) {
    return undefined;
  }

  if (isPlayCardRequest(value.action) || isBidRequest(value.action) || isPassRequest(value.action)) {
    return value.action;
  }

  return undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isPublicSuit(value: unknown): value is BidRequest["suit"] {
  return (
    value === "clubs" ||
    value === "diamonds" ||
    value === "hearts" ||
    value === "spades"
  );
}
