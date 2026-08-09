import type {
  BidRequest,
  ChooseAdjutantRequest,
  CreateGameAgentSelection,
  CreateGameRequest,
  DiscardCardsRequest,
  PassRequest,
  PlayCardRequest,
  PublicGameAction,
  RunAutomatedSimulationRequest
} from "@napoleon/protocol";

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

export function isDiscardCardsRequest(value: unknown): value is DiscardCardsRequest {
  if (!isRecord(value)) {
    return false;
  }

  const keys = Object.keys(value);
  return (
    keys.length === 2 &&
    keys.includes("type") &&
    keys.includes("cardIds") &&
    value.type === "discard-cards" &&
    Array.isArray(value.cardIds) &&
    value.cardIds.every((cardId) => typeof cardId === "string")
  );
}

export function isChooseAdjutantRequest(value: unknown): value is ChooseAdjutantRequest {
  if (!isRecord(value)) {
    return false;
  }

  const keys = Object.keys(value);
  return (
    keys.length === 2 &&
    keys.includes("type") &&
    keys.includes("cardId") &&
    value.type === "choose-adjutant" &&
    typeof value.cardId === "string"
  );
}

export function readActionBody(value: unknown): PublicGameAction | undefined {
  if (!isRecord(value) || !("action" in value)) {
    return undefined;
  }

  const keys = Object.keys(value);

  if (keys.length !== 1) {
    return undefined;
  }

  if (
    isPlayCardRequest(value.action) ||
    isBidRequest(value.action) ||
    isPassRequest(value.action) ||
    isDiscardCardsRequest(value.action) ||
    isChooseAdjutantRequest(value.action)
  ) {
    return value.action;
  }

  return undefined;
}

export function readCreateGameBody(value: unknown): CreateGameRequest | undefined {
  if (value === undefined) {
    return {};
  }

  if (!isRecord(value)) {
    return undefined;
  }

  const keys = Object.keys(value);

  if (keys.length === 0) {
    return {};
  }

  if (keys.length !== 1 || !keys.includes("aiAgents") || !Array.isArray(value.aiAgents)) {
    return undefined;
  }

  const aiAgents: CreateGameAgentSelection[] = [];

  for (const selection of value.aiAgents) {
    const parsed = readCreateGameAgentSelection(selection);

    if (parsed === undefined) {
      return undefined;
    }

    aiAgents.push(parsed);
  }

  return {
    aiAgents
  };
}

export function readRunAutomatedSimulationBody(
  value: unknown
): RunAutomatedSimulationRequest | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  const keys = Object.keys(value);

  if (keys.length !== 1 || !keys.includes("seed")) {
    return undefined;
  }

  if (
    typeof value.seed !== "number" ||
    !Number.isSafeInteger(value.seed) ||
    value.seed < 0 ||
    value.seed > 0xffffffff
  ) {
    return undefined;
  }

  return {
    seed: value.seed
  };
}

function readCreateGameAgentSelection(
  value: unknown
): CreateGameAgentSelection | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  const keys = Object.keys(value);

  if (
    keys.length !== 2 ||
    !keys.includes("playerId") ||
    !keys.includes("agentId") ||
    typeof value.playerId !== "string" ||
    typeof value.agentId !== "string"
  ) {
    return undefined;
  }

  return {
    playerId: value.playerId,
    agentId: value.agentId
  };
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
