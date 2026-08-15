import type {
  PublicGameAction,
  PublicGamePhase,
  PublicGameResult,
  PublicSimulationActualState,
  PublicSimulationDecision,
  PublicSimulationObservation,
  RunAutomatedSimulationResponse
} from "@napoleon/protocol";
import { suitSymbols } from "./cardSymbols";
import { formatCardId, formatWinningTeam } from "./displayText";

const phaseLabels: Record<PublicGamePhase, string> = {
  bidding: "競り",
  exchanging: "埋札交換",
  "choosing-adjutant": "副官指定",
  playing: "プレイ",
  finished: "終了"
};

const suitLabels = {
  spades: "スペード",
  hearts: "ハート",
  diamonds: "ダイヤ",
  clubs: "クラブ"
} as const;

export function formatSimulationAction(
  playerId: string,
  action: PublicGameAction
): string {
  switch (action.type) {
    case "bid":
      return `${playerId}: ${suitLabels[action.suit]} ${action.targetPointCards}枚で入札`;
    case "pass":
      return `${playerId}: パス`;
    case "choose-adjutant":
      return `${playerId}: ${formatCardId(action.cardId)}を副官指定`;
    case "discard-cards":
      return `${playerId}: ${action.cardIds.length}枚を捨てた`;
    case "play-card":
      return `${playerId}: ${formatCardId(action.cardId)}を出した`;
  }
}

export function formatSimulationPhase(phase: PublicGamePhase): string {
  return phaseLabels[phase];
}

export function formatSimulationTrump(
  simulation: RunAutomatedSimulationResponse
): string {
  const contract = findContract(simulation);
  return contract === null ? "未確定" : suitSymbols[contract.trumpSuit];
}

export function formatSimulationContractTarget(
  result: PublicGameResult
): string {
  if (result.resultType === "all-pass") {
    return "全員パス";
  }

  return `${result.targetPointCards}枚`;
}

export function getSimulationPlayerRole(
  playerId: string,
  result: PublicGameResult
): string {
  if (result.resultType === "all-pass") {
    return playerId === result.starterPlayerId ? "親" : "全員パス";
  }

  if (playerId === result.napoleonPlayerId) {
    return "ナポレオン";
  }

  if (playerId === result.adjutantPlayerId) {
    return "副官";
  }

  return "連合軍";
}

export function createSimulationFilename(seed: number): string {
  return `napoleon-simulation-seed-${seed}.json`;
}

export function formatSimulationWinner(result: PublicGameResult): string {
  if (result.resultType === "all-pass") {
    return "全員パス";
  }

  return formatWinningTeam(result.winner);
}

export function formatHandCounts(decision: PublicSimulationDecision): string {
  return Object.entries(decision.handCounts)
    .map(([playerId, count]) => `${playerId}:${count}`)
    .join(" / ");
}

export function validateSimulationSeedInput(seedInput: string): number | undefined {
  if (!/^\d+$/.test(seedInput.trim())) {
    return undefined;
  }

  const seed = Number(seedInput);

  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) {
    return undefined;
  }

  return seed;
}

export function formatSimulationLegalAction(action: PublicGameAction): string {
  switch (action.type) {
    case "bid":
      return `${suitLabels[action.suit]} ${action.targetPointCards}枚で入札`;
    case "pass":
      return "パス";
    case "choose-adjutant":
      return `${formatCardId(action.cardId)}を副官指定`;
    case "discard-cards":
      return `${action.cardIds.map(formatCardId).join("、")}を捨てる`;
    case "play-card":
      return `${formatCardId(action.cardId)}を出す`;
  }
}

export function formatSimulationBiddingStatus(
  observation: PublicSimulationObservation
): string {
  if (observation.bidding === null) {
    return "競りなし";
  }

  const highestBid = observation.bidding.highestBid;
  const highestBidText =
    highestBid === null
      ? "最高入札なし"
      : `${highestBid.playerId} ${suitLabels[highestBid.suit]} ${highestBid.targetPointCards}枚`;

  return `${highestBidText} / 履歴 ${observation.bidding.history.length}件`;
}

export function formatCompletedTrickStatus(
  observation: PublicSimulationObservation
): string {
  return `${observation.completedTrickCount}件完了 / 保存 ${observation.completedTricks.length}件`;
}

export function formatActualStateSummary(actualState: PublicSimulationActualState): string {
  const awardedCount = Object.values(actualState.awardedPointCardIds).reduce(
    (count, cardIds) => count + cardIds.length,
    0
  );

  return [
    `未使用 ${actualState.unusedCardIds.length}枚`,
    `除外 ${actualState.excludedCardIds.length}枚`,
    `埋札得点 ${awardedCount}枚`,
    `現在トリック ${actualState.currentTrickCardIds.length}枚`,
    `完了トリック ${actualState.completedTrickCardIds.length}枚`
  ].join(" / ");
}

function findContract(simulation: RunAutomatedSimulationResponse) {
  return simulation.decisions.find((decision) => decision.observation.contract !== null)
    ?.observation.contract ?? null;
}
