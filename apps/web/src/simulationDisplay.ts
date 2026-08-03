import type {
  PublicGameAction,
  PublicGamePhase,
  PublicGameResult,
  PublicSimulationDecision,
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
  return `${result.targetPointCards}枚`;
}

export function getSimulationPlayerRole(
  playerId: string,
  result: PublicGameResult
): string {
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
  return formatWinningTeam(result.winner);
}

export function formatHandCounts(decision: PublicSimulationDecision): string {
  return Object.entries(decision.handCounts)
    .map(([playerId, count]) => `${playerId}:${count}`)
    .join(" / ");
}

function findContract(simulation: RunAutomatedSimulationResponse) {
  return simulation.decisions.find((decision) => decision.observation.contract !== null)
    ?.observation.contract ?? null;
}
