import type {
  PublicBidAction,
  PublicGameState,
  PublicRank,
  PublicStandardCard,
  PublicSuit
} from "@napoleon/protocol";
import { isRedSuit, suitSymbols } from "./cardSymbols";
import type { TablePlayer } from "./tableTypes";

export interface StatusChip {
  label: string;
  value?: string;
  ariaLabel?: string;
  tone?: "phase" | "contract" | "role" | "special" | "result";
  suitColor?: "red" | "black";
}

export interface GameStatusDisplay {
  primary: readonly StatusChip[];
  secondary: readonly StatusChip[];
}

export function createGameStatusDisplay(
  state: PublicGameState | undefined,
  players: readonly TablePlayer[]
): GameStatusDisplay {
  if (state === undefined) {
    return {
      primary: [],
      secondary: []
    };
  }

  switch (state.phase) {
    case "bidding":
      return {
        primary: [
          {
            label: "競り",
            ariaLabel: `競り。現在の手番は${formatPlayerLabel(state.currentPlayerId, players)}です。`,
            tone: "phase"
          }
        ],
        secondary: []
      };
    case "exchanging":
      return {
        primary: [
          { label: "交換", ariaLabel: "埋札交換", tone: "phase" },
          ...createContractSetupChips(state, players)
        ],
        secondary: createAdjutantChip(state, players)
      };
    case "choosing-adjutant":
      return {
        primary: [
          { label: "副官", ariaLabel: "副官指定", tone: "phase" },
          ...createContractSetupChips(state, players)
        ],
        secondary: []
      };
    case "playing":
      return {
        primary: createContractSummaryChips(state, players),
        secondary: createAdjutantChip(state, players)
      };
    case "finished":
      return {
        primary: [
          { label: "終了", ariaLabel: "ゲーム終了", tone: "phase" },
          ...(state.result === null
            ? []
            : [
                {
                  label: formatWinningTeamShort(state.result.winner),
                  ariaLabel: `勝者: ${formatWinningTeam(state.result.winner)}`,
                  tone: "result" as const
                }
              ]),
          ...createContractSummaryChips(state, players)
        ],
        secondary: createAdjutantChip(state, players)
      };
  }
}

export function createMessage(
  state: PublicGameState,
  playerId: string,
  players: readonly TablePlayer[]
): string {
  if (state.latestEvent?.type === "buried-cards-resolved") {
    const napoleonLabel = formatPlayerLabel(state.latestEvent.napoleonPlayerId, players);
    const pointCardText =
      state.latestEvent.awardedPointCards.length === 0
        ? "得点札はありません"
        : `${state.latestEvent.awardedPointCards.map(formatPublicStandardCard).join("、")}を${napoleonLabel}の得点札へ加算しました`;

    return `埋札を確定しました。${pointCardText}。非得点札は${state.latestEvent.hiddenNonPointCardCount}枚です。`;
  }

  if (state.isGameOver) {
    return state.result === null
      ? "ゲーム終了です。"
      : `ゲーム終了です。勝者: ${formatWinningTeam(state.result.winner)}`;
  }

  if (state.phase === "bidding") {
    return state.currentPlayerId === playerId
      ? "あなたの競り手番です。入札またはパスを選んでください。"
      : `${formatPlayerLabel(state.currentPlayerId, players)}の競り手番です。`;
  }

  if (state.phase === "exchanging") {
    return state.exchange?.napoleonPlayerId === playerId
      ? "埋札交換です。捨てるカードを3枚選んでください。"
      : `${formatPlayerLabel(state.exchange?.napoleonPlayerId ?? state.currentPlayerId, players)}が埋札交換中です。`;
  }

  if (state.phase === "choosing-adjutant") {
    return state.adjutantChoice?.napoleonPlayerId === playerId
      ? "副官として呼ぶカードを指定してください。"
      : `${formatPlayerLabel(state.adjutantChoice?.napoleonPlayerId ?? state.currentPlayerId, players)}が副官を指定中です。`;
  }

  if (state.isTrickComplete) {
    return "5枚出ました。次のトリックへ進めます。";
  }

  if (state.currentPlayerId === playerId) {
    return "あなたの番です。カードを1枚選んでください。";
  }

  return `${formatPlayerLabel(state.currentPlayerId, players)}の番です。`;
}

export function formatPlayerLabel(
  playerId: string | null | undefined,
  players: readonly TablePlayer[]
): string {
  if (playerId === null) {
    return "なし";
  }

  if (playerId === undefined) {
    return "-";
  }

  const player = players.find((candidate) => candidate.id === playerId);
  return player === undefined ? playerId : player.label;
}

export function formatContract(
  state: PublicGameState | null,
  players: readonly TablePlayer[]
): string {
  if (state?.contract === undefined || state.contract === null) {
    return "未確定";
  }

  return `${formatPlayerLabel(state.contract.napoleonPlayerId, players)} ${formatSuit(state.contract.trumpSuit)}${state.contract.targetPointCards}`;
}

export function formatAdjutant(
  adjutant: PublicGameState["adjutant"],
  players: readonly TablePlayer[]
): string {
  if (adjutant === null) {
    return "未指定";
  }

  const card = formatCardId(adjutant.calledCardId);
  const owner =
    adjutant.revealedPlayerId === null
      ? "未判明"
      : formatPlayerLabel(adjutant.revealedPlayerId, players);
  return `${card} / ${owner}`;
}

function formatAdjutantStatus(
  adjutant: PublicGameState["adjutant"],
  players: readonly TablePlayer[]
): string {
  if (adjutant === null) {
    return "未指定";
  }

  const card = formatCardId(adjutant.calledCardId);
  const owner =
    adjutant.revealedPlayerId === null
      ? "?"
      : formatPlayerLabel(adjutant.revealedPlayerId, players);
  return `${card}・${owner}`;
}

export function formatWinningTeam(
  winner: NonNullable<PublicGameState["result"]>["winner"]
): string {
  return winner === "napoleon-team" ? "ナポレオン陣営" : "連合軍";
}

function formatWinningTeamShort(
  winner: NonNullable<PublicGameState["result"]>["winner"]
): string {
  return winner === "napoleon-team" ? "N勝" : "連勝";
}

export function formatCardId(cardId: string): string {
  if (cardId === "joker") {
    return "ジョーカー";
  }

  const [suit, rank] = cardId.split("-");

  if (isPublicSuit(suit) && isPublicRank(rank)) {
    return `${rank}${suitSymbols[suit]}`;
  }

  return cardId;
}

function formatPublicStandardCard(card: PublicStandardCard): string {
  return `${card.rank}${suitSymbols[card.suit]}`;
}

function formatSuit(suit: PublicBidAction["suit"]): string {
  return suitSymbols[suit];
}

function createContractSetupChips(
  state: PublicGameState,
  players: readonly TablePlayer[]
): readonly StatusChip[] {
  if (state.contract === null) {
    return [];
  }

  return [
    createContractChip(state.contract, players)
  ];
}

function createContractSummaryChips(
  state: PublicGameState,
  players: readonly TablePlayer[]
): readonly StatusChip[] {
  if (state.contract === null) {
    return [];
  }

  return [
    createContractChip(state.contract, players)
  ];
}

function createContractChip(
  contract: NonNullable<PublicGameState["contract"]>,
  players: readonly TablePlayer[]
): StatusChip {
  return {
    label: formatSuit(contract.trumpSuit),
    value: String(contract.targetPointCards),
    ariaLabel: `契約: ナポレオンは${formatPlayerLabel(contract.napoleonPlayerId, players)}、切り札は${formatSuit(contract.trumpSuit)}、契約は${contract.targetPointCards}枚です。`,
    tone: "contract",
    suitColor: isRedSuit(contract.trumpSuit) ? "red" : "black"
  };
}

function createAdjutantChip(
  state: PublicGameState,
  players: readonly TablePlayer[]
): readonly StatusChip[] {
  if (state.adjutant === null) {
    return [];
  }

  return [
    {
      label: "副",
      value: formatAdjutantStatus(state.adjutant, players),
      ariaLabel: `副官札: ${formatAdjutant(state.adjutant, players)}`,
      tone: "role"
    }
  ];
}

function isPublicSuit(value: string | undefined): value is PublicSuit {
  return (
    value === "spades" ||
    value === "hearts" ||
    value === "diamonds" ||
    value === "clubs"
  );
}

const publicRanks: readonly PublicRank[] = [
  "A",
  "K",
  "Q",
  "J",
  "10",
  "9",
  "8",
  "7",
  "6",
  "5",
  "4",
  "3",
  "2"
];

function isPublicRank(value: string | undefined): value is PublicRank {
  return publicRanks.some((rank) => rank === value);
}
