import type { PublicCard, PublicPlayedCard, PublicSuit } from "@napoleon/protocol";
import { determineCurrentWinningPlayer } from "@napoleon/game-core";
import { isRedSuit, suitSymbols } from "./cardSymbols";
import type { TablePlayer } from "./tableTypes";

interface TrickBoardProps {
  players: readonly TablePlayer[];
  currentTrick: readonly PublicPlayedCard[];
  collectingWinnerId?: string;
  highlightWinningCard: boolean;
  isResultEmphasisActive?: boolean;
  trickNumber: number | undefined;
  trumpSuit: PublicSuit | null | undefined;
}

export function TrickBoard({
  collectingWinnerId,
  players,
  currentTrick,
  highlightWinningCard,
  isResultEmphasisActive = false,
  trickNumber,
  trumpSuit
}: TrickBoardProps) {
  const playedCardsByPlayerId = new Map(
    currentTrick.map((played) => [played.playerId, played] as const)
  );
  const winningPlayerId =
    highlightWinningCard && trumpSuit !== null && trumpSuit !== undefined
      ? getCurrentWinningPlayerId(currentTrick, trumpSuit, trickNumber)
      : undefined;
  const collectingWinner = players.find((player) => player.id === collectingWinnerId);
  const boardClassName = [
    "trick-board",
    isResultEmphasisActive ? "trick-board-result" : "",
    collectingWinner === undefined
      ? ""
      : `trick-board-collecting trick-board-collecting-to-${collectingWinner.seat}`
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={boardClassName} aria-label="中央の場">
      {players.map((player) => (
        <TrickSlot
          key={player.seat}
          isCollecting={collectingWinner !== undefined}
          player={player}
          played={playedCardsByPlayerId.get(player.id)}
          isWinning={winningPlayerId === player.id}
        />
      ))}
      <div className="trick-message" aria-label={`現在のトリックは${currentTrick.length}枚出ています`}>
        <strong>{currentTrick.length} / 5</strong>
      </div>
    </div>
  );
}

interface TrickSlotProps {
  player: TablePlayer;
  played: PublicPlayedCard | undefined;
  isCollecting: boolean;
  isWinning: boolean;
}

function TrickSlot({ player, played, isCollecting, isWinning }: TrickSlotProps) {
  const playedCardClassName = [
    "played-card",
    `played-card-from-${player.seat}`,
    isWinning ? "played-card-winning" : "",
    isCollecting ? "played-card-collecting" : ""
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={`trick-slot trick-${player.seat}`}>
      <span className="played-owner">{player.label}</span>
      {played === undefined ? (
        <div aria-label={`${player.label}は未プレイ`} className="empty-played-card" />
      ) : (
        <div
          aria-label={`${player.label}が${formatCardForAria(played.card)}を出しました${
            isWinning ? "。現在勝っています" : ""
          }`}
          className={playedCardClassName}
        >
          {played.card.type === "joker" ? (
            <span className="joker-text">JOKER</span>
          ) : (
            <span className={isRedSuit(played.card.suit) ? "red-text" : "black-text"}>
              {played.card.rank}
              {suitSymbols[played.card.suit]}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function getCurrentWinningPlayerId(
  currentTrick: readonly PublicPlayedCard[],
  trumpSuit: NonNullable<TrickBoardProps["trumpSuit"]>,
  trickNumber: number | undefined
): string | undefined {
  if (currentTrick.length === 0) {
    return undefined;
  }

  return determineCurrentWinningPlayer(currentTrick, { trumpSuit }, { trickNumber });
}

function formatCardForAria(card: PublicCard): string {
  if (card.type === "joker") {
    return "JOKER";
  }

  return `${card.rank}${suitSymbols[card.suit]}`;
}
