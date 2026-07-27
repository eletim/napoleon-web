import type { PublicCard, PublicPlayedCard } from "@napoleon/protocol";
import { isRedSuit, suitSymbols } from "./cardSymbols";
import type { TablePlayer } from "./tableTypes";

interface TrickBoardProps {
  players: readonly TablePlayer[];
  currentTrick: readonly PublicPlayedCard[];
}

export function TrickBoard({ players, currentTrick }: TrickBoardProps) {
  const playedCardsByPlayerId = new Map(
    currentTrick.map((played) => [played.playerId, played] as const)
  );

  return (
    <div className="trick-board" aria-label="中央の場">
      {players.map((player) => (
        <TrickSlot
          key={player.seat}
          player={player}
          played={playedCardsByPlayerId.get(player.id)}
        />
      ))}
      <div className="trick-message">
        <span>現在のトリック</span>
        <strong>{currentTrick.length} / 5</strong>
      </div>
    </div>
  );
}

interface TrickSlotProps {
  player: TablePlayer;
  played: PublicPlayedCard | undefined;
}

function TrickSlot({ player, played }: TrickSlotProps) {
  return (
    <div className={`trick-slot trick-${player.seat}`}>
      <span className="played-owner">{player.label}</span>
      {played === undefined ? (
        <div aria-label={`${player.label}は未プレイ`} className="empty-played-card">
          未プレイ
        </div>
      ) : (
        <div
          aria-label={`${player.label}が${formatCardForAria(played.card)}を出しました`}
          className="played-card"
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

function formatCardForAria(card: PublicCard): string {
  if (card.type === "joker") {
    return "JOKER";
  }

  return `${card.rank}${suitSymbols[card.suit]}`;
}
