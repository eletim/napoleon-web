import type { PublicGameState } from "@napoleon/protocol";
import { PointCards } from "./PointCards";
import type { TablePlayer } from "./tableTypes";

interface PlayerSeatProps {
  player: TablePlayer;
  state: PublicGameState | undefined;
}

export function PlayerSeat({ player, state }: PlayerSeatProps) {
  const isCurrent = state?.currentPlayerId === player.id;
  const isNapoleon = state?.contract?.napoleonPlayerId === player.id;
  const isAdjutant = state?.adjutant?.revealedPlayerId === player.id;
  const hasRole = isNapoleon || isAdjutant;
  const seatClassName = [
    "player-seat",
    `seat-${player.seat}`,
    isCurrent ? "current-player" : ""
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <article aria-label={player.label} className={seatClassName}>
      <div className="seat-main-row">
        <div className="seat-title">
          <h2>{player.label}</h2>
        </div>
      </div>

      {hasRole ? (
        <div className="role-badges">
          {isNapoleon ? (
            <span aria-label="ナポレオン" className="role-badge napoleon-badge" role="img">
              N
            </span>
          ) : null}
          {isAdjutant ? (
            <span aria-label="副官" className="role-badge adjutant-badge" role="img">
              A
            </span>
          ) : null}
        </div>
      ) : null}

      <div
        className="captured-compact"
        aria-label={`${player.label}の獲得得点札は${player.capturedPointCards.length}枚`}
      >
        <span aria-hidden="true">★{player.capturedPointCards.length}</span>
        <div className="inline-cards compact-points">
          <PointCards cards={player.capturedPointCards} />
        </div>
      </div>
    </article>
  );
}
