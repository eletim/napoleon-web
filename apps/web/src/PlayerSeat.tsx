import type { PublicGameState } from "@napoleon/protocol";
import { BiddingDeclarationBadge } from "./BiddingDeclarationBadge";
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
  const seatAriaParts = [
    player.label,
    isCurrent ? "現在の手番" : undefined,
    isNapoleon ? "ナポレオン" : undefined,
    isAdjutant ? "副官" : undefined,
    `手札は残り${player.handCount}枚`
  ].filter((part): part is string => part !== undefined);
  const seatClassName = [
    "player-seat",
    `seat-${player.seat}`,
    isCurrent ? "current-player" : ""
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <article aria-label={seatAriaParts.join("、")} className={seatClassName}>
      <div className="seat-main-row">
        <div className="seat-title">
          <h2>{player.label}</h2>
        </div>
        <div className="compact-hand" aria-label={`${player.label}の手札は残り${player.handCount}枚`}>
          <span className="small-card-back" aria-hidden="true" />
          <strong>{player.handCount}</strong>
        </div>
      </div>

      <div className="role-badges">
        {isCurrent ? (
          <span aria-label="現在の手番" className="turn-dot" role="img" />
        ) : null}
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

      <BiddingDeclarationBadge playerLabel={player.label} declaration={player.biddingDeclaration} />

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
