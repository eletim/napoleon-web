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
  const seatClassName = [
    "player-seat",
    `seat-${player.seat}`,
    isCurrent ? "current-player" : ""
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <article className={seatClassName}>
      <div className="seat-main-row">
        <div className="seat-title">
          <h2>{player.label}</h2>
          <span className="player-id">{player.id}</span>
        </div>
        <div className="compact-hand" aria-label={`${player.label}の手札は残り${player.handCount}枚`}>
          <span className="small-card-back" aria-hidden="true" />
          <strong>{player.handCount}枚</strong>
        </div>
      </div>

      <div className="role-badges">
        {isCurrent ? <span className="role-badge turn-badge">手番</span> : null}
        {isNapoleon ? <span className="role-badge napoleon-badge">ナポレオン</span> : null}
        {isAdjutant ? <span className="role-badge adjutant-badge">副官</span> : null}
      </div>

      <BiddingDeclarationBadge playerLabel={player.label} declaration={player.biddingDeclaration} />

      <div className="captured-compact">
        <span>得点札 {player.capturedPointCards.length}</span>
        <div className="inline-cards compact-points">
          <PointCards cards={player.capturedPointCards} />
        </div>
      </div>
    </article>
  );
}
