import type { PublicCard, PublicGameState } from "@napoleon/protocol";
import { BiddingDeclarationBadge } from "./BiddingDeclarationBadge";
import { CardButton } from "./CardButton";
import { PointCards } from "./PointCards";
import type { TablePlayer } from "./tableTypes";

interface SelfHandPanelProps {
  self: PublicGameState["self"] | undefined;
  selfPlayer: TablePlayer | undefined;
  state: PublicGameState | undefined;
  isBusy: boolean;
  legalCardIds: ReadonlySet<string>;
  selectedDiscardCardIds: readonly string[];
  canExchange: boolean;
  onPlay: (card: PublicCard) => void;
}

export function SelfHandPanel({
  self,
  selfPlayer,
  state,
  isBusy,
  legalCardIds,
  selectedDiscardCardIds,
  canExchange,
  onPlay
}: SelfHandPanelProps) {
  const playerId = self?.id ?? selfPlayer?.id ?? "player-0";
  const capturedPointCards = self?.capturedPointCards ?? selfPlayer?.capturedPointCards ?? [];
  const isCurrent = state?.currentPlayerId === playerId;
  const isNapoleon = state?.contract?.napoleonPlayerId === playerId;
  const isAdjutant = state?.adjutant?.revealedPlayerId === playerId;

  return (
    <article className={["self-panel", isCurrent ? "current-player" : ""].filter(Boolean).join(" ")}>
      <div className="self-heading">
        <div className="self-info">
          <div>
            <h2>自分</h2>
            <span className="player-id">{playerId}</span>
          </div>
          <span className="self-hand-count">残り{self?.handCount ?? 0}枚</span>
          <span className="self-point-count">得点札{capturedPointCards.length}枚</span>
          <BiddingDeclarationBadge
            playerLabel="自分"
            declaration={selfPlayer?.biddingDeclaration}
          />
        </div>

        <div className="role-badges self-role-badges">
          {isCurrent ? <span className="role-badge turn-badge">手番</span> : null}
          {isNapoleon ? <span className="role-badge napoleon-badge">ナポレオン</span> : null}
          {isAdjutant ? <span className="role-badge adjutant-badge">副官</span> : null}
        </div>

        <div className="hand-legend" aria-label="手札凡例">
          <span className="legend-item legal">合法</span>
          <span className="legend-item blocked">不可</span>
          <span className="legend-item selectable">選択</span>
        </div>
      </div>

      <div className="self-points-row">
        <span>獲得得点札</span>
        <div className="inline-cards compact-points">
          <PointCards cards={capturedPointCards} />
        </div>
      </div>

      <div className="hand" aria-label="自分の手札">
        {self?.hand?.map((card) => {
          const interactionState = getCardInteractionState(card, state, legalCardIds, canExchange);

          return (
            <CardButton
              card={card}
              disabled={
                isBusy ||
                (state?.phase === "playing"
                  ? !legalCardIds.has(card.id)
                  : state?.phase === "exchanging"
                    ? !canExchange
                    : true)
              }
              interactionState={interactionState}
              key={card.id}
              onPlay={onPlay}
              selected={selectedDiscardCardIds.includes(card.id)}
            />
          );
        })}
      </div>
    </article>
  );
}

function getCardInteractionState(
  card: PublicCard,
  state: PublicGameState | undefined,
  legalCardIds: ReadonlySet<string>,
  canExchange: boolean
): "legal" | "blocked" | "selectable" {
  if (state?.phase === "exchanging" && canExchange) {
    return "selectable";
  }

  if (state?.phase === "playing" && legalCardIds.has(card.id)) {
    return "legal";
  }

  return "blocked";
}
