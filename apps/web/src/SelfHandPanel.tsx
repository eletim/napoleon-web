import { useMemo, useState } from "react";
import type { PublicCard, PublicGameState } from "@napoleon/protocol";
import { BiddingDeclarationBadge } from "./BiddingDeclarationBadge";
import { CardButton } from "./CardButton";
import { PointCards } from "./PointCards";
import { getDisplayedHandCards, type HandOrderMode } from "./handSorting";
import type { TablePlayer } from "./tableTypes";

interface SelfHandPanelProps {
  self: PublicGameState["self"] | undefined;
  selfPlayer: TablePlayer | undefined;
  state: PublicGameState | undefined;
  isBusy: boolean;
  legalCardIds: ReadonlySet<string>;
  selectedDiscardCardIds: readonly string[];
  canExchange: boolean;
  defaultHandOrderMode?: HandOrderMode;
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
  defaultHandOrderMode = "riipai",
  onPlay
}: SelfHandPanelProps) {
  const [handOrderMode, setHandOrderMode] = useState<HandOrderMode>(defaultHandOrderMode);
  const playerId = self?.id ?? selfPlayer?.id ?? "player-0";
  const capturedPointCards = self?.capturedPointCards ?? selfPlayer?.capturedPointCards ?? [];
  const isCurrent = state?.currentPlayerId === playerId;
  const isNapoleon = state?.contract?.napoleonPlayerId === playerId;
  const isAdjutant = state?.adjutant?.revealedPlayerId === playerId;
  const displayedHand = useMemo(
    () => getDisplayedHandCards(self?.hand ?? [], handOrderMode),
    [handOrderMode, self?.hand]
  );

  return (
    <article
      aria-label="自分"
      className={["self-panel", isCurrent ? "current-player" : ""].filter(Boolean).join(" ")}
    >
      <div className="self-heading">
        <div className="self-info">
          <h2>自分</h2>
          <BiddingDeclarationBadge
            playerLabel="自分"
            declaration={selfPlayer?.biddingDeclaration}
          />
        </div>

        <div className="role-badges self-role-badges">
          {isCurrent ? (
            <span aria-label="現在の手番" className="turn-dot" role="img">
              ▶
            </span>
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

        <div className="hand-sort-toggle" aria-label="理牌切り替え">
          <button
            aria-label={handOrderMode === "riipai" ? "理牌オン" : "理牌オフ"}
            aria-pressed={handOrderMode === "riipai"}
            className={getHandSortButtonClassName(handOrderMode)}
            onClick={() =>
              setHandOrderMode((current) => (current === "riipai" ? "original" : "riipai"))
            }
            type="button"
          >
            理
          </button>
        </div>
      </div>

      <div
        className="self-points-row"
        aria-label={`自分の獲得得点札は${capturedPointCards.length}枚`}
      >
        <span aria-hidden="true">★{capturedPointCards.length}</span>
        <div className="inline-cards compact-points">
          <PointCards cards={capturedPointCards} />
        </div>
      </div>

      <div className="hand" aria-label="自分の手札">
        {displayedHand.map((card) => {
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

function getHandSortButtonClassName(currentMode: HandOrderMode): string {
  return currentMode === "riipai"
    ? "hand-sort-button hand-sort-button-active"
    : "hand-sort-button";
}
