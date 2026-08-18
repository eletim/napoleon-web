import { useMemo, useState, type ReactNode } from "react";
import type {
  PublicCard,
  PublicGameState,
  PublicPlayedCard,
  PublicStandardCard,
  PublicSuit
} from "@napoleon/protocol";
import { determineCurrentWinningPlayer } from "@napoleon/game-core";
import { isRedSuit, suitSymbols } from "./cardSymbols";
import { CardButton } from "./CardButton";
import { getDisplayedHandCards, type HandOrderMode } from "./handSorting";
import type { TablePlayer } from "./tableTypes";

interface TableSurfaceProps {
  actionPanel: ReactNode;
  canExchange: boolean;
  collectingWinnerId?: string;
  currentTrick: readonly PublicPlayedCard[];
  highlightWinningCard: boolean;
  isBusy: boolean;
  isResultEmphasisActive?: boolean;
  legalCardIds: ReadonlySet<string>;
  onPlay: (card: PublicCard) => void;
  onToggleWinningCardHighlight: () => void;
  players: readonly TablePlayer[];
  selectedDiscardCardIds: readonly string[];
  state: PublicGameState | undefined;
  trickNumber: number | undefined;
  trumpSuit: PublicSuit | null | undefined;
}

export function TableSurface({
  actionPanel,
  canExchange,
  collectingWinnerId,
  currentTrick,
  highlightWinningCard,
  isBusy,
  isResultEmphasisActive = false,
  legalCardIds,
  onPlay,
  onToggleWinningCardHighlight,
  players,
  selectedDiscardCardIds,
  state,
  trickNumber,
  trumpSuit
}: TableSurfaceProps) {
  const [handOrderMode, setHandOrderMode] = useState<HandOrderMode>("riipai");
  const playedCardsByPlayerId = useMemo(
    () => new Map(currentTrick.map((played) => [played.playerId, played] as const)),
    [currentTrick]
  );
  const winningPlayerId =
    highlightWinningCard && trumpSuit !== null && trumpSuit !== undefined
      ? getCurrentWinningPlayerId(currentTrick, trumpSuit, trickNumber)
      : undefined;
  const collectingWinner = players.find((player) => player.id === collectingWinnerId);
  const surfaceClassName = [
    "table-surface",
    state?.phase === "bidding" ? "table-surface-bidding" : "table-surface-playing",
    isResultEmphasisActive ? "table-surface-result" : "",
    collectingWinner === undefined
      ? ""
      : `table-surface-collecting table-surface-collecting-to-${collectingWinner.seat}`
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={surfaceClassName}>
      <TableHud state={state} />
      <aside className="table-side-actions" aria-label="操作">
        {state !== undefined ? (
          <div className="table-side-tools" aria-label="補助操作">
            <button
              aria-label={handOrderMode === "riipai" ? "理牌オン" : "理牌オフ"}
              aria-pressed={handOrderMode === "riipai"}
              className={getSideToolButtonClassName(handOrderMode === "riipai")}
              onClick={() =>
                setHandOrderMode((current) => (current === "riipai" ? "original" : "riipai"))
              }
              type="button"
            >
              理牌
            </button>
            <button
              aria-label={highlightWinningCard ? "暫定勝ち札強調オン" : "暫定勝ち札強調オフ"}
              aria-pressed={highlightWinningCard}
              className={getSideToolButtonClassName(highlightWinningCard)}
              onClick={onToggleWinningCardHighlight}
              type="button"
            >
              勝札
            </button>
          </div>
        ) : null}
        {actionPanel}
      </aside>

      {players.map((player) => (
        <TableSeat
          canExchange={canExchange}
          handOrderMode={handOrderMode}
          isBusy={isBusy}
          isCollecting={collectingWinner !== undefined}
          isCurrent={state?.currentPlayerId === player.id}
          isWinning={winningPlayerId === player.id}
          key={player.seat}
          legalCardIds={legalCardIds}
          onPlay={onPlay}
          played={playedCardsByPlayerId.get(player.id)}
          player={player}
          selectedDiscardCardIds={selectedDiscardCardIds}
          state={state}
        />
      ))}

      <div className="table-core" aria-hidden="true">
        <div className="table-core-ring" />
      </div>
    </div>
  );
}

interface TableSeatProps {
  canExchange: boolean;
  handOrderMode: HandOrderMode;
  isBusy: boolean;
  isCollecting: boolean;
  isCurrent: boolean;
  isWinning: boolean;
  legalCardIds: ReadonlySet<string>;
  onPlay: (card: PublicCard) => void;
  played: PublicPlayedCard | undefined;
  player: TablePlayer;
  selectedDiscardCardIds: readonly string[];
  state: PublicGameState | undefined;
}

function TableSeat({
  canExchange,
  handOrderMode,
  isBusy,
  isCollecting,
  isCurrent,
  isWinning,
  legalCardIds,
  onPlay,
  played,
  player,
  selectedDiscardCardIds,
  state
}: TableSeatProps) {
  const capturedPointCards =
    player.isSelf ? (state?.self.capturedPointCards ?? player.capturedPointCards) : player.capturedPointCards;
  const seatClassName = [
    "table-player-zone",
    `table-player-${player.seat}`,
    isCurrent ? "table-player-current" : ""
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <section aria-label={player.label} className={seatClassName}>
      <div className="table-seat-guide" aria-hidden="true" />
      <div className="table-seat-name">
        <span>{player.label}</span>
      </div>
      <div className="table-hand-zone" aria-label={`${player.label}の手札領域`}>
        {player.isSelf ? (
          <SelfTableHand
            canExchange={canExchange}
            handOrderMode={handOrderMode}
            isBusy={isBusy}
            legalCardIds={legalCardIds}
            onPlay={onPlay}
            selectedDiscardCardIds={selectedDiscardCardIds}
            state={state}
          />
        ) : (
          <OpponentCardBacks count={player.handCount} playerLabel={player.label} />
        )}
      </div>
      <div className="table-trick-zone" aria-label={`${player.label}の現在トリック領域`}>
        <TrickCardSlot
          isCollecting={isCollecting}
          isWinning={isWinning}
          played={played}
          player={player}
        />
      </div>
      <div className="table-river-zone" aria-label={`${player.label}のポイント札の河`}>
        {state?.phase === "bidding" ? (
          <BiddingToken player={player} />
        ) : (
          <PointRiver cards={capturedPointCards} player={player} />
        )}
      </div>
      <RoleMarker player={player} state={state} />
    </section>
  );
}

function SelfTableHand({
  canExchange,
  handOrderMode,
  isBusy,
  legalCardIds,
  onPlay,
  selectedDiscardCardIds,
  state
}: {
  canExchange: boolean;
  handOrderMode: HandOrderMode;
  isBusy: boolean;
  legalCardIds: ReadonlySet<string>;
  onPlay: (card: PublicCard) => void;
  selectedDiscardCardIds: readonly string[];
  state: PublicGameState | undefined;
}) {
  const displayedHand = useMemo(
    () => getDisplayedHandCards(state?.self.hand ?? [], handOrderMode),
    [handOrderMode, state?.self.hand]
  );
  const shouldReserveHandSlots = state !== undefined && !state.isGameOver;
  const emptyHandSlotCount = shouldReserveHandSlots ? Math.max(0, 10 - displayedHand.length) : 0;

  return (
    <div className="table-self-hand" aria-label="自分の手札">
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
      {Array.from({ length: emptyHandSlotCount }, (_, index) => (
        <span
          aria-hidden="true"
          className="hand-card-empty-slot"
          key={`empty-hand-slot-${index}`}
        />
      ))}
    </div>
  );
}

function OpponentCardBacks({ count, playerLabel }: { count: number; playerLabel: string }) {
  return (
    <div className="opponent-card-backs" aria-label={`${playerLabel}の裏向き手札 ${count}枚`}>
      {Array.from({ length: count }, (_, index) => (
        <span aria-hidden="true" className="card-back" key={`${playerLabel}-back-${index}`} />
      ))}
    </div>
  );
}

function TrickCardSlot({
  isCollecting,
  isWinning,
  played,
  player
}: {
  isCollecting: boolean;
  isWinning: boolean;
  played: PublicPlayedCard | undefined;
  player: TablePlayer;
}) {
  if (played === undefined) {
    return <div aria-label={`${player.label}は未プレイ`} className="table-empty-card-slot" />;
  }

  return (
    <div
      aria-label={`${player.label}が${formatCardForAria(played.card)}を出しました${
        isWinning ? "。現在勝っています" : ""
      }`}
      className={[
        "table-card",
        "table-trick-card",
        `table-card-from-${player.seat}`,
        isWinning ? "table-card-winning" : "",
        isCollecting ? "table-card-collecting" : ""
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <CardFace card={played.card} />
    </div>
  );
}

function PointRiver({
  cards,
  player
}: {
  cards: readonly PublicStandardCard[];
  player: TablePlayer;
}) {
  return (
    <div
      className="point-river-grid"
      aria-label={`${player.label}の獲得ポイント札 ${cards.length}枚`}
    >
      {Array.from({ length: 20 }, (_, index) => {
        const card = cards[index];

        return card === undefined ? (
          <span
            aria-hidden="true"
            className="point-river-slot point-river-slot-empty"
            key={`river-empty-${player.id}-${index}`}
          />
        ) : (
          <span className="table-card table-point-card" key={card.id}>
            <CardFace card={card} />
          </span>
        );
      })}
    </div>
  );
}

function BiddingToken({ player }: { player: TablePlayer }) {
  const declaration = player.biddingDeclaration;
  const tokenClassName = [
    "table-bid-token",
    declaration?.type === "bid" && declaration.color === "red" ? "table-bid-token-red" : "",
    declaration?.type === "bid" && declaration.color === "black" ? "table-bid-token-black" : "",
    declaration?.type === "pass" ? "table-bid-token-pass" : "",
    declaration === undefined || declaration.type === "none" ? "table-bid-token-none" : ""
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={tokenClassName} aria-label={`${player.label}の競り宣言`}>
      {declaration?.type === "bid" ? (
        <>
          <span>{declaration.suit === undefined ? "" : suitSymbols[declaration.suit]}</span>
          <strong>{declaration.targetPointCards}</strong>
        </>
      ) : declaration?.type === "pass" ? (
        <strong>Pass</strong>
      ) : (
        <strong>?</strong>
      )}
    </div>
  );
}

function RoleMarker({
  player,
  state
}: {
  player: TablePlayer;
  state: PublicGameState | undefined;
}) {
  const isNapoleon = state?.contract?.napoleonPlayerId === player.id;
  const isAdjutant = state?.adjutant?.revealedPlayerId === player.id;
  const markerText =
    state === undefined || state.contract === null
      ? "?"
      : isNapoleon && isAdjutant
        ? "N/A"
        : isNapoleon
          ? "N"
          : isAdjutant
            ? "A"
            : "";

  return (
    <div
      aria-label={`${player.label}の役職${markerText === "" ? "なし" : markerText}`}
      className={[
        "table-role-marker",
        markerText === "" ? "table-role-marker-empty" : "",
        isNapoleon ? "table-role-marker-napoleon" : "",
        isAdjutant ? "table-role-marker-adjutant" : ""
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {markerText}
    </div>
  );
}

function TableHud({ state }: { state: PublicGameState | undefined }) {
  return (
    <div className="table-hud" aria-label="ゲーム全体情報">
      <strong className={createSuitTextClassName(state?.contract?.trumpSuit)}>
        {formatContractSummary(state?.contract)}
      </strong>
      <span>副官 {renderCalledCard(state?.adjutant?.calledCardId)}</span>
    </div>
  );
}

function CardFace({ card }: { card: PublicCard }) {
  if (card.type === "joker") {
    return (
      <>
        <span className="card-corner card-corner-top">JOKER</span>
        <span className="card-center-mark joker-text">Joker</span>
        <span className="card-corner card-corner-bottom">JOKER</span>
      </>
    );
  }

  return (
    <>
      <span className="card-corner card-corner-top">
        {card.rank}
        {suitSymbols[card.suit]}
      </span>
      <span className={isRedSuit(card.suit) ? "card-center-mark red-text" : "card-center-mark black-text"}>
        {suitSymbols[card.suit]}
      </span>
      <span className="card-corner card-corner-bottom">
        {card.rank}
        {suitSymbols[card.suit]}
      </span>
    </>
  );
}

function getCurrentWinningPlayerId(
  currentTrick: readonly PublicPlayedCard[],
  trumpSuit: NonNullable<TableSurfaceProps["trumpSuit"]>,
  trickNumber: number | undefined
): string | undefined {
  if (currentTrick.length === 0) {
    return undefined;
  }

  return determineCurrentWinningPlayer(currentTrick, { trumpSuit }, { trickNumber });
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

function getSideToolButtonClassName(enabled: boolean): string {
  return enabled ? "table-side-tool table-side-tool-active" : "table-side-tool";
}

function formatCardForAria(card: PublicCard): string {
  if (card.type === "joker") {
    return "JOKER";
  }

  return `${card.rank}${suitSymbols[card.suit]}`;
}

function formatContractSummary(contract: PublicGameState["contract"] | undefined): string {
  if (contract === undefined || contract === null) {
    return "-";
  }

  return `${suitSymbols[contract.trumpSuit]}${contract.targetPointCards}`;
}

function renderCalledCard(cardId: string | undefined) {
  if (cardId === undefined) {
    return "-";
  }

  const suit = getCalledCardSuit(cardId);
  const label = formatCalledCardId(cardId);

  if (suit !== undefined && isRedSuit(suit)) {
    return <span className="red-text">{label}</span>;
  }

  return label;
}

function formatCalledCardId(cardId: string): string {
  if (cardId === "joker") {
    return "Joker";
  }

  const [suit, rank] = cardId.split("-");

  if (isPublicSuit(suit) && rank !== undefined) {
    return `${suitSymbols[suit]}${rank}`;
  }

  return cardId;
}

function getCalledCardSuit(cardId: string): PublicSuit | undefined {
  const [suit] = cardId.split("-");

  return isPublicSuit(suit) ? suit : undefined;
}

function createSuitTextClassName(suit: PublicSuit | null | undefined): string {
  return suit !== null && suit !== undefined && isRedSuit(suit) ? "red-text" : "black-text";
}

function isPublicSuit(value: string | undefined): value is PublicSuit {
  return (
    value === "spades" ||
    value === "hearts" ||
    value === "diamonds" ||
    value === "clubs"
  );
}
