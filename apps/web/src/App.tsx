import { useMemo, useState, type ReactNode } from "react";
import type {
  PublicBidAction,
  PublicCard,
  PublicGameAction,
  PublicGameState,
  PublicRank,
  PublicStandardCard,
  PublicSuit
} from "@napoleon/protocol";
import { CardButton } from "./CardButton";
import { createGame, nextTrick, sendAction } from "./api";
import { isRedSuit, suitSymbols } from "./cardSymbols";
import "./styles.css";

interface Session {
  gameId: string;
  playerId: string;
  state: PublicGameState;
}

type Seat = "left" | "top-left" | "top-right" | "right" | "self";

interface TablePlayer {
  id: string;
  label: string;
  seat: Seat;
  handCount: number;
  capturedPointCards: readonly PublicStandardCard[];
  isSelf: boolean;
}

export function App() {
  const [session, setSession] = useState<Session | undefined>();
  const [message, setMessage] = useState("ゲームを開始してください。");
  const [isBusy, setIsBusy] = useState(false);
  const [selectedDiscardCardIds, setSelectedDiscardCardIds] = useState<readonly string[]>([]);
  const [selectedAdjutantSuit, setSelectedAdjutantSuit] = useState<PublicSuit>("spades");
  const [selectedAdjutantRank, setSelectedAdjutantRank] = useState<PublicRank>("A");

  const legalCardIds = useMemo(() => {
    const actions = session?.state.legalActions ?? [];
    return new Set(
      actions
        .filter((action) => action.type === "play-card")
        .map((action) => action.cardId)
    );
  }, [session]);
  const legalBidActions = useMemo(
    () =>
      (session?.state.legalActions ?? []).filter(
        (action): action is PublicBidAction => action.type === "bid"
      ),
    [session]
  );
  const canPass = (session?.state.legalActions ?? []).some((action) => action.type === "pass");
  const requiredDiscardCount = session?.state.exchange?.requiredDiscardCount ?? 3;
  const canExchange =
    session?.state.phase === "exchanging" &&
    session.state.exchange?.napoleonPlayerId === session.playerId;
  const canChooseAdjutant =
    session?.state.phase === "choosing-adjutant" &&
    session.state.adjutantChoice?.napoleonPlayerId === session.playerId;

  const self = session?.state.self;
  const tablePlayers = useMemo(() => createTablePlayers(session?.state), [session?.state]);
  const playedCardsByPlayerId = useMemo(() => {
    const entries =
      session?.state.currentTrick.map((played) => [played.playerId, played] as const) ?? [];
    return new Map(entries);
  }, [session?.state.currentTrick]);

  async function handleCreateGame(): Promise<void> {
    await runRequest(async () => {
      const response = await createGame();
      setSession(response);
      setSelectedDiscardCardIds([]);
      setSelectedAdjutantSuit("spades");
      setSelectedAdjutantRank("A");
      setMessage(createMessage(response.state, response.playerId));
    });
  }

  async function handlePlay(card: PublicCard): Promise<void> {
    if (session === undefined) {
      return;
    }

    if (session.state.phase === "exchanging") {
      toggleDiscardSelection(card.id);
      return;
    }

    await runRequest(async () => {
      const response = await sendAction(session.gameId, {
        type: "play-card",
        cardId: card.id
      });
      setSession(response);
      setMessage(createMessage(response.state, response.playerId));
    });
  }

  async function handleSendAction(action: PublicGameAction): Promise<void> {
    if (session === undefined) {
      return;
    }

    await runRequest(async () => {
      const response = await sendAction(session.gameId, action);
      setSession(response);
      if (action.type === "discard-cards") {
        setSelectedDiscardCardIds([]);
      }
      setMessage(createMessage(response.state, response.playerId));
    });
  }

  function toggleDiscardSelection(cardId: string): void {
    if (!canExchange) {
      return;
    }

    setSelectedDiscardCardIds((current) => {
      if (current.includes(cardId)) {
        return current.filter((selectedCardId) => selectedCardId !== cardId);
      }

      if (current.length >= requiredDiscardCount) {
        return current;
      }

      return [...current, cardId];
    });
  }

  async function handleNextTrick(): Promise<void> {
    if (session === undefined) {
      return;
    }

    await runRequest(async () => {
      const response = await nextTrick(session.gameId);
      setSession(response);
      setMessage(createMessage(response.state, response.playerId));
    });
  }

  async function runRequest(work: () => Promise<void>): Promise<void> {
    setIsBusy(true);
    setMessage("通信中です。");

    try {
      await work();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "予期しないエラーが発生しました。");
    } finally {
      setIsBusy(false);
    }
  }

  function renderPointCards(cards: readonly PublicStandardCard[]): ReactNode {
    if (cards.length === 0) {
      return <span className="muted-text">なし</span>;
    }

    return cards.map((card) => (
      <span
        className={isRedSuit(card.suit) ? "mini-card red-text" : "mini-card black-text"}
        key={card.id}
      >
        {formatStandardCard(card)}
      </span>
    ));
  }

  function renderPlayerSeat(player: TablePlayer): ReactNode {
    const isCurrent = session?.state.currentPlayerId === player.id;
    const isNapoleon = session?.state.contract?.napoleonPlayerId === player.id;
    const isAdjutant = session?.state.adjutant?.revealedPlayerId === player.id;
    const seatClassName = [
      "player-seat",
      `seat-${player.seat}`,
      isCurrent ? "current-player" : "",
      player.isSelf ? "self-seat" : ""
    ]
      .filter(Boolean)
      .join(" ");

    return (
      <article className={seatClassName} key={player.seat}>
        <div className="seat-header">
          <div>
            <h2>{player.label}</h2>
            <span className="player-id">{player.id}</span>
          </div>
          <div className="role-badges">
            {player.isSelf ? <span className="role-badge self-badge">自分</span> : null}
            {isCurrent ? <span className="role-badge turn-badge">手番</span> : null}
            {isNapoleon ? <span className="role-badge napoleon-badge">ナポレオン</span> : null}
            {isAdjutant ? <span className="role-badge adjutant-badge">副官</span> : null}
          </div>
        </div>

        {!player.isSelf ? (
          <div className="hidden-hand" aria-label={`${player.label}の手札`}>
            <div className="card-back-stack" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
            <strong>{player.handCount}枚</strong>
          </div>
        ) : (
          <div className="self-seat-count">
            <strong>{player.handCount}枚</strong>
          </div>
        )}

        <div className="captured-points">
          <div className="captured-heading">
            <span>獲得得点札</span>
            <strong>{player.capturedPointCards.length}</strong>
          </div>
          <div className="inline-cards">{renderPointCards(player.capturedPointCards)}</div>
        </div>
      </article>
    );
  }

  function renderTrickSlot(player: TablePlayer): ReactNode {
    const played = playedCardsByPlayerId.get(player.id);

    return (
      <div className={`trick-slot trick-${player.seat}`} key={player.seat}>
        <span className="played-owner">{player.label}</span>
        {played === undefined ? (
          <div className="empty-played-card">未プレイ</div>
        ) : (
          <div className="played-card">
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

  function getCardInteractionState(card: PublicCard): "legal" | "blocked" | "selectable" {
    if (session?.state.phase === "exchanging" && canExchange) {
      return "selectable";
    }

    if (session?.state.phase === "playing" && legalCardIds.has(card.id)) {
      return "legal";
    }

    return "blocked";
  }

  return (
    <main className="app-shell">
      <section className="top-bar">
        <div>
          <h1>Napoleon Web</h1>
          <p>{message}</p>
        </div>
        <button className="primary-button" disabled={isBusy} onClick={handleCreateGame} type="button">
          ゲーム開始
        </button>
      </section>

      <section className="table" aria-label="ゲームテーブル">
        <div className="table-grid">
          {tablePlayers.map(renderPlayerSeat)}

          <div className="table-center">
            <div className="status-line">
              <span>現在: {formatPlayerLabel(session?.state.currentPlayerId, tablePlayers)}</span>
              <span>フェーズ: {formatPhase(session?.state.phase)}</span>
              <span>トリック: {session?.state.trickNumber ?? "-"}</span>
              <span>切り札: {formatTrumpSuit(session?.state.trumpSuit ?? null)}</span>
              <span>契約: {formatContract(session?.state ?? null)}</span>
              <span>副官札: {formatAdjutant(session?.state.adjutant ?? null)}</span>
            </div>

            <div className="special-line">
              <span>オルマ: {formatCardId(session?.state.specialCards.orumaCardId ?? "")}</span>
              <span>よろめき: {formatCardId(session?.state.specialCards.yoromekiCardId ?? "")}</span>
              <span>正J: {formatOptionalCardId(session?.state.specialCards.seiJackCardId)}</span>
              <span>裏J: {formatOptionalCardId(session?.state.specialCards.uraJackCardId)}</span>
            </div>

            <div className="trick-board" aria-label="中央の場">
              {tablePlayers.map(renderTrickSlot)}
              <div className="trick-message">
                <span>現在のトリック</span>
                <strong>{session?.state.currentTrick.length ?? 0} / 5</strong>
              </div>
            </div>

            <button
              className="secondary-button next-trick-button"
              disabled={
                !session?.state.isTrickComplete ||
                session.state.isGameOver ||
                session.state.phase !== "playing" ||
                isBusy
              }
              onClick={handleNextTrick}
              type="button"
            >
              次のトリック
            </button>
          </div>
        </div>

        <div className="action-area">
          {session?.state.phase === "bidding" ? (
            <section className="bidding-panel" aria-label="競り">
              <div className="bidding-summary">
                <span>競り開始: {formatPlayerLabel(session.state.bidding?.starterPlayerId, tablePlayers)}</span>
                <span>最高入札: {formatBid(session.state.bidding?.highestBid ?? null, tablePlayers)}</span>
                <span>連続パス: {session.state.bidding?.consecutivePassCount ?? 0}</span>
              </div>
              <div className="bidding-actions">
                <button
                  className="secondary-button"
                  disabled={!canPass || isBusy}
                  onClick={() => void handleSendAction({ type: "pass" })}
                  type="button"
                >
                  パス
                </button>
                <div className="bid-buttons">
                  {legalBidActions.map((action) => (
                    <button
                      className="bid-button"
                      disabled={isBusy}
                      key={`${action.suit}-${action.targetPointCards}`}
                      onClick={() => void handleSendAction(action)}
                      type="button"
                    >
                      {formatBidAction(action)}
                    </button>
                  ))}
                </div>
              </div>
              <div className="bidding-history" aria-label="競り履歴">
                {(session.state.bidding?.history.length ?? 0) > 0 ? (
                  session.state.bidding?.history.map((entry, index) => (
                    <span key={`${entry.playerId}-${entry.type}-${index}`}>
                      {entry.type === "bid"
                        ? `${formatPlayerLabel(entry.playerId, tablePlayers)}: ${formatSuit(entry.suit)}${entry.targetPointCards}`
                        : `${formatPlayerLabel(entry.playerId, tablePlayers)}: パス`}
                    </span>
                  ))
                ) : (
                  <span>履歴はまだありません。</span>
                )}
              </div>
            </section>
          ) : null}

          {session?.state.phase === "exchanging" ? (
            <section className="exchange-panel" aria-label="埋札交換">
              <div>
                <h2>埋札交換</h2>
                <p>埋札3枚を受け取りました。捨てるカードを3枚選んでください。</p>
              </div>
              <span>
                選択中: {selectedDiscardCardIds.length} / {requiredDiscardCount}
              </span>
              <button
                className="secondary-button"
                disabled={
                  !canExchange ||
                  selectedDiscardCardIds.length !== requiredDiscardCount ||
                  isBusy
                }
                onClick={() =>
                  void handleSendAction({
                    type: "discard-cards",
                    cardIds: selectedDiscardCardIds
                  })
                }
                type="button"
              >
                3枚を捨てる
              </button>
            </section>
          ) : null}

          {session?.state.phase === "choosing-adjutant" ? (
            <section className="adjutant-panel" aria-label="副官指定">
              <div>
                <h2>副官指定</h2>
                <p>副官として呼ぶ通常カードを1枚指定してください。</p>
              </div>
              <div className="adjutant-controls">
                <label>
                  スート
                  <select
                    disabled={!canChooseAdjutant || isBusy}
                    onChange={(event) =>
                      setSelectedAdjutantSuit(event.target.value as PublicSuit)
                    }
                    value={selectedAdjutantSuit}
                  >
                    {suitOptions.map((suit) => (
                      <option key={suit} value={suit}>
                        {suitSymbols[suit]}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  ランク
                  <select
                    disabled={!canChooseAdjutant || isBusy}
                    onChange={(event) =>
                      setSelectedAdjutantRank(event.target.value as PublicRank)
                    }
                    value={selectedAdjutantRank}
                  >
                    {rankOptions.map((rank) => (
                      <option key={rank} value={rank}>
                        {rank}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  className="secondary-button"
                  disabled={!canChooseAdjutant || isBusy}
                  onClick={() =>
                    void handleSendAction({
                      type: "choose-adjutant",
                      cardId: `${selectedAdjutantSuit}-${selectedAdjutantRank}`
                    })
                  }
                  type="button"
                >
                  副官を指定
                </button>
              </div>
            </section>
          ) : null}

          {session?.state.buriedCards !== null && session?.state.buriedCards !== undefined ? (
            <section className="buried-panel" aria-label="埋札公開情報">
              <h2>埋札</h2>
              <div className="buried-row">
                <span>公開得点札:</span>
                <div className="inline-cards">
                  {renderPointCards(session.state.buriedCards.revealedPointCards)}
                </div>
              </div>
              <div className="buried-row">
                <span>非公開札:</span>
                <strong>{session.state.buriedCards.hiddenCardCount}枚</strong>
              </div>
            </section>
          ) : null}

          {session?.state.result !== null && session?.state.result !== undefined ? (
            <section className="result-panel" aria-label="ゲーム結果">
              <h2>ゲーム終了</h2>
              <div className="result-grid">
                <span>勝者</span>
                <strong>{formatWinningTeam(session.state.result.winner)}</strong>
                <span>契約</span>
                <strong>{session.state.result.targetPointCards}枚</strong>
                <span>ナポレオン陣営</span>
                <strong>{session.state.result.napoleonTeamPointCards}枚</strong>
                <span>連合軍</span>
                <strong>{session.state.result.alliancePointCards}枚</strong>
                <span>うち埋札得点札</span>
                <strong>{session.state.result.buriedPointCards}枚</strong>
                <span>ナポレオン</span>
                <strong>{formatPlayerLabel(session.state.result.napoleonPlayerId, tablePlayers)}</strong>
                <span>副官</span>
                <strong>{formatPlayerLabel(session.state.result.adjutantPlayerId, tablePlayers)}</strong>
              </div>
            </section>
          ) : null}
        </div>

        <article className="self-panel">
          <div className="self-heading">
            <div>
              <h2>自分の手札</h2>
              <span>{self?.handCount ?? 0}枚</span>
            </div>
            <div className="hand-legend" aria-label="手札凡例">
              <span className="legend-item legal">合法</span>
              <span className="legend-item blocked">不可</span>
              <span className="legend-item selectable">選択</span>
            </div>
          </div>
          <div className="hand" aria-label="自分の手札">
            {self?.hand?.map((card) => {
              const interactionState = getCardInteractionState(card);

              return (
                <CardButton
                  card={card}
                  disabled={
                    isBusy ||
                    (session?.state.phase === "playing"
                      ? !legalCardIds.has(card.id)
                      : session?.state.phase === "exchanging"
                        ? !canExchange
                        : true)
                  }
                  interactionState={interactionState}
                  key={card.id}
                  onPlay={handlePlay}
                  selected={selectedDiscardCardIds.includes(card.id)}
                />
              );
            })}
          </div>
        </article>
      </section>
    </main>
  );
}

function createMessage(state: PublicGameState, playerId: string): string {
  if (state.isGameOver) {
    return state.result === null
      ? "ゲーム終了です。"
      : `ゲーム終了です。勝者: ${formatWinningTeam(state.result.winner)}`;
  }

  if (state.phase === "bidding") {
    return state.currentPlayerId === playerId
      ? "あなたの競り手番です。入札またはパスを選んでください。"
      : `${state.currentPlayerId} の競り手番です。`;
  }

  if (state.phase === "exchanging") {
    return state.exchange?.napoleonPlayerId === playerId
      ? "埋札交換です。捨てるカードを3枚選んでください。"
      : `${state.exchange?.napoleonPlayerId ?? state.currentPlayerId} が埋札交換中です。`;
  }

  if (state.phase === "choosing-adjutant") {
    return state.adjutantChoice?.napoleonPlayerId === playerId
      ? "副官として呼ぶカードを指定してください。"
      : `${state.adjutantChoice?.napoleonPlayerId ?? state.currentPlayerId} が副官を指定中です。`;
  }

  if (state.isTrickComplete) {
    return "5枚出ました。次のトリックへ進めます。";
  }

  if (state.currentPlayerId === playerId) {
    return "あなたの番です。カードを1枚選んでください。";
  }

  return `${state.currentPlayerId} の番です。`;
}

function createTablePlayers(state: PublicGameState | undefined): ReadonlyArray<TablePlayer> {
  const opponents = state?.opponents ?? [];
  const seatDefinitions: ReadonlyArray<{ seat: Seat; label: string }> = [
    { seat: "left", label: "左側AI" },
    { seat: "top-left", label: "奥左AI" },
    { seat: "top-right", label: "奥右AI" },
    { seat: "right", label: "右側AI" }
  ];
  const opponentPlayers = seatDefinitions.map((definition, index) => {
    const player = opponents[index];

    return {
      id: player?.id ?? `player-${index + 1}`,
      label: definition.label,
      seat: definition.seat,
      handCount: player?.handCount ?? 0,
      capturedPointCards: player?.capturedPointCards ?? [],
      isSelf: false
    };
  });

  return [
    ...opponentPlayers,
    {
      id: state?.self.id ?? "player-0",
      label: "自分",
      seat: "self",
      handCount: state?.self.handCount ?? 0,
      capturedPointCards: state?.self.capturedPointCards ?? [],
      isSelf: true
    }
  ];
}

function formatPlayerLabel(
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

function formatTrumpSuit(trumpSuit: PublicGameState["trumpSuit"]): string {
  return trumpSuit === null ? "未定" : suitSymbols[trumpSuit];
}

function formatOptionalCardId(cardId: string | null | undefined): string {
  return cardId === null || cardId === undefined ? "未決定" : formatCardId(cardId);
}

function formatPhase(phase: PublicGameState["phase"] | undefined): string {
  switch (phase) {
    case "bidding":
      return "競り";
    case "exchanging":
      return "交換";
    case "choosing-adjutant":
      return "副官指定";
    case "playing":
      return "プレイ";
    case "finished":
      return "終了";
    default:
      return "-";
  }
}

const suitOptions: readonly PublicSuit[] = ["spades", "hearts", "diamonds", "clubs"];
const rankOptions: readonly PublicRank[] = [
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

function formatSuit(suit: PublicBidAction["suit"]): string {
  return suitSymbols[suit];
}

function formatBidAction(action: PublicBidAction): string {
  return `${formatSuit(action.suit)} ${action.targetPointCards}`;
}

function formatBid(
  bid: NonNullable<PublicGameState["bidding"]>["highestBid"],
  players: readonly TablePlayer[]
): string {
  return bid === null
    ? "なし"
    : `${formatPlayerLabel(bid.playerId, players)} ${formatSuit(bid.suit)}${bid.targetPointCards}`;
}

function formatContract(state: PublicGameState | null): string {
  if (state?.contract === undefined || state.contract === null) {
    return "未確定";
  }

  return `${state.contract.napoleonPlayerId} ${formatSuit(state.contract.trumpSuit)}${state.contract.targetPointCards}`;
}

function formatAdjutant(adjutant: PublicGameState["adjutant"]): string {
  if (adjutant === null) {
    return "未指定";
  }

  const card = formatCardId(adjutant.calledCardId);
  const owner = adjutant.revealedPlayerId === null ? "未判明" : adjutant.revealedPlayerId;
  return `${card} / ${owner}`;
}

function formatWinningTeam(winner: NonNullable<PublicGameState["result"]>["winner"]): string {
  return winner === "napoleon-team" ? "ナポレオン陣営" : "連合軍";
}

function formatStandardCard(card: Extract<PublicCard, { type: "standard" }>): string {
  return `${card.rank}${suitSymbols[card.suit]}`;
}

function formatCardId(cardId: string): string {
  const [suit, rank] = cardId.split("-");

  if (isPublicSuit(suit) && isPublicRank(rank)) {
    return `${rank}${suitSymbols[suit]}`;
  }

  return cardId;
}

function isPublicSuit(value: string | undefined): value is PublicSuit {
  return (
    value === "spades" ||
    value === "hearts" ||
    value === "diamonds" ||
    value === "clubs"
  );
}

function isPublicRank(value: string | undefined): value is PublicRank {
  return rankOptions.some((rank) => rank === value);
}
