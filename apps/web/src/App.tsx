import { useMemo, useState } from "react";
import type {
  PublicBidAction,
  PublicCard,
  PublicGameAction,
  PublicGameState
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

export function App() {
  const [session, setSession] = useState<Session | undefined>();
  const [message, setMessage] = useState("ゲームを開始してください。");
  const [isBusy, setIsBusy] = useState(false);
  const [selectedDiscardCardIds, setSelectedDiscardCardIds] = useState<readonly string[]>([]);

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

  const self = session?.state.self;
  const otherPlayers = session?.state.opponents ?? [];

  async function handleCreateGame(): Promise<void> {
    await runRequest(async () => {
      const response = await createGame();
      setSession(response);
      setSelectedDiscardCardIds([]);
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
        <div className="opponents">
          {otherPlayers.map((player) => (
            <article
              className={`player-panel ${
                session?.state.currentPlayerId === player.id ? "current-player" : ""
              }`}
              key={player.id}
            >
              <h2>{player.id}</h2>
              <p>{player.handCount}枚</p>
            </article>
          ))}
        </div>

        <div className="center-area">
          <div className="status-line">
            <span>現在のプレイヤー: {session?.state.currentPlayerId ?? "-"}</span>
            <span>フェーズ: {formatPhase(session?.state.phase)}</span>
            <span>トリック: {session?.state.trickNumber ?? "-"}</span>
            <span>切り札: {formatTrumpSuit(session?.state.trumpSuit ?? null)}</span>
            <span>契約: {formatContract(session?.state ?? null)}</span>
          </div>

          {session?.state.phase === "bidding" ? (
            <section className="bidding-panel" aria-label="競り">
              <div className="bidding-summary">
                <span>競り開始: {session.state.bidding?.starterPlayerId ?? "-"}</span>
                <span>最高入札: {formatBid(session.state.bidding?.highestBid ?? null)}</span>
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
                        ? `${entry.playerId}: ${formatSuit(entry.suit)}${entry.targetPointCards}`
                        : `${entry.playerId}: パス`}
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

          <div className="trick" aria-label="中央の場">
            {session?.state.currentTrick.length ? (
              session.state.currentTrick.map((played) => (
                <div className="played-card" key={`${played.playerId}-${played.card.id}`}>
                  <span className="played-owner">{played.playerId}</span>
                  {played.card.type === "joker" ? (
                    <span className="joker-text">JOKER</span>
                  ) : (
                    <span className={isRedSuit(played.card.suit) ? "red-text" : "black-text"}>
                      {played.card.rank}
                      {suitSymbols[played.card.suit]}
                    </span>
                  )}
                </div>
              ))
            ) : (
              <div className="empty-trick">場にカードはありません。</div>
            )}
          </div>

          <button
            className="secondary-button"
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

        <article
          className={`self-panel ${
            session?.state.currentPlayerId === session?.playerId ? "current-player" : ""
          }`}
        >
          <div className="self-heading">
            <h2>あなた</h2>
            <span>{self?.handCount ?? 0}枚</span>
          </div>
          <div className="hand" aria-label="自分の手札">
            {self?.hand?.map((card) => (
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
                key={card.id}
                onPlay={handlePlay}
                selected={selectedDiscardCardIds.includes(card.id)}
              />
            ))}
          </div>
        </article>
      </section>
    </main>
  );
}

function createMessage(state: PublicGameState, playerId: string): string {
  if (state.isGameOver) {
    return "ゲーム終了です。";
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

  if (state.isTrickComplete) {
    return "5枚出ました。次のトリックへ進めます。";
  }

  if (state.currentPlayerId === playerId) {
    return "あなたの番です。カードを1枚選んでください。";
  }

  return `${state.currentPlayerId} の番です。`;
}

function formatTrumpSuit(trumpSuit: PublicGameState["trumpSuit"]): string {
  return trumpSuit === null ? "未定" : suitSymbols[trumpSuit];
}

function formatPhase(phase: PublicGameState["phase"] | undefined): string {
  switch (phase) {
    case "bidding":
      return "競り";
    case "exchanging":
      return "交換";
    case "playing":
      return "プレイ";
    case "finished":
      return "終了";
    default:
      return "-";
  }
}

function formatSuit(suit: PublicBidAction["suit"]): string {
  return suitSymbols[suit];
}

function formatBidAction(action: PublicBidAction): string {
  return `${formatSuit(action.suit)} ${action.targetPointCards}`;
}

function formatBid(bid: NonNullable<PublicGameState["bidding"]>["highestBid"]): string {
  return bid === null ? "なし" : `${bid.playerId} ${formatSuit(bid.suit)}${bid.targetPointCards}`;
}

function formatContract(state: PublicGameState | null): string {
  if (state?.contract === undefined || state.contract === null) {
    return "未確定";
  }

  return `${state.contract.napoleonPlayerId} ${formatSuit(state.contract.trumpSuit)}${state.contract.targetPointCards}`;
}
