import { useMemo, useState } from "react";
import type { PublicCard, PublicGameState, PublicLegalAction } from "@napoleon/protocol";
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

  const legalCardIds = useMemo(() => {
    const actions = session?.state.legalActions ?? [];
    return new Set(
      actions
        .filter((action): action is PublicLegalAction => {
          return action.type === "play-card";
        })
        .map((action) => action.cardId)
    );
  }, [session]);

  const self = session?.state.self;
  const otherPlayers = session?.state.opponents ?? [];

  async function handleCreateGame(): Promise<void> {
    await runRequest(async () => {
      const response = await createGame();
      setSession(response);
      setMessage("あなたの番です。カードを1枚選んでください。");
    });
  }

  async function handlePlay(card: PublicCard): Promise<void> {
    if (session === undefined) {
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
            <span>トリック: {session?.state.trickNumber ?? "-"}</span>
          </div>

          <div className="trick" aria-label="中央の場">
            {session?.state.currentTrick.length ? (
              session.state.currentTrick.map((played) => (
                <div className="played-card" key={`${played.playerId}-${played.card.id}`}>
                  <span className="played-owner">{played.playerId}</span>
                  <span className={isRedSuit(played.card.suit) ? "red-text" : "black-text"}>
                    {played.card.rank}
                    {suitSymbols[played.card.suit]}
                  </span>
                </div>
              ))
            ) : (
              <div className="empty-trick">場にカードはありません。</div>
            )}
          </div>

          <button
            className="secondary-button"
            disabled={!session?.state.isTrickComplete || session.state.isGameOver || isBusy}
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
                disabled={isBusy || !legalCardIds.has(card.id)}
                key={card.id}
                onPlay={handlePlay}
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

  if (state.isTrickComplete) {
    return "5枚出ました。次のトリックへ進めます。";
  }

  if (state.currentPlayerId === playerId) {
    return "あなたの番です。カードを1枚選んでください。";
  }

  return `${state.currentPlayerId} の番です。`;
}
