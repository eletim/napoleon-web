import { useMemo, useState } from "react";
import type {
  PublicBidAction,
  PublicCard,
  PublicGameAction,
  PublicGameState,
  PublicRank,
  PublicSuit
} from "@napoleon/protocol";
import { AutomatedSimulationViewer } from "./AutomatedSimulationViewer";
import { BiddingPanel } from "./BiddingPanel";
import { GameStatus } from "./GameStatus";
import { PlayerSeat } from "./PlayerSeat";
import { SelfHandPanel } from "./SelfHandPanel";
import { TrickBoard } from "./TrickBoard";
import {
  createAdjutantCardId,
  createAdjutantSelectionLabel,
  createAdjutantShortcutOptions,
  defaultAdjutantSelection,
  selectAdjutantRank,
  selectAdjutantShortcut,
  selectAdjutantSuitOption,
  type AdjutantSelection,
  type AdjutantSuitOption
} from "./adjutantSelection";
import { createGame, nextTrick, sendAction } from "./api";
import { suitSymbols } from "./cardSymbols";
import {
  createGameStatusDisplay,
  createMessage,
  formatPlayerLabel,
  formatWinningTeam
} from "./displayText";
import { createTablePlayers } from "./tablePlayers";
import "./styles.css";

interface Session {
  gameId: string;
  playerId: string;
  state: PublicGameState;
}

type AppMode = "game" | "simulation";

export function App() {
  const [mode, setMode] = useState<AppMode>("game");
  const [session, setSession] = useState<Session | undefined>();
  const [message, setMessage] = useState("ゲームを開始してください。");
  const [isBusy, setIsBusy] = useState(false);
  const [selectedDiscardCardIds, setSelectedDiscardCardIds] = useState<readonly string[]>([]);
  const [adjutantSelection, setAdjutantSelection] =
    useState<AdjutantSelection>(defaultAdjutantSelection);

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
  const aiPlayers = tablePlayers.filter((player) => !player.isSelf);
  const selfPlayer = tablePlayers.find((player) => player.isSelf);
  const gameStatusDisplay = useMemo(
    () => createGameStatusDisplay(session?.state, tablePlayers),
    [session?.state, tablePlayers]
  );
  const adjutantShortcutOptions = useMemo(
    () => createAdjutantShortcutOptions(session?.state.specialCards),
    [session?.state.specialCards]
  );
  const selectedAdjutantCardId = createAdjutantCardId(adjutantSelection);
  const selectedAdjutantLabel = createAdjutantSelectionLabel(
    adjutantSelection,
    adjutantShortcutOptions
  );
  const canSelectJoker = session?.state.adjutantChoice?.jokerAllowed === true;

  async function handleCreateGame(): Promise<void> {
    await runRequest(async () => {
      const response = await createGame();
      setSession(response);
      setSelectedDiscardCardIds([]);
      setAdjutantSelection(defaultAdjutantSelection);
      setMessage(
        createMessage(response.state, response.playerId, createTablePlayers(response.state))
      );
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
      setMessage(
        createMessage(response.state, response.playerId, createTablePlayers(response.state))
      );
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
      if (action.type === "choose-adjutant") {
        setSelectedDiscardCardIds([]);
        setAdjutantSelection(defaultAdjutantSelection);
      }
      setMessage(
        createMessage(response.state, response.playerId, createTablePlayers(response.state))
      );
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
      setMessage(
        createMessage(response.state, response.playerId, createTablePlayers(response.state))
      );
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
      <nav className="mode-switch" aria-label="画面切り替え">
        <button
          aria-pressed={mode === "game"}
          className={mode === "game" ? "mode-button mode-button-active" : "mode-button"}
          onClick={() => setMode("game")}
          type="button"
        >
          通常プレイ
        </button>
        <button
          aria-pressed={mode === "simulation"}
          className={mode === "simulation" ? "mode-button mode-button-active" : "mode-button"}
          onClick={() => setMode("simulation")}
          type="button"
        >
          AI対戦ログ
        </button>
      </nav>

      {mode === "game" ? (
        <>
          <section className="top-bar">
            <div>
              <h1>Napoleon Web</h1>
              <p aria-live="polite">{message}</p>
            </div>
            <button
              className="primary-button"
              disabled={isBusy}
              onClick={handleCreateGame}
              type="button"
            >
              ゲーム開始
            </button>
          </section>

          <section className="table" aria-label="ゲームテーブル">
            <div className="table-grid">
              {aiPlayers.map((player) => (
                <PlayerSeat key={player.seat} player={player} state={session?.state} />
              ))}

              <div className="table-center">
                <GameStatus display={gameStatusDisplay} />

                {session?.state.phase === "bidding" ? (
                  <BiddingPanel
                    bidding={session.state.bidding}
                    canPass={canPass}
                    currentPlayerId={session.state.currentPlayerId}
                    formatPlayerLabel={(playerId) => formatPlayerLabel(playerId, tablePlayers)}
                    isBusy={isBusy}
                    legalBidActions={legalBidActions}
                    onBid={(action) => void handleSendAction(action)}
                    onPass={() => void handleSendAction({ type: "pass" })}
                    selfPlayerId={session.playerId}
                  />
                ) : (
                  <>
                    <TrickBoard
                      currentTrick={session?.state.currentTrick ?? []}
                      players={tablePlayers}
                    />

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
                  </>
                )}
              </div>
            </div>

            <div className="action-area">
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
                <p>副官として呼ぶカードを、埋札を見る前に1枚指定してください。</p>
              </div>
              <div className="adjutant-shortcuts" aria-label="特殊札ショートカット">
                <span>特殊札ショートカット</span>
                <div className="adjutant-shortcut-buttons">
                  {adjutantShortcutOptions.map((shortcut) => (
                    <button
                      aria-pressed={
                        adjutantSelection.shortcutId === shortcut.id &&
                        selectedAdjutantCardId === shortcut.cardId
                      }
                      className={
                        adjutantSelection.shortcutId === shortcut.id &&
                        selectedAdjutantCardId === shortcut.cardId
                          ? "adjutant-shortcut-button adjutant-shortcut-selected"
                          : "adjutant-shortcut-button"
                      }
                      disabled={!canChooseAdjutant || isBusy}
                      key={shortcut.id}
                      onClick={() =>
                        setAdjutantSelection((current) =>
                          selectAdjutantShortcut(current, shortcut)
                        )
                      }
                      type="button"
                    >
                      <span>{shortcut.label}</span>
                      <strong>{shortcut.display}</strong>
                    </button>
                  ))}
                </div>
              </div>
              <div className="adjutant-controls">
                <label>
                  カード
                  <select
                    disabled={!canChooseAdjutant || isBusy}
                    onChange={(event) =>
                      setAdjutantSelection((current) =>
                        selectAdjutantSuitOption(
                          current,
                          event.target.value as AdjutantSuitOption
                        )
                      )
                    }
                    value={adjutantSelection.suitOption}
                  >
                    {suitOptions.map((suit) => (
                      <option key={suit} value={suit}>
                        {suitSymbols[suit]}
                      </option>
                    ))}
                    {canSelectJoker ? <option value="joker">ジョーカー</option> : null}
                  </select>
                </label>
                <label>
                  ランク
                  <select
                    disabled={
                      !canChooseAdjutant ||
                      isBusy ||
                      adjutantSelection.suitOption === "joker"
                    }
                    onChange={(event) =>
                      setAdjutantSelection((current) =>
                        selectAdjutantRank(current, event.target.value as PublicRank)
                      )
                    }
                    value={adjutantSelection.rank}
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
                      cardId: selectedAdjutantCardId
                    })
                  }
                  type="button"
                >
                  副官を指定
                </button>
                <span className="selected-adjutant">
                  選択中: <strong>{selectedAdjutantLabel}</strong>
                </span>
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
                <span>ナポレオン</span>
                <strong>{formatPlayerLabel(session.state.result.napoleonPlayerId, tablePlayers)}</strong>
                <span>副官</span>
                <strong>{formatPlayerLabel(session.state.result.adjutantPlayerId, tablePlayers)}</strong>
              </div>
            </section>
          ) : null}
            </div>

            <SelfHandPanel
              canExchange={canExchange}
              isBusy={isBusy}
              legalCardIds={legalCardIds}
              onPlay={handlePlay}
              selectedDiscardCardIds={selectedDiscardCardIds}
              self={self}
              selfPlayer={selfPlayer}
              state={session?.state}
            />
          </section>
        </>
      ) : (
        <AutomatedSimulationViewer />
      )}
    </main>
  );
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
