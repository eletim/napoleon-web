import { useEffect, useMemo, useRef, useState } from "react";
import type {
  CreateGameAgentSelection,
  PublicAgentDescriptor,
  PublicCard,
  PublicGameAction,
  PublicGameState,
  PublicRank,
  PublicSuit
} from "@napoleon/protocol";
import { AutomatedSimulationViewer } from "./AutomatedSimulationViewer";
import { CardDesignMock } from "./CardDesignMock";
import { TableSurface } from "./TableSurface";
import { TableDesignMock } from "./TableDesignMock";
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
import { createGame, getAgents, nextTrick, sendAction } from "./api";
import { suitSymbols } from "./cardSymbols";
import { createMessage, formatPlayerLabel, formatWinningTeam } from "./displayText";
import { createTablePlayers } from "./tablePlayers";
import { useTrickAnimation } from "./useTrickAnimation";
import "./styles.css";

interface Session {
  gameId: string;
  playerId: string;
  state: PublicGameState;
}

type AppMode = "game" | "simulation";

const defaultAgentId = "rule-based";
const defaultAiPlayerIds = ["player-1", "player-2", "player-3", "player-4"] as const;
const fallbackAgents: readonly PublicAgentDescriptor[] = [
  {
    id: defaultAgentId,
    displayName: "Rule-based AI",
    isAvailable: true
  }
];
const defaultAgentSelections: Record<string, string> = Object.fromEntries(
  defaultAiPlayerIds.map((playerId) => [playerId, defaultAgentId])
);

export function App() {
  if (window.location.pathname.endsWith("/mock/card-design")) {
    return <CardDesignMock />;
  }

  if (window.location.pathname.endsWith("/mock/table-design-bidding")) {
    return <TableDesignMock variant="bidding" />;
  }

  if (window.location.pathname.endsWith("/mock/table-design-projected")) {
    return <TableDesignMock variant="projected" />;
  }

  if (window.location.pathname.endsWith("/mock/table-design-world")) {
    return <TableDesignMock variant="world" />;
  }

  if (window.location.pathname.endsWith("/mock/table-design")) {
    return <TableDesignMock />;
  }

  return <GameApp />;
}

function GameApp() {
  const [mode, setMode] = useState<AppMode>("game");
  const [session, setSession] = useState<Session | undefined>();
  const [message, setMessage] = useState("ゲームを開始してください。");
  const [isBusy, setIsBusy] = useState(false);
  const [agents, setAgents] = useState<readonly PublicAgentDescriptor[]>(fallbackAgents);
  const [agentSelections, setAgentSelections] =
    useState<Record<string, string>>(defaultAgentSelections);
  const [selectedDiscardCardIds, setSelectedDiscardCardIds] = useState<readonly string[]>([]);
  const [adjutantSelection, setAdjutantSelection] =
    useState<AdjutantSelection>(defaultAdjutantSelection);
  const [hasRequestError, setHasRequestError] = useState(false);
  const [winningCardHighlightEnabled, setWinningCardHighlightEnabled] = useState(true);
  const requestInFlightRef = useRef(false);

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
        (action) => action.type === "bid"
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

  const tablePlayers = useMemo(() => createTablePlayers(session?.state), [session?.state]);
  const aiPlayers = tablePlayers.filter((player) => !player.isSelf);
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
  const hasUnavailableAgentSelection = aiPlayers.some((player) => {
    const agentId = agentSelections[player.id] ?? defaultAgentId;
    const agent = agents.find((candidate) => candidate.id === agentId);

    return agent?.isAvailable !== true;
  });
  const hasActionPrompt =
    session !== undefined &&
    (session.state.currentPlayerId === session.playerId ||
      session.state.isTrickComplete ||
      canExchange ||
      canChooseAdjutant);
  const showVisibleMessage =
    session === undefined ||
    isBusy ||
    hasActionPrompt ||
    session.state.latestEvent?.type === "buried-cards-resolved" ||
    hasRequestError;
  const trickAnimation = useTrickAnimation({ state: session?.state });
  const isInteractionLocked = isBusy || trickAnimation.isAnimating;
  const hasTableActionPanel =
    session !== undefined &&
    session.state.phase !== "bidding" &&
    ((session.state.phase === "playing" &&
      session.state.isTrickComplete &&
      !session.state.isGameOver) ||
      session.state.phase === "exchanging" ||
      session.state.phase === "choosing-adjutant" ||
      session.state.result !== null);

  useEffect(() => {
    let cancelled = false;

    void getAgents()
      .then((response) => {
        if (cancelled) {
          return;
        }

        setAgents(response.agents.length > 0 ? response.agents : fallbackAgents);
        setAgentSelections((current) => normalizeAgentSelections(current, response.agents));
      })
      .catch((error) => {
        if (!cancelled) {
          setMessage(error instanceof Error ? error.message : "AI一覧を取得できませんでした。");
          setHasRequestError(true);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  async function handleCreateGame(): Promise<void> {
    await runRequest(async () => {
      const aiAgents: CreateGameAgentSelection[] = aiPlayers.map((player) => ({
        playerId: player.id,
        agentId: agentSelections[player.id] ?? defaultAgentId
      }));
      const response = await createGame({ aiAgents });
      setSession(response);
      setSelectedDiscardCardIds([]);
      setAdjutantSelection(defaultAdjutantSelection);
      setMessage(
        createMessage(response.state, response.playerId, createTablePlayers(response.state))
      );
    });
  }

  function handleAgentSelectionChange(playerId: string, agentId: string): void {
    setAgentSelections((current) => ({
      ...current,
      [playerId]: agentId
    }));
  }

  async function handlePlay(card: PublicCard): Promise<void> {
    if (session === undefined || isInteractionLocked) {
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
    if (session === undefined || isInteractionLocked) {
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
    if (!canExchange || isInteractionLocked) {
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
    if (session === undefined || isInteractionLocked) {
      return;
    }

    await trickAnimation.playCollectionBefore(() =>
      runRequest(async () => {
        const response = await nextTrick(session.gameId);
        setSession(response);
        setMessage(
          createMessage(response.state, response.playerId, createTablePlayers(response.state))
        );
      })
    );
  }

  async function runRequest(work: () => Promise<void>): Promise<void> {
    if (requestInFlightRef.current) {
      return;
    }

    requestInFlightRef.current = true;
    setIsBusy(true);
    setHasRequestError(false);
    setMessage("通信中です。");

    try {
      await work();
    } catch (error) {
      setHasRequestError(true);
      setMessage(error instanceof Error ? error.message : "予期しないエラーが発生しました。");
    } finally {
      requestInFlightRef.current = false;
      setIsBusy(false);
    }
  }

  const isStartedGame = session !== undefined && mode === "game";
  const isGameInProgress = isStartedGame && !session.state.isGameOver;
  const appShellClassName = [
    "app-shell",
    isStartedGame ? "app-shell-game-active" : "",
    isGameInProgress ? "app-shell-game-in-progress" : "",
    isGameInProgress ? `app-shell-phase-${session.state.phase}` : ""
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <main className={appShellClassName}>
      <nav
        className={
          session !== undefined && mode === "game"
            ? "mode-switch mode-switch-compact"
            : "mode-switch"
        }
        aria-label="画面切り替え"
      >
        <button
          aria-label={
            session !== undefined && mode === "game" ? "卓（通常プレイ）" : "通常プレイ"
          }
          aria-pressed={mode === "game"}
          className={mode === "game" ? "mode-button mode-button-active" : "mode-button"}
          onClick={() => setMode("game")}
          type="button"
        >
          {session !== undefined && mode === "game" ? "卓" : "通常プレイ"}
        </button>
        <button
          aria-label={session !== undefined && mode === "game" ? "ログ（AI対戦ログ）" : "AI対戦ログ"}
          aria-pressed={mode === "simulation"}
          className={mode === "simulation" ? "mode-button mode-button-active" : "mode-button"}
          onClick={() => setMode("simulation")}
          type="button"
        >
          {session !== undefined && mode === "game" ? "ログ" : "AI対戦ログ"}
        </button>
      </nav>

      {mode === "game" ? (
        <>
          <section
            className={[
              session === undefined ? "top-bar" : "top-bar top-bar-compact",
              isGameInProgress && showVisibleMessage ? "top-bar-alert" : ""
            ]
              .filter(Boolean)
              .join(" ")}
          >
            <div>
              <h1 aria-label="Napoleon Web">{session === undefined ? "Napoleon Web" : "NW"}</h1>
              <p aria-live="polite" className={showVisibleMessage ? undefined : "visually-hidden"}>
                {message}
              </p>
            </div>
            <button
              className="primary-button"
              disabled={isInteractionLocked || hasUnavailableAgentSelection}
              onClick={handleCreateGame}
              type="button"
            >
              {session === undefined ? "ゲーム開始" : "新規"}
            </button>
          </section>

          {session === undefined ? (
            <section className="agent-setup" aria-label="AI選択">
              <h2>AI選択</h2>
              <div className="agent-selector-grid">
                {aiPlayers.map((player) => (
                  <label className="agent-selector" key={player.id}>
                    <span>{player.label}</span>
                    <select
                      disabled={isBusy}
                      onChange={(event) =>
                        handleAgentSelectionChange(player.id, event.target.value)
                      }
                      value={agentSelections[player.id] ?? defaultAgentId}
                    >
                      {agents.map((agent) => (
                        <option disabled={!agent.isAvailable} key={agent.id} value={agent.id}>
                          {agent.displayName}
                          {agent.isAvailable ? "" : " (未設定)"}
                        </option>
                      ))}
                    </select>
                  </label>
                ))}
              </div>
            </section>
          ) : null}

          <section className="mobile-landscape-guide" aria-label="横向きプレイ案内">
            <strong>横向きでプレイしてください</strong>
            <span>スマートフォンを横にすると、5人卓と手札を見やすく表示します。</span>
          </section>

          <section className="table" aria-label="ゲームテーブル">
            <TableSurface
              actionPanel={
                hasTableActionPanel ? (
                  <div className="action-area">
                    {session?.state.phase === "playing" &&
                    session.state.isTrickComplete &&
                    !session.state.isGameOver ? (
                      <button
                        aria-label="次のトリックへ進む"
                        className="secondary-button next-trick-button"
                        disabled={isInteractionLocked}
                        onClick={handleNextTrick}
                        type="button"
                      >
                        次へ
                      </button>
                    ) : null}

                    {session?.state.phase === "exchanging" ? (
                      <section className="exchange-panel" aria-label="埋札交換">
                        <h2>交換</h2>
                        <span
                          aria-label={`選択中 ${selectedDiscardCardIds.length}枚、必要 ${requiredDiscardCount}枚`}
                        >
                          {selectedDiscardCardIds.length} / {requiredDiscardCount}
                        </span>
                        <button
                          aria-label={`${requiredDiscardCount}枚を捨てる`}
                          className="secondary-button"
                          disabled={
                            !canExchange ||
                            selectedDiscardCardIds.length !== requiredDiscardCount ||
                            isInteractionLocked
                          }
                          onClick={() =>
                            void handleSendAction({
                              type: "discard-cards",
                              cardIds: selectedDiscardCardIds
                            })
                          }
                          type="button"
                        >
                          捨てる
                        </button>
                      </section>
                    ) : null}

                    {session?.state.phase === "choosing-adjutant" ? (
                      <section
                        aria-describedby="adjutant-phase-note"
                        aria-label="副官指定"
                        className="adjutant-panel"
                      >
                        <div className="phase-title-row">
                          <h2>副官</h2>
                          <span aria-label="埋札前に1枚指定します" className="phase-count">
                            1枚
                          </span>
                        </div>
                        <p className="visually-hidden" id="adjutant-phase-note">
                          埋札前に1枚指定します。
                        </p>
                        <div className="adjutant-shortcuts" aria-label="特殊札ショートカット">
                          <span className="visually-hidden">特殊札ショートカット</span>
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
                                disabled={!canChooseAdjutant || isInteractionLocked}
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
                            札
                            <select
                              aria-label="副官に指定するカード種別"
                              disabled={!canChooseAdjutant || isInteractionLocked}
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
                            位
                            <select
                              aria-label="副官に指定するランク"
                              disabled={
                                !canChooseAdjutant ||
                                isInteractionLocked ||
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
                            aria-label="副官を指定"
                            className="secondary-button"
                            disabled={!canChooseAdjutant || isInteractionLocked}
                            onClick={() =>
                              void handleSendAction({
                                type: "choose-adjutant",
                                cardId: selectedAdjutantCardId
                              })
                            }
                            type="button"
                          >
                            指定
                          </button>
                          <span
                            aria-label={`選択中の副官札: ${selectedAdjutantLabel}`}
                            className="selected-adjutant"
                          >
                            <strong>{selectedAdjutantLabel}</strong>
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
                          <strong>
                            {formatPlayerLabel(session.state.result.napoleonPlayerId, tablePlayers)}
                          </strong>
                          <span>副官</span>
                          <strong>
                            {formatPlayerLabel(session.state.result.adjutantPlayerId, tablePlayers)}
                          </strong>
                        </div>
                      </section>
                    ) : null}
                  </div>
                ) : null
              }
              canExchange={canExchange}
              canPass={canPass}
              collectingWinnerId={trickAnimation.collectingWinnerId}
              currentTrick={trickAnimation.displayedTrick}
              highlightWinningCard={winningCardHighlightEnabled}
              isBusy={isInteractionLocked}
              isResultEmphasisActive={trickAnimation.isResultEmphasisActive}
              legalBidActions={legalBidActions}
              legalCardIds={legalCardIds}
              onBid={(action) => void handleSendAction(action)}
              onPass={() => void handleSendAction({ type: "pass" })}
              onToggleWinningCardHighlight={() =>
                setWinningCardHighlightEnabled((current) => !current)
              }
              onPlay={handlePlay}
              players={tablePlayers}
              selectedDiscardCardIds={selectedDiscardCardIds}
              selfPlayerId={session?.playerId}
              state={session?.state}
              trickNumber={session?.state.trickNumber}
              trumpSuit={session?.state.trumpSuit}
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

function normalizeAgentSelections(
  current: Record<string, string>,
  agents: readonly PublicAgentDescriptor[]
): Record<string, string> {
  const availableAgentIds = new Set(
    agents.filter((agent) => agent.isAvailable).map((agent) => agent.id)
  );
  const next = { ...current };

  for (const playerId of defaultAiPlayerIds) {
    if (!availableAgentIds.has(next[playerId])) {
      next[playerId] = defaultAgentId;
    }
  }

  return next;
}
