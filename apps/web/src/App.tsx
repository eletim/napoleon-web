import { useEffect, useMemo, useRef, useState } from "react";
import type {
  AiPolicyComposition,
  AiPreset,
  AiPresetId,
  PublicCard,
  PublicGameAction,
  PublicGameState,
  PublicMatchState,
  PublicRank,
  PublicSuit,
  PublicPhasePolicyRegistry
} from "@napoleon/protocol";
import {
  AiSettingsPanel,
  GamePresetSelector,
  isPresetCompositionAvailable
} from "./AiPresetControls";
import { AutomatedSimulationViewer } from "./AutomatedSimulationViewer";
import { CardDesignMock } from "./CardDesignMock";
import { TableSurface } from "./TableSurface";
import { TableDesignMock } from "./TableDesignMock";
import { hasCompletedMatchResult, MatchFinalResults } from "./MatchFinalResults";
import { getMatchAdvanceLabel } from "./MatchProgress";
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
import { advanceMatch, createGame, getAiPresets, nextTrick, sendAction, updateAiPreset } from "./api";
import { suitSymbols } from "./cardSymbols";
import { createMessage, formatPlayerLabel, formatWinningTeam } from "./displayText";
import { createTablePlayers } from "./tablePlayers";
import { useTrickAnimation } from "./useTrickAnimation";
import "./styles.css";

interface Session {
  gameId: string;
  playerId: string;
  state: PublicGameState;
  match?: PublicMatchState;
}

type AppMode = "game" | "simulation" | "ai-settings";

const defaultPresetId: AiPresetId = "com-ai";

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
  const [presets, setPresets] = useState<readonly AiPreset[]>([]);
  const [policyRegistry, setPolicyRegistry] = useState<PublicPhasePolicyRegistry>();
  const [selectedPresetId, setSelectedPresetId] = useState<AiPresetId>(defaultPresetId);
  const [presetDrafts, setPresetDrafts] =
    useState<Partial<Record<AiPresetId, AiPolicyComposition>>>({});
  const [settingsMessage, setSettingsMessage] = useState("設定を読み込んでいます。");
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
  const selectedPreset = presets.find((preset) => preset.id === selectedPresetId);
  const hasUnavailablePresetSelection = selectedPreset === undefined ||
    !isPresetCompositionAvailable(selectedPreset.composition, policyRegistry);
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

    void getAiPresets()
      .then((response) => {
        if (cancelled) {
          return;
        }

        setPresets(response.presets);
        setPolicyRegistry(response.policyRegistry);
        setPresetDrafts(Object.fromEntries(
          response.presets.map((preset) => [preset.id, { ...preset.composition }])
        ));
        setSelectedPresetId((current) =>
          response.presets.some(({ id }) => id === current)
            ? current
            : response.presets[0]?.id ?? defaultPresetId
        );
        setSettingsMessage("変更後はpresetごとに保存・適用してください。");
      })
      .catch((error) => {
        if (!cancelled) {
          const errorMessage = error instanceof Error
            ? error.message
            : "AI preset一覧を取得できませんでした。";
          setMessage(errorMessage);
          setSettingsMessage(errorMessage);
          setHasRequestError(true);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  async function handleCreateGame(): Promise<void> {
    await runRequest(async () => {
      const response = await createGame({ aiPresetId: selectedPresetId });
      setSession(response);
      setSelectedDiscardCardIds([]);
      setAdjutantSelection(defaultAdjutantSelection);
      setMessage(
        createMessage(response.state, response.playerId, createTablePlayers(response.state))
      );
    });
  }

  function handlePresetDraftChange(
    presetId: AiPresetId,
    phase: keyof AiPolicyComposition,
    policyId: string
  ): void {
    const current = presetDrafts[presetId] ?? presets.find(({ id }) => id === presetId)?.composition;
    if (current === undefined) {
      return;
    }
    setPresetDrafts((drafts) => ({
      ...drafts,
      [presetId]: { ...current, [phase]: policyId } as AiPolicyComposition
    }));
  }

  async function handleSavePreset(presetId: AiPresetId): Promise<void> {
    const composition = presetDrafts[presetId];
    if (composition === undefined) {
      return;
    }
    await runRequest(async () => {
      const saved = await updateAiPreset(presetId, composition);
      setPresets((current) => current.map((preset) =>
        preset.id === saved.id ? saved : preset
      ));
      setPresetDrafts((current) => ({ ...current, [saved.id]: saved.composition }));
      setSettingsMessage(`${saved.displayName}を保存・適用しました。`);
    });
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

  async function handleAdvanceMatch(): Promise<void> {
    if (session === undefined || session.match === undefined || isInteractionLocked) {
      return;
    }

    await runRequest(async () => {
      const response = await advanceMatch(session.gameId);
      setSession(response);
      setSelectedDiscardCardIds([]);
      setAdjutantSelection(defaultAdjutantSelection);
      setMessage(
        response.match.completed
          ? "5局が終了しました。試合結果を確認してください。"
          : `第${response.match.currentRound}局を開始します。`
      );
    });
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
      const errorMessage = error instanceof Error
        ? error.message
        : "予期しないエラーが発生しました。";
      setHasRequestError(true);
      setMessage(errorMessage);
      if (mode === "ai-settings") {
        setSettingsMessage(errorMessage);
      }
    } finally {
      requestInFlightRef.current = false;
      setIsBusy(false);
    }
  }

  const isStartedGame = session !== undefined && mode === "game";
  const isGameInProgress = isStartedGame && !session.state.isGameOver;
  const completedMatch = hasCompletedMatchResult(session?.match) ? session.match : undefined;
  const appShellClassName = [
    "app-shell",
    isStartedGame ? "app-shell-game-active" : "",
    isGameInProgress ? "app-shell-game-in-progress" : "",
    completedMatch !== undefined ? "app-shell-match-completed" : "",
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
          aria-label="AI設定"
          aria-pressed={mode === "ai-settings"}
          className={mode === "ai-settings" ? "mode-button mode-button-active" : "mode-button"}
          onClick={() => setMode("ai-settings")}
          type="button"
        >
          AI設定
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
              disabled={isInteractionLocked || hasUnavailablePresetSelection}
              onClick={handleCreateGame}
              type="button"
            >
              {session === undefined ? "ゲーム開始" : "新規"}
            </button>
          </section>

          {session === undefined ? (
            <GamePresetSelector
              disabled={isBusy || presets.length === 0}
              onChange={setSelectedPresetId}
              presets={presets}
              selectedPresetId={selectedPresetId}
            />
          ) : null}

          {completedMatch !== undefined ? (
            <MatchFinalResults
              disabled={isInteractionLocked || hasUnavailablePresetSelection}
              match={completedMatch}
              onStartNewMatch={() => void handleCreateGame()}
              players={tablePlayers}
            />
          ) : (
            <>
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
                        {session.state.result.resultType === "all-pass" ? (
                          <div className="result-grid">
                            <span>結果</span>
                            <strong>全員パス</strong>
                            <span>親</span>
                            <strong>
                              {formatPlayerLabel(
                                session.state.result.starterPlayerId,
                                tablePlayers
                              )}
                            </strong>
                            <span>親の報酬</span>
                            <strong>+1</strong>
                            <span>他プレイヤー</span>
                            <strong>-1</strong>
                          </div>
                        ) : (
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
                              {formatPlayerLabel(
                                session.state.result.napoleonPlayerId,
                                tablePlayers
                              )}
                            </strong>
                            <span>副官</span>
                            <strong>
                              {formatPlayerLabel(
                                session.state.result.adjutantPlayerId,
                                tablePlayers
                              )}
                            </strong>
                          </div>
                        )}
                        {session.match === undefined || session.match.completed ? null : (
                          <button
                            className="primary-button match-advance-button"
                            disabled={isInteractionLocked}
                            onClick={() => void handleAdvanceMatch()}
                            type="button"
                          >
                            {getMatchAdvanceLabel(session.match)}
                          </button>
                        )}
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
              match={session?.match}
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
          )}
        </>
      ) : mode === "simulation" ? (
        <AutomatedSimulationViewer />
      ) : (
        <AiSettingsPanel
          disabled={isBusy}
          drafts={presetDrafts}
          message={settingsMessage}
          onChange={handlePresetDraftChange}
          onSave={(presetId) => void handleSavePreset(presetId)}
          policyRegistry={policyRegistry}
          presets={presets}
        />
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
