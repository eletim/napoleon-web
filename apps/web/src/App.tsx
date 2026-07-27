import { useMemo, useState } from "react";
import type {
  PublicBidAction,
  PublicCard,
  PublicGameAction,
  PublicGameState,
  PublicRank,
  PublicStandardCard,
  PublicSuit
} from "@napoleon/protocol";
import { BiddingPanel } from "./BiddingPanel";
import { PlayerSeat } from "./PlayerSeat";
import { SelfHandPanel } from "./SelfHandPanel";
import { TrickBoard } from "./TrickBoard";
import { createGame, nextTrick, sendAction } from "./api";
import { suitSymbols } from "./cardSymbols";
import { createTablePlayers } from "./tablePlayers";
import type { TablePlayer } from "./tableTypes";
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
  const aiPlayers = tablePlayers.filter((player) => !player.isSelf);
  const selfPlayer = tablePlayers.find((player) => player.isSelf);

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
          {aiPlayers.map((player) => (
            <PlayerSeat key={player.seat} player={player} state={session?.state} />
          ))}

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
    </main>
  );
}

function createMessage(state: PublicGameState, playerId: string): string {
  if (state.latestEvent?.type === "buried-cards-resolved") {
    const players = createTablePlayers(state);
    const napoleonLabel = formatPlayerLabel(state.latestEvent.napoleonPlayerId, players);
    const pointCardText =
      state.latestEvent.awardedPointCards.length === 0
        ? "得点札はありません"
        : `${state.latestEvent.awardedPointCards.map(formatPublicStandardCard).join("、")}を${napoleonLabel}の得点札へ加算しました`;

    return `埋札を確定しました。${pointCardText}。非得点札は${state.latestEvent.hiddenNonPointCardCount}枚です。`;
  }

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

function formatPublicStandardCard(card: PublicStandardCard): string {
  return `${card.rank}${suitSymbols[card.suit]}`;
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
