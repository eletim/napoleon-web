import { useState } from "react";
import type { RunAutomatedSimulationResponse } from "@napoleon/protocol";
import { runAutomatedSimulation } from "./api";
import { formatCardId, formatPlayerLabel } from "./displayText";
import {
  createSimulationFilename,
  formatActualStateSummary,
  formatCompletedTrickStatus,
  formatHandCounts,
  formatSimulationBiddingStatus,
  formatSimulationAction,
  formatSimulationLegalAction,
  formatSimulationContractTarget,
  formatSimulationPhase,
  formatSimulationTrump,
  formatSimulationWinner,
  getSimulationPlayerRole,
  validateSimulationSeedInput
} from "./simulationDisplay";
import type { TablePlayer } from "./tableTypes";

const simulationPlayers: readonly TablePlayer[] = [
  {
    id: "player-0",
    label: "player-0",
    seat: "self",
    handCount: 0,
    capturedPointCards: [],
    isSelf: true
  },
  {
    id: "player-1",
    label: "player-1",
    seat: "left",
    handCount: 0,
    capturedPointCards: [],
    isSelf: false
  },
  {
    id: "player-2",
    label: "player-2",
    seat: "top-left",
    handCount: 0,
    capturedPointCards: [],
    isSelf: false
  },
  {
    id: "player-3",
    label: "player-3",
    seat: "top-right",
    handCount: 0,
    capturedPointCards: [],
    isSelf: false
  },
  {
    id: "player-4",
    label: "player-4",
    seat: "right",
    handCount: 0,
    capturedPointCards: [],
    isSelf: false
  }
];

export function AutomatedSimulationViewer() {
  const [seedInput, setSeedInput] = useState("1");
  const [simulation, setSimulation] = useState<RunAutomatedSimulationResponse | undefined>();
  const [expandedSteps, setExpandedSteps] = useState<ReadonlySet<number>>(() => new Set());
  const [isRunning, setIsRunning] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | undefined>();

  async function handleRunSimulation(): Promise<void> {
    const seed = validateSimulationSeedInput(seedInput);

    if (seed === undefined) {
      setErrorMessage("seedは0以上4294967295以下の整数で入力してください。");
      return;
    }

    setIsRunning(true);
    setErrorMessage(undefined);

    try {
      const response = await runAutomatedSimulation(seed);
      setSimulation(response);
      setExpandedSteps(new Set());
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "シミュレーションに失敗しました。");
    } finally {
      setIsRunning(false);
    }
  }

  function toggleStep(step: number): void {
    setExpandedSteps((current) => {
      const next = new Set(current);

      if (next.has(step)) {
        next.delete(step);
      } else {
        next.add(step);
      }

      return next;
    });
  }

  function downloadJson(): void {
    if (simulation === undefined) {
      return;
    }

    const blob = new Blob([JSON.stringify(simulation, null, 2)], {
      type: "application/json"
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = createSimulationFilename(simulation.seed);
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className="simulation-viewer" aria-label="AI対戦ログ">
      <div className="simulation-header">
        <div>
          <h1>AI対戦ログ</h1>
          <p>指定seedの5人AI対戦を実行し、判断データを確認します。</p>
        </div>
        <div className="simulation-controls">
          <label>
            seed
            <input
              disabled={isRunning}
              inputMode="numeric"
              max={4294967295}
              min={0}
              onChange={(event) => setSeedInput(event.target.value)}
              step={1}
              type="number"
              value={seedInput}
            />
          </label>
          <button
            className="primary-button"
            disabled={isRunning}
            onClick={() => void handleRunSimulation()}
            type="button"
          >
            {isRunning ? "実行中" : "シミュレーション実行"}
          </button>
          <button
            className="secondary-button"
            disabled={simulation === undefined || isRunning}
            onClick={downloadJson}
            type="button"
          >
            JSONダウンロード
          </button>
        </div>
      </div>

      {errorMessage === undefined ? null : (
        <p className="simulation-error" role="alert">
          {errorMessage}
        </p>
      )}

      {simulation === undefined ? (
        <div className="simulation-empty">実行結果はまだありません。</div>
      ) : (
        <>
          <SimulationOverview simulation={simulation} />
          <InitialCardLayout simulation={simulation} />
          <SimulationSummary simulation={simulation} />
          <DecisionTimeline
            expandedSteps={expandedSteps}
            onToggleStep={toggleStep}
            simulation={simulation}
          />
        </>
      )}
    </section>
  );
}

function SimulationOverview({ simulation }: { simulation: RunAutomatedSimulationResponse }) {
  const result = simulation.result;

  return (
    <section className="simulation-section" aria-label="概要">
      <h2>概要</h2>
      <div className="simulation-overview-grid">
        <Metric label="seed" value={String(simulation.seed)} />
        <Metric label="勝利陣営" value={formatSimulationWinner(result)} />
        <Metric label="契約枚数" value={formatSimulationContractTarget(result)} />
        <Metric label="切り札" value={formatSimulationTrump(simulation)} />
        <Metric label="ナポレオン" value={formatPlayerLabel(result.napoleonPlayerId, simulationPlayers)} />
        <Metric label="副官" value={formatPlayerLabel(result.adjutantPlayerId, simulationPlayers)} />
        <Metric label="ナポレオン陣営" value={`${result.napoleonTeamPointCards}枚`} />
        <Metric label="連合軍" value={`${result.alliancePointCards}枚`} />
        <Metric label="総判断回数" value={String(simulation.summary.totalDecisionCount)} />
      </div>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="simulation-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function InitialCardLayout({ simulation }: { simulation: RunAutomatedSimulationResponse }) {
  return (
    <section className="simulation-section" aria-label="初期カード配置">
      <h2>初期カード配置（教師ラベル・完全情報）</h2>
      <div className="simulation-hand-grid">
        {simulation.playerIds.map((playerId) => (
          <div className="simulation-hand-panel" key={playerId}>
            <h3>{playerId}</h3>
            <CardIdList cardIds={simulation.initialHands[playerId] ?? []} />
          </div>
        ))}
      </div>
      <div className="simulation-substate-grid">
        <div className="simulation-hand-panel">
          <h3>初期の未使用札</h3>
          <CardIdList cardIds={simulation.initialActualState.unusedCardIds} />
        </div>
      </div>
    </section>
  );
}

function SimulationSummary({ simulation }: { simulation: RunAutomatedSimulationResponse }) {
  return (
    <section className="simulation-section" aria-label="集計">
      <h2>集計</h2>
      <div className="simulation-summary-grid">
        <SummaryTable
          columns={["プレイヤーID", "判断回数", "役割"]}
          rows={simulation.playerIds.map((playerId) => [
            playerId,
            String(simulation.summary.decisionCountByPlayer[playerId] ?? 0),
            getSimulationPlayerRole(playerId, simulation.result)
          ])}
        />
        <SummaryTable
          columns={["フェーズ", "判断回数"]}
          rows={Object.entries(simulation.summary.decisionCountByPhase).map(([phase, count]) => [
            formatSimulationPhase(phase as keyof typeof simulation.summary.decisionCountByPhase),
            String(count)
          ])}
        />
        <SummaryTable
          columns={["行動種別", "回数"]}
          rows={["bid", "pass", "discard-cards", "choose-adjutant", "play-card"].map((actionType) => [
            actionType,
            String(simulation.summary.actionCountByType[actionType] ?? 0)
          ])}
        />
      </div>
    </section>
  );
}

function SummaryTable({
  columns,
  rows
}: {
  columns: readonly string[];
  rows: readonly (readonly string[])[];
}) {
  return (
    <div className="simulation-table-wrap">
      <table className="simulation-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.join(":")}>
              {row.map((cell, index) => (
                <td key={`${row[0]}-${index}`}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DecisionTimeline({
  expandedSteps,
  onToggleStep,
  simulation
}: {
  expandedSteps: ReadonlySet<number>;
  onToggleStep: (step: number) => void;
  simulation: RunAutomatedSimulationResponse;
}) {
  return (
    <section className="simulation-section" aria-label="判断タイムライン">
      <h2>判断タイムライン</h2>
      <div className="simulation-timeline">
        {simulation.decisions.map((decision) => {
          const expanded = expandedSteps.has(decision.step);

          return (
            <article className="simulation-decision" key={decision.step}>
              <button
                aria-expanded={expanded}
                className="simulation-decision-row"
                onClick={() => onToggleStep(decision.step)}
                type="button"
              >
                <span>#{decision.step}</span>
                <span>{formatSimulationPhase(decision.phase)}</span>
                <span>第{decision.trickNumber}トリック</span>
                <span>{formatSimulationAction(decision.playerId, decision.action)}</span>
                <span>合法手 {decision.legalActionCount}件</span>
                <span>{formatHandCounts(decision)}</span>
              </button>
              {expanded ? (
                <div className="simulation-decision-detail">
                  <h3>Agentが見ていた観測</h3>
                  <div className="simulation-observation-grid">
                    <Metric label="競り" value={formatSimulationBiddingStatus(decision.observation)} />
                    <Metric
                      label="完了トリック"
                      value={formatCompletedTrickStatus(decision.observation)}
                    />
                    <Metric
                      label="特殊札"
                      value={[
                        `オルマ ${formatCardId(decision.observation.specialCards.orumaCardId)}`,
                        `よろめき ${formatCardId(decision.observation.specialCards.yoromekiCardId)}`,
                        `正J ${
                          decision.observation.specialCards.seiJackCardId === null
                            ? "未確定"
                            : formatCardId(decision.observation.specialCards.seiJackCardId)
                        }`,
                        `裏J ${
                          decision.observation.specialCards.uraJackCardId === null
                            ? "未確定"
                            : formatCardId(decision.observation.specialCards.uraJackCardId)
                        }`
                      ].join(" / ")}
                    />
                    <Metric
                      label="副官"
                      value={
                        decision.observation.adjutant === null
                          ? "未指定"
                          : `${formatCardId(decision.observation.adjutant.calledCardId)} / ${
                              decision.observation.adjutant.revealedPlayerId ?? "未公開"
                            }`
                      }
                    />
                  </div>
                  <div className="simulation-substate-grid">
                    <div className="simulation-hand-panel">
                      <h4>現在トリック</h4>
                      <CardIdList
                        cardIds={decision.observation.currentTrick.map((played) => played.card.id)}
                      />
                    </div>
                    <div className="simulation-hand-panel">
                      <h4>合法手一覧</h4>
                      <ul className="simulation-action-list">
                        {decision.legalActions.map((action, index) => (
                          <li key={`${decision.step}-legal-${index}`}>
                            {formatSimulationLegalAction(action)}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                  <h3>この時点の完全情報（教師ラベル）</h3>
                  <p className="simulation-state-summary">
                    {formatActualStateSummary(decision.actualState)}
                  </p>
                  <div className="simulation-hand-grid">
                    {simulation.playerIds.map((playerId) => (
                      <div className="simulation-hand-panel" key={`${decision.step}-${playerId}`}>
                        <h4>{playerId}</h4>
                        <CardIdList cardIds={decision.actualState.hands[playerId] ?? []} />
                      </div>
                    ))}
                  </div>
                  <div className="simulation-substate-grid">
                    <div className="simulation-hand-panel">
                      <h4>未使用札</h4>
                      <CardIdList cardIds={decision.actualState.unusedCardIds} />
                    </div>
                    <div className="simulation-hand-panel">
                      <h4>除外札</h4>
                      <CardIdList cardIds={decision.actualState.excludedCardIds} />
                    </div>
                    <div className="simulation-hand-panel">
                      <h4>埋札処理で獲得扱いになった得点札</h4>
                      {simulation.playerIds.map((playerId) => (
                        <div className="simulation-player-card-row" key={`${decision.step}-award-${playerId}`}>
                          <strong>{playerId}</strong>
                          <CardIdList cardIds={decision.actualState.awardedPointCardIds[playerId] ?? []} />
                        </div>
                      ))}
                    </div>
                    <div className="simulation-hand-panel">
                      <h4>現在トリック</h4>
                      <CardIdList cardIds={decision.actualState.currentTrickCardIds} />
                    </div>
                    <div className="simulation-hand-panel">
                      <h4>完了済みトリック</h4>
                      <p>{decision.actualState.completedTrickCardIds.length}枚</p>
                    </div>
                  </div>
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
    </section>
  );
}

function CardIdList({ cardIds }: { cardIds: readonly string[] }) {
  return (
    <div className="simulation-card-list">
      {cardIds.map((cardId) => (
        <span className="mini-card" key={cardId}>
          {formatCardId(cardId)}
        </span>
      ))}
    </div>
  );
}
