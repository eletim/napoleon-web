import { useState } from "react";
import type { RunAutomatedSimulationResponse } from "@napoleon/protocol";
import { runAutomatedSimulation } from "./api";
import { formatCardId, formatPlayerLabel } from "./displayText";
import {
  createSimulationFilename,
  formatHandCounts,
  formatSimulationAction,
  formatSimulationContractTarget,
  formatSimulationPhase,
  formatSimulationTrump,
  formatSimulationWinner,
  getSimulationPlayerRole
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
    const seed = Number(seedInput);

    if (!Number.isFinite(seed) || !Number.isInteger(seed)) {
      setErrorMessage("seedは整数で入力してください。");
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
              onChange={(event) => setSeedInput(event.target.value)}
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
          <InitialHands simulation={simulation} />
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

function InitialHands({ simulation }: { simulation: RunAutomatedSimulationResponse }) {
  return (
    <section className="simulation-section" aria-label="初期配札">
      <h2>初期配札（教師ラベル・完全情報）</h2>
      <div className="simulation-hand-grid">
        {simulation.playerIds.map((playerId) => (
          <div className="simulation-hand-panel" key={playerId}>
            <h3>{playerId}</h3>
            <CardIdList cardIds={simulation.initialHands[playerId] ?? []} />
          </div>
        ))}
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
                <span>合法手 {decision.legalActionCount}</span>
                <span>{formatHandCounts(decision)}</span>
              </button>
              {expanded ? (
                <div className="simulation-decision-detail">
                  <h3>この時点の実手札（教師ラベル・完全情報）</h3>
                  <div className="simulation-hand-grid">
                    {simulation.playerIds.map((playerId) => (
                      <div className="simulation-hand-panel" key={`${decision.step}-${playerId}`}>
                        <h4>{playerId}</h4>
                        <CardIdList cardIds={decision.actualHands[playerId] ?? []} />
                      </div>
                    ))}
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
