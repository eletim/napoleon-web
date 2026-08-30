import type {
  PublicMatchPlayerFinalScore,
  PublicMatchState
} from "@napoleon/protocol";
import type { TablePlayer } from "./tableTypes";

interface MatchFinalResultsProps {
  disabled?: boolean;
  match: PublicMatchState;
  onStartNewMatch: () => void;
  players: readonly TablePlayer[];
}

export function MatchFinalResults({
  disabled = false,
  match,
  onStartNewMatch,
  players
}: MatchFinalResultsProps) {
  if (!hasCompletedMatchResult(match)) {
    return null;
  }

  const labels = new Map(players.map(({ id, label }) => [id, label]));
  const rankedPlayers = [...match.finalScores].sort((first, second) => first.rank - second.rank);

  return (
    <section className="match-final-results" aria-labelledby="match-final-results-title">
      <header className="match-final-header">
        <p>全{match.roundCount}局 終了</p>
        <h2 id="match-final-results-title">最終結果</h2>
      </header>

      <ol className="match-final-ranking" aria-label="最終順位">
        {rankedPlayers.map((player) => (
          <li className="match-final-player" key={player.playerId}>
            <div className="match-final-player-heading">
              <strong className="match-final-rank">{player.rank}位</strong>
              <span>{labels.get(player.playerId) ?? player.playerId}</span>
            </div>
            <dl className="match-final-values">
              <MatchFinalValue label="5局の素点合計" value={player.rawMatchScore} />
              <MatchFinalValue label="ウマ" value={player.uma} />
              <MatchFinalValue label="score" value={player.score} />
              <MatchFinalValue emphasized label="finalValue" value={player.finalValue} />
            </dl>
          </li>
        ))}
      </ol>

      <details className="match-final-round-details">
        <summary>局ごとの得点</summary>
        <div className="match-final-round-list">
          {rankedPlayers.map((player) => (
            <div className="match-final-round-player" key={player.playerId}>
              <strong>{labels.get(player.playerId) ?? player.playerId}</strong>
              <span>{player.roundScores.map(formatMatchValue).join(" / ")}</span>
            </div>
          ))}
        </div>
      </details>

      <button
        className="primary-button match-new-game-button"
        disabled={disabled}
        onClick={onStartNewMatch}
        type="button"
      >
        新しい試合を始める
      </button>
    </section>
  );
}

function MatchFinalValue({
  emphasized = false,
  label,
  value
}: {
  emphasized?: boolean;
  label: string;
  value: number;
}) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{emphasized ? <strong>{formatMatchValue(value)}</strong> : formatMatchValue(value)}</dd>
    </div>
  );
}

export function hasCompletedMatchResult(
  match: PublicMatchState | undefined
): match is PublicMatchState & {
  completed: true;
  finalScores: readonly PublicMatchPlayerFinalScore[];
} {
  return match?.completed === true && match.finalScores !== null;
}

export function formatMatchValue(value: number): string {
  const normalized = Object.is(value, -0) ? 0 : value;
  return `${normalized > 0 ? "+" : ""}${normalized}`;
}
