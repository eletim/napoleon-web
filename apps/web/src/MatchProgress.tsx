import type {
  PublicMatchPlayerFinalScore,
  PublicMatchPlayerProgress,
  PublicMatchState
} from "@napoleon/protocol";
import type { TablePlayer } from "./tableTypes";

interface MatchProgressProps {
  match: PublicMatchState;
  players: readonly TablePlayer[];
}

export function MatchProgress({ match, players }: MatchProgressProps) {
  const labels = new Map(players.map(({ id, label }) => [id, label]));

  return (
    <section className="match-progress" aria-label="試合進行">
      <div className="match-round-indicator" aria-label={`現在 第${match.currentRound}局 / 全${match.roundCount}局`}>
        <span>第{match.currentRound}局</span>
        <strong>{match.currentRound}/{match.roundCount}</strong>
      </div>
      <details className="match-score-details" open={match.completed}>
        <summary>{match.completed ? "試合結果" : "試合スコア"}</summary>
        <div className="match-score-scroll">
          <table>
            <thead>
              <tr>
                <th scope="col">プレイヤー</th>
                {Array.from({ length: match.completedRoundCount }, (_, index) => (
                  <th scope="col" key={index + 1}>{index + 1}局</th>
                ))}
                <th scope="col">素点計</th>
                {match.completed ? <th scope="col">順位</th> : null}
                {match.completed ? <th scope="col">ウマ</th> : null}
                {match.completed ? <th scope="col">最終値</th> : null}
              </tr>
            </thead>
            <tbody>
              {(match.completed ? match.finalScores ?? [] : match.players).map((player) => {
                const final = isFinalScore(player) ? player : undefined;
                return (
                  <tr key={player.playerId}>
                    <th scope="row">{labels.get(player.playerId) ?? player.playerId}</th>
                    {player.roundScores.map((score, index) => (
                      <td key={index}>{formatScore(score)}</td>
                    ))}
                    <td><strong>{formatScore(player.rawMatchScore)}</strong></td>
                    {final === undefined ? null : <td>{final.rank}位</td>}
                    {final === undefined ? null : <td>{formatScore(final.uma)}</td>}
                    {final === undefined ? null : <td><strong>{formatScore(final.finalValue)}</strong></td>}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {!match.completed ? (
          <p className="match-score-note">ウマ・最終値は5局終了後に確定します。</p>
        ) : null}
      </details>
    </section>
  );
}

function formatScore(score: number): string {
  return `${score > 0 ? "+" : ""}${Number.isInteger(score) ? score : score.toFixed(2)}`;
}

export function getMatchAdvanceLabel(match: PublicMatchState): string {
  return match.remainingRounds === 0 ? "試合結果へ" : "次局へ";
}

function isFinalScore(
  player: PublicMatchPlayerProgress
): player is PublicMatchPlayerFinalScore {
  return "finalValue" in player;
}
