import type { PublicGameResult, PublicMatchState } from "@napoleon/protocol";
import { formatMatchValue } from "./MatchFinalResults";
import { formatPlayerLabel, formatWinningTeam } from "./displayText";
import { getMatchAdvanceLabel } from "./MatchProgress";
import type { TablePlayer } from "./tableTypes";

interface RoundResultPanelProps {
  disabled: boolean;
  match: PublicMatchState | undefined;
  onAdvanceMatch: () => void;
  players: readonly TablePlayer[];
  result: PublicGameResult;
  selfPlayerId: string | undefined;
}

/**
 * The one-round result screen: leads with which side won, then who held
 * which role, then a today-vs-cumulative score comparison per player, and
 * finally the round/match progress and advance action - in that priority
 * order, without long prose.
 */
export function RoundResultPanel({
  disabled,
  match,
  onAdvanceMatch,
  players,
  result,
  selfPlayerId
}: RoundResultPanelProps) {
  return (
    <section className="round-result-panel" aria-label="ゲーム結果">
      <h2 className="visually-hidden">ゲーム終了</h2>

      {result.resultType === "all-pass" ? (
        <p className="round-result-winner round-result-winner-draw">流局(全員パス)</p>
      ) : (
        <p
          className={
            result.winner === "napoleon-team"
              ? "round-result-winner round-result-winner-napoleon"
              : "round-result-winner round-result-winner-alliance"
          }
        >
          {formatWinningTeam(result.winner)}の勝利
        </p>
      )}

      {result.resultType === "all-pass" ? (
        <p className="round-result-roles">
          親 {formatPlayerLabel(result.starterPlayerId, players)}のみ+1、他は-1
        </p>
      ) : (
        <>
          <p className="round-result-roles">
            <span className="round-result-role-chip round-result-role-napoleon">
              <span aria-hidden="true">♛</span>
              ナポレオン {formatPlayerLabel(result.napoleonPlayerId, players)}
            </span>
            <span className="round-result-role-chip round-result-role-adjutant">
              <span aria-hidden="true">★</span>
              {result.adjutantPlayerId === null
                ? "ソロ"
                : `副官 ${formatPlayerLabel(result.adjutantPlayerId, players)}`}
            </span>
          </p>
          <p className="round-result-detail">
            契約{result.targetPointCards}枚(ナポレオン陣営{result.napoleonTeamPointCards}
            ・連合軍{result.alliancePointCards})
          </p>
        </>
      )}

      {match !== undefined ? (
        <>
          <p className="round-result-progress">
            第{match.currentRound}局 / 全{match.roundCount}局
          </p>
          <table className="round-result-scores">
            <caption className="visually-hidden">今回の得点と累積得点</caption>
            <thead>
              <tr>
                <th scope="col">プレイヤー</th>
                <th scope="col">今回</th>
                <th scope="col">累積</th>
              </tr>
            </thead>
            <tbody>
              {match.players.map((player) => (
                <tr
                  className={
                    player.playerId === selfPlayerId ? "round-result-scores-self" : undefined
                  }
                  key={player.playerId}
                >
                  <th scope="row">{formatPlayerLabel(player.playerId, players)}</th>
                  <td>{formatMatchValue(player.roundScores.at(-1) ?? 0)}</td>
                  <td>{formatMatchValue(player.rawMatchScore)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : null}

      {match === undefined || match.completed ? null : (
        <button
          className="primary-button match-advance-button"
          disabled={disabled}
          onClick={onAdvanceMatch}
          type="button"
        >
          {getMatchAdvanceLabel(match)}
        </button>
      )}
    </section>
  );
}
