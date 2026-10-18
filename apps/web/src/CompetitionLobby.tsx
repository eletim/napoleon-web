import { useEffect, useState } from "react";
import type { CreateGameResponse } from "@napoleon/protocol";
import { cancelMatchmaking, getCpuOpponents, getGame, getMatchmaking, getMyRating, joinMatchmaking,
  startCpuMatch, getMatchHistory, type MatchHistoryEntry, type CpuEntry, type QueueState } from "./api";

export function CompetitionLobby({ onStart }: { onStart: (session: CreateGameResponse) => void }) {
  const [history, setHistory] = useState<MatchHistoryEntry[]>([]);
  const [historyOffset, setHistoryOffset] = useState(0);
  const [cpus, setCpus] = useState<CpuEntry[]>([]);
  const [cpuId, setCpuId] = useState("");
  const [rating, setRating] = useState<number>();
  const [queue, setQueue] = useState<QueueState>({ status: "idle" });
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let active = true; let timer: ReturnType<typeof setTimeout>;
    let started = false;
    async function poll() {
      try {
        const result = await getMatchmaking();
        if (!active) return;
        setQueue(result);
        if (result.status === "matched" && result.gameId && !started) {
          started = true;
          const match = await getGame(result.gameId);
          if (active) onStart(match);
          return;
        }
      } catch (error) { started = false; if (active) setMessage(error instanceof Error ? error.message : "通信に失敗しました。"); }
      if (active) timer = setTimeout(() => void poll(), 1000);
    }
    Promise.all([getCpuOpponents(), getMyRating()]).then(([result, me]) => {
      if (active) { setCpus(result.cpus); setCpuId(result.cpus[0]?.id ?? ""); setRating(me.rating); }
    }).catch((error) => { if (active) setMessage(String(error.message)); });
    void getMatchHistory().then((result) => { if (active) setHistory(result.matches); }).catch((error) => { if (active) setMessage(error.message); });
    void poll();
    return () => { active = false; clearTimeout(timer); };
  }, [onStart]);
  async function run(work: () => Promise<void>) {
    if (busy) return; setBusy(true); setMessage("");
    try { await work(); } catch (error) { setMessage(error instanceof Error ? error.message : "通信に失敗しました。"); }
    finally { setBusy(false); }
  }
  return <section className="competition-lobby" aria-label="レート対戦">
    <h2>レート対戦</h2><p>あなたのレート：{rating === undefined ? "読み込み中…" : Math.round(rating)}</p>
    <p>全5局の最終順位で、人間・CPU全員のレートが変動します。</p>
    <label>対戦・補充に使うCPU <select value={cpuId} disabled={busy || queue.status === "waiting"} onChange={(e) => setCpuId(e.target.value)}>
      {cpus.map((cpu) => <option key={cpu.id} value={cpu.id}>{cpu.displayName}（{Math.round(cpu.rating)}）</option>)}
    </select></label>
    {queue.status === "waiting" ? <>
      <p role="status">対戦相手を探しています… {Math.floor((queue.waitedMs ?? 0) / 1000)}秒（10秒で不足分をCPU補充）</p>
      <button disabled={busy} onClick={() => void run(async () => setQueue(await cancelMatchmaking()))}>キャンセル</button>
    </> : <div>
      <button disabled={busy || !cpuId} onClick={() => void run(async () => setQueue(await joinMatchmaking(cpuId)))}>マッチング開始</button>
      <button disabled={busy || !cpuId} onClick={() => void run(async () => onStart(await startCpuMatch(cpuId)))}>CPUとすぐ対戦</button>
    </div>}
    <details className="match-history"><summary>対戦履歴</summary>
      {history.length === 0 ? <p>まだ対戦履歴はありません。</p> : <table>
        <thead><tr><th>終了日時</th><th>順位</th><th>レート</th><th>対戦相手</th></tr></thead>
        <tbody>{history.map((entry) => <tr key={entry.matchId}><td>{new Date(entry.completedAt).toLocaleString()}</td>
          <td>{entry.rank}位</td><td>{Math.round(entry.before)} → {Math.round(entry.after)}（{entry.delta >= 0 ? "+" : ""}{entry.delta.toFixed(1)}）</td>
          <td>{entry.participants.map((p) => `${p.displayName}${p.kind === "cpu" ? " [CPU]" : ""} ${p.rank}位`).join(" / ")}</td></tr>)}</tbody>
      </table>}
      {history.length === 20 ? <button disabled={busy} onClick={() => void run(async () => {
        const offset = historyOffset + 20; const result = await getMatchHistory(offset);
        setHistory(result.matches); setHistoryOffset(offset);
      })}>次の20件</button> : null}
      {historyOffset > 0 ? <button disabled={busy} onClick={() => void run(async () => {
        const offset = Math.max(0, historyOffset - 20); const result = await getMatchHistory(offset);
        setHistory(result.matches); setHistoryOffset(offset);
      })}>前の20件</button> : null}
    </details>
    {message && <p role="alert">{message}</p>}
  </section>;
}
