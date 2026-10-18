import { Pool } from "pg";
import { calculateRatingChanges, type RatingChange, type RatingEntry, type Standing } from "./ratings.js";

export interface HistoryRecord { matchId: string; completedAt: string; snapshot: unknown; changes: RatingChange[]; }
export interface CompetitionStore {
  history(participantId: string, offset: number): Promise<HistoryRecord[]>;
  ensure(entry: Pick<RatingEntry, "id" | "kind" | "displayName">): Promise<RatingEntry>;
  save(id: string, snapshot: unknown, completed: boolean): Promise<void>;
  load(id: string): Promise<unknown | undefined>;
  active(): Promise<unknown[]>;
  settle(id: string, snapshot: unknown, standings: Standing[]): Promise<RatingChange[]>;
  close(): Promise<void>;
}
export async function createCompetitionStore(databaseUrl?: string): Promise<CompetitionStore> {
  if (!databaseUrl) return memoryCompetitionStore();
  const pool = new Pool({ connectionString: databaseUrl });
  try {
    await pool.query(`CREATE TABLE IF NOT EXISTS rating_accounts (
      id TEXT PRIMARY KEY, display_name TEXT NOT NULL, kind TEXT NOT NULL CHECK (kind IN ('human','cpu')),
      rating DOUBLE PRECISION NOT NULL DEFAULT 1500, games INTEGER NOT NULL DEFAULT 0);
      CREATE TABLE IF NOT EXISTS rated_matches (
        id TEXT PRIMARY KEY, snapshot JSONB NOT NULL, completed BOOLEAN NOT NULL DEFAULT FALSE);
      CREATE TABLE IF NOT EXISTS rating_results (
        match_id TEXT PRIMARY KEY REFERENCES rated_matches(id), changes JSONB NOT NULL, settled_at TIMESTAMPTZ NOT NULL DEFAULT now());
      ALTER TABLE rating_results ADD COLUMN IF NOT EXISTS settled_at TIMESTAMPTZ NOT NULL DEFAULT now();`);
  } catch (error) { await pool.end(); throw error; }
  return {
    async history(participantId, offset) {
      const result = await pool.query(`SELECT m.id AS "matchId", r.settled_at AS "completedAt", m.snapshot, r.changes
        FROM rating_results r JOIN rated_matches m ON m.id=r.match_id
        WHERE m.snapshot->'participants' @> $1::jsonb ORDER BY r.settled_at DESC, m.id DESC LIMIT 20 OFFSET $2`,
        [JSON.stringify([{ id: participantId, kind: "human" }]), offset]);
      return result.rows.map((row) => ({ ...row, completedAt: row.completedAt.toISOString() }));
    },
    async ensure(entry) {
      const result = await pool.query(`INSERT INTO rating_accounts (id, display_name, kind) VALUES ($1,$2,$3)
        ON CONFLICT (id) DO UPDATE SET display_name = EXCLUDED.display_name
        RETURNING id, display_name AS "displayName", kind, rating, games`, [entry.id, entry.displayName, entry.kind]);
      return result.rows[0];
    },
    async save(id, snapshot, completed) {
      await pool.query(`INSERT INTO rated_matches (id,snapshot,completed) VALUES ($1,$2,$3)
        ON CONFLICT (id) DO UPDATE SET snapshot=EXCLUDED.snapshot, completed=EXCLUDED.completed`, [id, JSON.stringify(snapshot), completed]);
    },
    async load(id) { return (await pool.query("SELECT snapshot FROM rated_matches WHERE id=$1", [id])).rows[0]?.snapshot; },
    async active() { return (await pool.query("SELECT snapshot FROM rated_matches WHERE completed=FALSE")).rows.map((r) => r.snapshot); },
    async settle(id, snapshot, standings) {
      const client = await pool.connect();
      try {
        await client.query("BEGIN");
        await client.query("SELECT pg_advisory_xact_lock(hashtext($1))", [id]);
        const previous = await client.query("SELECT changes FROM rating_results WHERE match_id=$1", [id]);
        if (previous.rows[0]) { await client.query("COMMIT"); return previous.rows[0].changes; }
        const ids = [...new Set(standings.map((p) => p.participantId))].sort();
        const rows = await client.query("SELECT id, rating FROM rating_accounts WHERE id=ANY($1::text[]) ORDER BY id FOR UPDATE", [ids]);
        const changes = calculateRatingChanges(standings, new Map(rows.rows.map((r) => [r.id, r.rating])));
        for (const participantId of ids) {
          const seats = changes.filter((c) => c.participantId === participantId);
          await client.query("UPDATE rating_accounts SET rating=$2, games=games+$3 WHERE id=$1", [participantId, seats[0].after, seats.length]);
        }
        await client.query(`INSERT INTO rated_matches (id,snapshot,completed) VALUES ($1,$2,TRUE)
          ON CONFLICT (id) DO UPDATE SET snapshot=EXCLUDED.snapshot, completed=TRUE`, [id, JSON.stringify({ ...(snapshot as object), ratingChanges: changes })]);
        await client.query("INSERT INTO rating_results (match_id,changes) VALUES ($1,$2)", [id, JSON.stringify(changes)]);
        await client.query("COMMIT"); return changes;
      } catch (error) { await client.query("ROLLBACK"); throw error; }
      finally { client.release(); }
    },
    close: () => pool.end()
  };
}

export function memoryCompetitionStore(): CompetitionStore {
  const accounts = new Map<string, RatingEntry>();
  const snapshots = new Map<string, { value: unknown; completed: boolean }>();
  const results = new Map<string, RatingChange[]>();
  const completedAt = new Map<string, string>();
  return {
    async history(participantId, offset) {
      return [...results].reverse().filter(([id]) => {
        const snapshot = snapshots.get(id)?.value as { participants?: { id: string; kind: string }[] };
        return snapshot?.participants?.some((p) => p.id === participantId && p.kind === "human");
      }).slice(offset, offset + 20).map(([id, changes]) => ({ matchId: id, completedAt: completedAt.get(id)!,
        snapshot: structuredClone(snapshots.get(id)!.value), changes: structuredClone(changes) }));
    },
    async ensure(entry) {
      const account = accounts.get(entry.id) ?? { ...entry, rating: 1500, games: 0 };
      account.displayName = entry.displayName; accounts.set(entry.id, account); return { ...account };
    },
    async save(id, value, completed) { snapshots.set(id, { value: structuredClone(value), completed }); },
    async load(id) { return structuredClone(snapshots.get(id)?.value); },
    async active() { return [...snapshots.values()].filter((s) => !s.completed).map((s) => structuredClone(s.value)); },
    async settle(id, snapshot, standings) {
      if (results.has(id)) return results.get(id)!;
      const changes = calculateRatingChanges(standings, new Map([...accounts].map(([id, a]) => [id, a.rating])));
      for (const change of changes) { const account = accounts.get(change.participantId)!; account.rating = change.after; account.games++; }
      completedAt.set(id, new Date().toISOString());
      results.set(id, changes); snapshots.set(id, { value: structuredClone({ ...(snapshot as object), ratingChanges: changes }), completed: true }); return changes;
    },
    async close() {}
  };
}
