import { Pool } from "pg";

export interface Identity { issuer: string; subject: string; username: string; }
export interface PlayerProfile { id: string; displayName: string; }
export interface PlayerRepository {
  getOrCreate(identity: Identity): Promise<PlayerProfile>;
  rename(identity: Identity, displayName: string): Promise<PlayerProfile>;
  close(): Promise<void>;
}

export async function createPlayerRepository(connectionString: string): Promise<PlayerRepository> {
  const pool = new Pool({ connectionString });
  try {
    await pool.query(`CREATE TABLE IF NOT EXISTS players (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      issuer TEXT NOT NULL,
      subject TEXT NOT NULL,
      display_name TEXT NOT NULL,
      UNIQUE (issuer, subject)
    )`);
  } catch (error) {
    await pool.end();
    throw error;
  }
  const upsert = async (identity: Identity, displayName?: string): Promise<PlayerProfile> => {
    const result = await pool.query<{ id: string; displayName: string }>(
      `INSERT INTO players (issuer, subject, display_name) VALUES ($1, $2, $3)
       ON CONFLICT (issuer, subject) DO UPDATE SET display_name =
         CASE WHEN $4::boolean THEN EXCLUDED.display_name ELSE players.display_name END
       RETURNING id, display_name AS "displayName"`,
      [identity.issuer, identity.subject, displayName ?? identity.username.slice(0, 30), displayName !== undefined]
    );
    return result.rows[0];
  };
  return {
    getOrCreate: (identity) => upsert(identity),
    rename: (identity, displayName) => upsert(identity, displayName),
    close: () => pool.end()
  };
}
