export interface RatingEntry { id: string; displayName: string; kind: "human" | "cpu"; rating: number; games: number; }
export interface Standing { seatId: string; participantId: string; rank: number; }
export interface RatingChange extends Standing { before: number; after: number; delta: number; }

/** Multiplayer Elo: compare final ranks pairwise; score differences never enter this calculation. */
export function calculateRatingChanges(standings: readonly Standing[], ratings: ReadonlyMap<string, number>): RatingChange[] {
  if (standings.length !== 5 || new Set(standings.map((p) => p.seatId)).size !== 5 ||
      standings.some((p) => !Number.isInteger(p.rank) || p.rank < 1 || p.rank > 5 || !Number.isFinite(ratings.get(p.participantId)))) {
    throw new Error("Five valid final standings and ratings are required.");
  }
  const deltas = standings.map(() => 0);
  for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) {
    if (standings[i].participantId === standings[j].participantId) continue;
    const expected = 1 / (1 + 10 ** ((ratings.get(standings[j].participantId)! - ratings.get(standings[i].participantId)!) / 400));
    const actual = standings[i].rank === standings[j].rank ? .5 : standings[i].rank < standings[j].rank ? 1 : 0;
    const delta = 32 / 4 * (actual - expected);
    deltas[i] += delta; deltas[j] -= delta;
  }
  const total = new Map<string, number>();
  standings.forEach((p, i) => total.set(p.participantId, (total.get(p.participantId) ?? 0) + deltas[i]));
  return standings.map((p) => ({ ...p, before: ratings.get(p.participantId)!,
    after: ratings.get(p.participantId)! + total.get(p.participantId)!, delta: total.get(p.participantId)! }));
}
