import { describe, expect, it } from "vitest";
import { calculateRatingChanges } from "../src/ratings.js";
import { memoryCompetitionStore } from "../src/competitionStore.js";
const standings = [1,2,3,4,5].map((rank, i) => ({ seatId: `s${i}`, participantId: `p${i}`, rank }));
describe("five-player ratings", () => {
  it("uses final ranks with zero-sum updates and rewards an upset more", () => {
    const ratings = new Map(standings.map((p) => [p.participantId, 1500]));
    const changes = calculateRatingChanges(standings, ratings);
    expect(changes.map((p) => p.delta)).toEqual([16,8,0,-8,-16]);
    expect(changes.reduce((s,p) => s+p.delta,0)).toBeCloseTo(0);
    ratings.set("p0", 1100);
    expect(calculateRatingChanges(standings, ratings)[0].delta).toBeGreaterThan(16);
    expect(calculateRatingChanges(standings.map((p) => ({...p,rank:1})), new Map(standings.map((p) => [p.participantId,1500]))).every((p) => p.delta===0)).toBe(true);
  });
  it("aggregates repeated CPU identities and settles each match only once", async () => {
    const store = memoryCompetitionStore();
    await store.ensure({id:"human",kind:"human",displayName:"Alice"});
    await store.ensure({id:"cpu",kind:"cpu",displayName:"CPU"});
    const result = standings.map((p,i) => ({...p,participantId:i===0?"human":"cpu"}));
    const changes = await store.settle("match", {}, result);
    expect(changes[0].delta).toBe(16);
    expect(changes[1].delta).toBe(-16);
    expect(await store.settle("match", {}, result)).toEqual(changes);
    expect(await store.ensure({id:"human",kind:"human",displayName:"Alice"})).toMatchObject({rating:1516,games:1});
    expect(await store.ensure({id:"cpu",kind:"cpu",displayName:"CPU"})).toMatchObject({rating:1484,games:4});
    expect(await store.load("match")).toHaveProperty("ratingChanges", changes);
  });
});
