import { createHash, randomUUID } from "node:crypto";
import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { advanceToNextTrick, applyAction, clearLatestEvent, completeCurrentRound, createInitialMatch,
  updateCurrentGame, GameRuleError, MatchRuleError, type GameState, type MatchState } from "@napoleon/game-core";
import { RuleBasedAgent, type Agent, type PublicActionRecord } from "@napoleon/ai";
import type { AiPolicyComposition, CreateGameResponse, PublicRatedParticipant } from "@napoleon/protocol";
import type { AgentRegistry } from "./agentRegistry.js";
import type { AiPresetRegistry } from "./aiPresetRegistry.js";
import type { PlayerRepository } from "./players.js";
import type { CompetitionStore } from "./competitionStore.js";
import type { RatingChange, RatingEntry } from "./ratings.js";
import { advanceAiTurns, appendPublicBiddingAction, toInternalAction } from "./routes.js";
import { readActionBody } from "./validation.js";
import { toPublicGameState } from "./publicState.js";
import { toPublicMatchState } from "./publicMatch.js";

interface Participant extends RatingEntry { seatId: string; composition?: AiPolicyComposition; }
interface Snapshot {
  id: string; version: number; state: GameState; match: MatchState; participants: Participant[];
  history: readonly PublicActionRecord[]; ratingChanges?: RatingChange[];
  deadlineAt?: number; substituteSeats?: string[];
}
interface Ticket { account: RatingEntry; joinedAt: number; cpuId: string; }
class RequestError extends Error { constructor(readonly status: number, message: string) { super(message); } }
export interface RatedOptions {
  players?: PlayerRepository; store: CompetitionStore; agents: AgentRegistry; presets: AiPresetRegistry;
  now?: () => number; waitMs?: number; turnMs?: number;
}

export async function registerRatedMatches(app: FastifyInstance, options: RatedOptions) {
  const { players, store, agents, presets } = options;
  const now = options.now ?? Date.now;
  const waitMs = options.waitMs ?? 10000;
  const turnMs = options.turnMs ?? 30000;
  const substitutes = (s: Snapshot) => s.substituteSeats ?? [];
  const humanSeats = (s: Snapshot) => s.participants.filter((p) => p.kind === "human" && !substitutes(s).includes(p.seatId)).map((p) => p.seatId);
  const stateKey = (s: Snapshot) => `${s.match.currentRound}:${s.state.phase}:${s.state.trickNumber}:${s.state.currentPlayerId}:${s.state.isTrickComplete}:${s.state.isGameOver}`;
  function clock(s: Snapshot, before?: Snapshot) {
    if (s.match.completed) { delete s.deadlineAt; return; }
    if (before?.deadlineAt !== undefined && stateKey(s) === stateKey(before)) { s.deadlineAt = before.deadlineAt; return; }
    s.deadlineAt = now() + (s.state.isGameOver ? 8000 : s.state.isTrickComplete ? 5000 : turnMs);
  }
  const matches = new Map<string, Snapshot>();
  const queue = new Map<string, Ticket>();
  const active = new Map<string, string>();
  const agentCache = new Map<string, { signature: string; agents: Map<string, Agent> }>();
  const locks = new Map<string, Promise<unknown>>();
  let ticking: Promise<void> | undefined;
  let timer: ReturnType<typeof setInterval> | undefined;
  app.addHook("onClose", async () => { clearInterval(timer); await ticking; await store.close(); });
  async function locked<T>(key: string, work: () => Promise<T>): Promise<T> {
    const previous = locks.get(key) ?? Promise.resolve();
    const pending = previous.catch(() => {}).then(work); locks.set(key, pending);
    try { return await pending; } finally { if (locks.get(key) === pending) locks.delete(key); }
  }
  for (const raw of await store.active()) {
    const snapshot = raw as Snapshot;
    if (snapshot.deadlineAt === undefined) { clock(snapshot); await store.save(snapshot.id, snapshot, false); }
    matches.set(snapshot.id, snapshot);
    for (const p of snapshot.participants) if (p.kind === "human") active.set(p.id, snapshot.id);
  }
  async function account(request: FastifyRequest) {
    if (!request.identity || !players) throw new RequestError(401, "ログインしてください。");
    const p = await players.getOrCreate(request.identity);
    return store.ensure({ id: `human:${p.id}`, displayName: p.displayName, kind: "human" });
  }
  async function cpus() {
    const result: Array<RatingEntry & { presetId: string; composition: AiPolicyComposition }> = [];
    for (const descriptor of presets.list()) {
      let preset;
      try { preset = presets.resolve(descriptor.id); } catch { continue; }
      {
        const c = preset.composition;
        const hash = createHash("sha256").update(JSON.stringify([c.playing, c.bidding, c.nonPlaying])).digest("hex");
        result.push({ ...await store.ensure({ id: `cpu:${hash}`, kind: "cpu", displayName: preset.displayName }),
          presetId: preset.id, composition: c });
      }
    }
    return result;
  }
  async function selectedCpu(body: unknown) {
    const id = (body as { cpuId?: unknown } | null)?.cpuId;
    const available = await cpus();
    const selected = id === undefined ? available[0] : available.find((p) => p.id === id);
    if (!selected) throw new RequestError(400, "利用可能なCPUを選んでください。");
    return selected;
  }
  async function create(humans: RatingEntry[], cpu: RatingEntry & { composition: AiPolicyComposition }) {
    const match = createInitialMatch();
    const participants: Participant[] = Array.from({ length: 5 }, (_, i) => ({
      ...(humans[i] ?? cpu), seatId: `player-${i}`
    }));
    // Each human gets a distinct seat; public views are created for that seat.
    const snapshot: Snapshot = { id: `rated-${randomUUID()}`, version: 0, state: match.currentGame!, match,
      participants, history: [], substituteSeats: [] };
    clock(snapshot);
    getAgents(snapshot); // Validate CPU setup before registering the match.
    try { await store.save(snapshot.id, snapshot, false); }
    catch (error) { agentCache.delete(snapshot.id); throw error; }
    matches.set(snapshot.id, snapshot);
    for (const human of humans) { active.set(human.id, snapshot.id); queue.delete(human.id); }
    return snapshot;
  }
  function getAgents(snapshot: Snapshot) {
    const signature = [...substitutes(snapshot)].sort().join(",");
    let cached = agentCache.get(snapshot.id);
    if (!cached || cached.signature !== signature) {
      const registered = new Map(snapshot.participants.filter((p) => p.kind === "cpu").map((p) => [p.seatId,
        agents.createComposedAgent(p.composition!)]));
      for (const seat of substitutes(snapshot)) registered.set(seat, new RuleBasedAgent());
      cached = { signature, agents: registered }; agentCache.set(snapshot.id, cached);
    }
    return cached.agents;
  }
  async function progress(draft: Snapshot, before: Snapshot) {
    if (!draft.match.completed) {
      const advanced = await advanceAiTurns(draft.state, humanSeats(draft), getAgents(draft), draft.history);
      draft.state = advanced.state; draft.history = advanced.publicActionHistory;
      draft.match = updateCurrentGame(draft.match, draft.state);
      if (draft.match.currentRound === 5 && draft.state.isGameOver) draft.match = completeCurrentRound(draft.match);
    }
    draft.version++;
    clock(draft, before);
    if (draft.match.completed) {
      const final = toPublicMatchState(draft.match, draft.participants.map((p) => p.seatId));
      draft.ratingChanges = await store.settle(draft.id, draft, final.finalScores!.map((p) => ({
        seatId: p.playerId, participantId: draft.participants.find((s) => s.seatId === p.playerId)!.id, rank: p.rank
      })));
    } else await store.save(draft.id, draft, false);
    matches.set(draft.id, draft);
    if (draft.match.completed) {
      agentCache.delete(draft.id);
      for (const p of draft.participants) if (p.kind === "human") active.delete(p.id);
    }
    return draft;
  }
  async function expire(current: Snapshot) {
    if (current.match.completed || current.deadlineAt === undefined || now() < current.deadlineAt) return current;
    const draft = structuredClone(current);
    if (draft.state.isGameOver) {
      draft.match = completeCurrentRound(updateCurrentGame(draft.match, draft.state));
      if (draft.match.currentGame) { draft.state = draft.match.currentGame; draft.history = []; }
    } else if (draft.state.isTrickComplete) draft.state = advanceToNextTrick(clearLatestEvent(draft.state));
    else if (!substitutes(draft).includes(draft.state.currentPlayerId)) {
      draft.substituteSeats = [...substitutes(draft), draft.state.currentPlayerId];
    }
    return progress(draft, current);
  }
  if (!options.now) {
    timer = setInterval(() => {
      if (ticking) return;
      ticking = (async () => {
        for (const snapshot of matches.values()) if (!snapshot.match.completed) {
          try { await locked(snapshot.id, async () => { await expire(matches.get(snapshot.id)!); }); }
          catch (error) { app.log.error(error, "Automatic match progression failed"); }
        }
      })().finally(() => { ticking = undefined; });
    }, 500);
    timer.unref();
  }
  async function sweep() {
    while (queue.size) {
      const first = [...queue.values()].sort((a,b) => a.joinedAt - b.joinedAt)[0];
      const candidates = [...queue.values()].filter((t) => Math.abs(t.account.rating - first.account.rating) <= 400)
        .sort((a,b) => Math.abs(a.account.rating - first.account.rating) - Math.abs(b.account.rating - first.account.rating));
      if (candidates.length < 5 && now() - first.joinedAt < waitMs) break;
      const cpu = await selectedCpu({ cpuId: first.cpuId });
      await create(candidates.slice(0,5).map((t) => t.account), cpu);
    }
  }
  function response(snapshot: Snapshot, id: string): CreateGameResponse {
    const self = snapshot.participants.find((p) => p.id === id && p.kind === "human");
    if (!self) throw new RequestError(404, "対戦が見つかりません。");
    const participants: PublicRatedParticipant[] = snapshot.participants.map((p) => ({
      seatId: p.seatId, id: p.id, displayName: p.displayName, kind: p.kind, rating: p.rating
    }));
    return { gameId: snapshot.id, playerId: self.seatId, state: toPublicGameState(snapshot.state, self.seatId),
      match: toPublicMatchState(snapshot.match, snapshot.participants.map((p) => p.seatId)),
      competition: { version: snapshot.version, participants, ratingChanges: snapshot.ratingChanges, deadlineAt: snapshot.deadlineAt, serverTime: now(), substituteSeats: substitutes(snapshot) } };
  }
  async function find(id: string, participantId: string) {
    let snapshot = matches.get(id);
    if (!snapshot) {
      snapshot = await store.load(id) as Snapshot | undefined;
      if (snapshot) matches.set(id, snapshot);
    }
    if (!snapshot || !snapshot.participants.some((p) => p.kind === "human" && p.id === participantId)) {
      throw new RequestError(404, "対戦が見つかりません。");
    }
    return snapshot;
  }
  async function queueState(id: string) {
    await sweep();
    const matchId = active.get(id);
    if (matchId) return { status: "matched", gameId: matchId };
    const ticket = queue.get(id);
    return ticket ? { status: "waiting", waitedMs: now() - ticket.joinedAt, fillAfterMs: waitMs }
      : { status: "idle" };
  }
  function route(work: (request: FastifyRequest, reply: FastifyReply) => Promise<unknown>) {
    return async (request: FastifyRequest, reply: FastifyReply) => {
      try { return await work(request, reply); }
      catch (error) {
        if (error instanceof RequestError || error instanceof GameRuleError || error instanceof MatchRuleError) {
          return reply.code(error instanceof RequestError ? error.status : 400).send({ error: { code: "MATCH_REQUEST_ERROR", message: error.message } });
        }
        throw error;
      }
    };
  }
  app.get("/api/matches/history", route(async (r) => {
    const human = await account(r);
    const value = (r.query as { offset?: string }).offset ?? "0";
    if (!/^\d{1,7}$/.test(value)) throw new RequestError(400, "有効な履歴位置を指定してください。");
    const records = await store.history(human.id, Number(value));
    return { matches: records.map((record) => {
      const snapshot = record.snapshot as Snapshot;
      const self = record.changes.find((c) => c.participantId === human.id)!;
      return { matchId: record.matchId, completedAt: record.completedAt, rank: self.rank,
        before: self.before, after: self.after, delta: self.delta,
        participants: snapshot.participants.map((p) => ({ displayName: p.displayName, kind: p.kind,
          rank: record.changes.find((c) => c.seatId === p.seatId)!.rank })) };
    }) };
  }));
  app.get("/api/ratings/me", route(async (r) => account(r)));
  app.get("/api/cpus", route(async (r) => { await account(r); return { cpus: (await cpus()).map(({ composition: _, ...p }) => p) }; }));
  app.post("/api/matchmaking/cpu", route(async (r) => {
    const human = await account(r);
    return locked("queue", async () => {
      if (active.has(human.id)) return response(await find(active.get(human.id)!, human.id), human.id);
      return response(await create([human], await selectedCpu(r.body)), human.id);
    });
  }));
  app.post("/api/matchmaking", route(async (r) => {
    const human = await account(r);
    return locked("queue", async () => {
      if (!active.has(human.id) && !queue.has(human.id)) {
        const cpu = await selectedCpu(r.body);
        queue.set(human.id, { account: human, joinedAt: now(), cpuId: cpu.id });
      }
      return queueState(human.id);
    });
  }));
  app.get("/api/matchmaking", route(async (r) => {
    const human = await account(r); return locked("queue", () => queueState(human.id));
  }));
  app.delete("/api/matchmaking", route(async (r) => {
    const human = await account(r);
    return locked("queue", async () => { queue.delete(human.id); return { status: active.has(human.id) ? "matched" : "idle", gameId: active.get(human.id) }; });
  }));
  app.get("/api/rated-matches/:matchId", route(async (r) => {
    const human = await account(r); const { matchId } = r.params as { matchId: string };
    return locked(matchId, async () => response(await expire(await find(matchId, human.id)), human.id));
  }));
  for (const operation of ["actions", "next-trick", "next-round", "leave", "resume"] as const) {
    app.post(`/api/rated-matches/:matchId/${operation}`, route(async (r) => {
      const human = await account(r); const { matchId } = r.params as { matchId: string };
      return locked(matchId, async () => {
        const current = await expire(await find(matchId, human.id));
        if (current.match.completed) return response(current, human.id);
        if (r.headers["if-match"] !== String(current.version)) throw new RequestError(409, "対戦状況が更新されました。最新の状態を取得します。");
        const draft = structuredClone(current);
        const self = draft.participants.find((p) => p.id === human.id)!;
        if (operation === "leave") {
          draft.substituteSeats = [...new Set([...substitutes(draft), self.seatId])];
        } else if (operation === "resume") {
          draft.substituteSeats = substitutes(draft).filter((s) => s !== self.seatId);
        } else if (operation === "actions") {
          if (substitutes(draft).includes(self.seatId)) throw new RequestError(409, "CPU代打中です。復帰してから操作してください。");
          const body = readActionBody(r.body);
          if (!body) throw new RequestError(400, "有効な操作が必要です。");
          const action = toInternalAction(body, self.seatId);
          draft.state = applyAction(clearLatestEvent(draft.state), action);
          draft.history = appendPublicBiddingAction(draft.history, action);
        } else if (operation === "next-trick") {
          draft.state = advanceToNextTrick(clearLatestEvent(draft.state));
        } else {
          draft.match = completeCurrentRound(updateCurrentGame(draft.match, draft.state));
          if (draft.match.currentGame) { draft.state = draft.match.currentGame; draft.history = []; }
        }
        await progress(draft, current);
        return response(draft, human.id);
      });
    }));
  }
}
