import { beforeAll, afterEach, describe, it, expect } from "vitest";
import { createLocalJWKSet, exportJWK, generateKeyPair, SignJWT } from "jose";
import { RuleBasedAgent, type Agent } from "@napoleon/ai";
import type { FastifyInstance } from "fastify";
import type { AgentRegistry } from "../src/agentRegistry.js";
import type { AiPresetRegistry } from "../src/aiPresetRegistry.js";
import type { PlayerRepository } from "../src/players.js";
import type { CreateGameResponse } from "@napoleon/protocol";
import { buildApp } from "../src/app.js";
import { memoryCompetitionStore, type CompetitionStore } from "../src/competitionStore.js";
const issuer="https://auth.test/realms/napoleon";
const apps: FastifyInstance[]=[];
let verificationKey: ReturnType<typeof createLocalJWKSet>;
let privateKey: CryptoKey;
let clock=0;
beforeAll(async()=>{
  const keys=await generateKeyPair("RS256"); privateKey=keys.privateKey;
  verificationKey=createLocalJWKSet({keys:[{...await exportJWK(keys.publicKey),kid:"test"}]});
});
afterEach(async()=>{await Promise.all(apps.splice(0).map(a=>a.close()));clock=0;});
class PassAgent extends RuleBasedAgent implements Agent {
  async selectAction(input: Parameters<Agent["selectAction"]>[0]) {
    return input.legalActions.find(a=>a.type==="pass") ?? input.legalActions[0];
  }
}
const registry: AgentRegistry={ initializePhasePolicies:async()=>{},listAgents:()=>[], listPhasePolicies:()=>({playing:[],bidding:[],nonPlaying:[]}),
  createAgent:()=>new PassAgent(),createComposedAgent:()=>new PassAgent(),createCompositionDiagnostics:()=>({playing:{callCount:0,fallbackCount:0},bidding:{callCount:0,fallbackCount:0},nonPlaying:{callCount:0,fallbackCount:0}})};
const preset={id:"com-rule-base" as const,displayName:"CPU",composition:{playing:"rule-based" as const,bidding:"rule-based" as const,nonPlaying:"rule-based" as const}};
const presets:AiPresetRegistry={list:()=>[preset],resolve:()=>preset,update:()=>preset};
const players:PlayerRepository={getOrCreate:async(i)=>({id:i.subject,displayName:i.username}),rename:async(i,n)=>({id:i.subject,displayName:n}),close:async()=>{}};
async function setup(store:CompetitionStore=memoryCompetitionStore(), realCpu = false) {
  const app=await buildApp({auth:{config:{issuer,clientId:"napoleon-web",databaseUrl:"unused"},players,verificationKey},
    competitionStore:store,agentRegistry:realCpu ? { ...registry, createComposedAgent: () => new RuleBasedAgent() } : registry,aiPresetRegistry:presets,now:()=>clock});
  apps.push(app);return app;
}
async function headers(sub="alice") {return {authorization:`Bearer ${await new SignJWT({typ:"Bearer",azp:"napoleon-web",realm_access:{roles:["user"]},preferred_username:sub})
  .setProtectedHeader({alg:"RS256",kid:"test"}).setIssuer(issuer).setAudience("napoleon-web").setSubject(sub).setExpirationTime("1h").sign(privateKey)}`};}
async function mutate(app:FastifyInstance,session:CreateGameResponse,operation:string,sub="alice") {
  return app.inject({method:"POST",url:`/api/rated-matches/${session.gameId}/${operation}`,headers:{...await headers(sub),"if-match":String(session.competition!.version)},
    ...(operation==="actions"?{payload:{action:{type:"pass"}}}:{})});
}
describe("rated matchmaking and matches",()=>{
  it("plays actual cards through five rounds with RuleBased CPU agents", async () => {
    const app=await setup(memoryCompetitionStore(),true);const h=await headers();
    let session=(await app.inject({method:"POST",url:"/api/matchmaking/cpu",headers:h})).json<CreateGameResponse>();
    let steps=0, cards=0;
    while (!session.match?.completed && steps++<1000) {
      let operation="actions";
      let action:unknown;
      if(session.state.isGameOver) operation="next-round";
      else if(session.state.isTrickComplete) operation="next-trick";
      else if(session.state.phase==="exchanging") action={type:"discard-cards",cardIds:session.state.self.hand.slice(0,session.state.exchange!.requiredDiscardCount).map(c=>c.id)};
      else if(session.state.phase==="choosing-adjutant") action={type:"choose-adjutant",cardId:"spades-A"};
      else { action=session.state.legalActions[0]; if ((action as {type:string})?.type==="play-card") cards++; }
      const result=await app.inject({method:"POST",url:`/api/rated-matches/${session.gameId}/${operation}`,
        headers:{...h,"if-match":String(session.competition!.version)},...(operation==="actions"?{payload:{action}}:{})});
      expect(result.statusCode,JSON.stringify(result.json())).toBe(200);session=result.json();
    }
    expect(cards).toBeGreaterThan(0);expect(session.match?.completed).toBe(true);
    expect(session.competition?.ratingChanges).toHaveLength(5);
    expect((await app.inject({url:"/api/ratings/me",headers:h})).json().games).toBe(1);
  });
  it("fills a timed queue with rated CPUs, supports cancellation, and prevents duplicate entry",async()=>{
    const app=await setup();const h=await headers();
    await app.inject({method:"POST",url:"/api/matchmaking",headers:h});
    await app.inject({method:"POST",url:"/api/matchmaking",headers:h});
    expect((await app.inject({url:"/api/matchmaking",headers:h})).json().status).toBe("waiting");
    await app.inject({method:"DELETE",url:"/api/matchmaking",headers:h});
    expect((await app.inject({url:"/api/matchmaking",headers:h})).json().status).toBe("idle");
    await app.inject({method:"POST",url:"/api/matchmaking",headers:h});clock=10000;
    const ticket=(await app.inject({url:"/api/matchmaking",headers:h})).json();
    const response=await app.inject({url:`/api/rated-matches/${ticket.gameId}`,headers:h});
    expect(response.statusCode).toBe(200);
    expect(response.json().competition.participants.filter((p:{kind:string})=>p.kind==="cpu")).toHaveLength(4);
  });
  it("matches five humans, returns private hands and enforces turns, membership and versions",async()=>{
    const app=await setup();
    for(let i=0;i<5;i++) await app.inject({method:"POST",url:"/api/matchmaking",headers:await headers(`p${i}`)});
    const ticket=(await app.inject({url:"/api/matchmaking",headers:await headers("p0")})).json();
    const sessions:CreateGameResponse[]=[];
    for(let i=0;i<5;i++) sessions.push((await app.inject({url:`/api/rated-matches/${ticket.gameId}`,headers:await headers(`p${i}`)})).json());
    expect(new Set(sessions.map(s=>s.playerId)).size).toBe(5);
    expect(sessions[1].state.self.id).toBe("player-1");
    expect(sessions[1].state.opponents.every(p=>!("hand" in p))).toBe(true);
    expect((await app.inject({url:`/api/rated-matches/${ticket.gameId}`,headers:await headers("outsider")})).statusCode).toBe(404);
    expect((await mutate(app,sessions[1],"actions","p1")).statusCode).toBe(400);
    const concurrent = await Promise.all([mutate(app,sessions[0],"actions","p0"), mutate(app,sessions[0],"actions","p0")]);
    expect(concurrent.map((r) => r.statusCode).sort()).toEqual([200,409]);
    expect((await mutate(app,sessions[0],"actions","p0")).statusCode).toBe(409);
  });
  it("completes five rounds against CPU and settles exactly once, including after a restart",async()=>{
    const store=memoryCompetitionStore();let app=await setup(store);const h=await headers();
    let session=(await app.inject({method:"POST",url:"/api/matchmaking/cpu",headers:h})).json<CreateGameResponse>();
    expect((await app.inject({url:"/api/ratings/me",headers:h})).json().games).toBe(0);
    // Restart the application with persisted match snapshots before continuing.
    await app.close();apps.splice(apps.indexOf(app),1);app=await setup(store);
    expect((await app.inject({url:"/api/matchmaking",headers:h})).json().gameId).toBe(session.gameId);
    for(let round=1;round<=5;round++) {
      const result=await mutate(app,session,"actions");expect(result.statusCode).toBe(200);session=result.json();
      if(round<5) {
        expect(session.competition!.ratingChanges).toBeUndefined();
        expect((await app.inject({url:"/api/ratings/me",headers:h})).json().games).toBe(0);
        session=(await mutate(app,session,"next-round")).json();
      }
    }
    expect(session.match!.completed).toBe(true);expect(session.competition!.ratingChanges).toHaveLength(5);
    expect((await app.inject({url:"/api/ratings/me",headers:h})).json().games).toBe(1);
    const duplicate=await mutate(app,session,"next-round");expect(duplicate.statusCode).toBe(200);
    expect((await app.inject({url:"/api/ratings/me",headers:h})).json().games).toBe(1);
    const cpus=(await app.inject({url:"/api/cpus",headers:h})).json().cpus;
    expect(cpus[0].games).toBe(4);
    expect((await app.inject({method:"POST",url:"/api/matchmaking/cpu",headers:h})).json().gameId).not.toBe(session.gameId);
  });
});
