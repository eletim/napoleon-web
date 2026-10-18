import { afterEach, describe, expect, it } from "vitest";
import { buildApp } from "../src/app.js";
import type { FastifyInstance } from "fastify";
import type { PlayerRepository } from "../src/players.js";
const apps: FastifyInstance[]=[];
const players:PlayerRepository={getOrCreate:async()=>({id:"test",displayName:"test"}),rename:async()=>({id:"test",displayName:"test"}),close:async()=>{}};
afterEach(async()=>{await Promise.all(apps.splice(0).map(a=>a.close()));});
describe("authorization review",()=>{
  it("does not expose protected handlers through alternate request paths",async()=>{
    const app=await buildApp({auth:{config:{issuer:"https://auth.test/realms/napoleon",clientId:"napoleon-web",databaseUrl:"unused"},players}});apps.push(app);
    for(const url of ["/api/agents","/%61pi/agents","/api%2fagents","/api//agents","//api/agents","/api/agents/","/api/agents?x=1"]){
      const r=await app.inject({url});expect([401,404],`${url} => ${r.statusCode}`).toContain(r.statusCode);
    }
  });
});
