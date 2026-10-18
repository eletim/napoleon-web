import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { describe, expect, it, vi } from "vitest";
const source=readFileSync(new URL("../public/sw.js",import.meta.url),"utf8");
function worker(base="/") {
  const handlers=new Map<string,(event:any)=>void>();
  const put=vi.fn(async()=>{});const del=vi.fn(async()=>true);
  const cache={put,match:vi.fn(async()=>undefined),keys:vi.fn(async()=>[])};
  const fetch=vi.fn(async()=>new Response("data",{headers:{"Cache-Control":"no-store"}}));
  runInNewContext(source,{URL,fetch,caches:{open:async()=>cache,keys:async()=>["napoleon-web-v1"],delete:del},
    self:{location:{href:`https://game.test${base}sw.js`,origin:"https://game.test"},registration:{scope:`https://game.test${base}`},
      clients:{claim:async()=>{}},addEventListener:(name:string,handler:(event:any)=>void)=>handlers.set(name,handler)}});
  return {handlers,put,del,fetch};
}
describe("service worker authorization boundaries",()=>{
  it.each(["/","/napoleon/"])("never intercepts authenticated requests or encoded API paths at %s",async(base)=>{
    const w=worker(base);
    for(const [path,headers] of [["private-resource",{Authorization:"Bearer token"}],["%61pi/me",{}],["api/me",{}]] as const){
      const respondWith=vi.fn();w.handlers.get("fetch")!({request:new Request(`https://game.test${base}${path}`,{headers}),respondWith});
      expect(respondWith).not.toHaveBeenCalled();expect(w.put).not.toHaveBeenCalled();
    }
  });
  it("honors no-store/private responses and clears the previous cache",async()=>{
    const w=worker();
    for(const control of ["no-store","private, max-age=60"]){
      w.fetch.mockResolvedValueOnce(new Response("private",{headers:{"Cache-Control":control}}));
      let response:Promise<Response>|undefined;
      w.handlers.get("fetch")!({request:new Request("https://game.test/asset.js"),respondWith:(r:Promise<Response>)=>{response=r;}});
      expect(await response).toBeDefined();expect(w.put).not.toHaveBeenCalled();
    }
    let activate:Promise<unknown>|undefined;w.handlers.get("activate")!({waitUntil:(p:Promise<unknown>)=>{activate=p;}});await activate;
    expect(w.del).toHaveBeenCalledWith("napoleon-web-v1");
  });
});
