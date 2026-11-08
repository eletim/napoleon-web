import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { describe, expect, it, vi } from "vitest";
const source=readFileSync(new URL("../public/sw.js",import.meta.url),"utf8");
function worker(base="/") {
  const handlers=new Map<string,(event:any)=>void>();
  const put=vi.fn(async()=>{});const del=vi.fn(async()=>true);
  const cache={put,match:vi.fn(async(): Promise<Response | undefined> => undefined),keys:vi.fn(async()=>[])};
  const fetch=vi.fn(async()=>new Response("data",{headers:{"Cache-Control":"no-store"}}));
  runInNewContext(source,{URL,fetch,caches:{open:async()=>cache,keys:async()=>["napoleon-web-v1"],delete:del},
    self:{location:{href:`https://game.test${base}sw.js`,origin:"https://game.test"},registration:{scope:`https://game.test${base}`},
      clients:{claim:async()=>{}},addEventListener:(name:string,handler:(event:any)=>void)=>handlers.set(name,handler)}});
  return {handlers,put,del,fetch,cache};
}
describe("service worker layout updates",()=>{
  it.each(["manifest.webmanifest", "src/main.tsx"])("refreshes cached layout configuration %s, with an offline fallback", async path => {
    const w = worker("/napoleon/");
    w.cache.match.mockResolvedValue(new Response("old-layout"));
    w.fetch.mockResolvedValue(new Response("new-layout"));
    let response: Promise<Response> | undefined;
    const event = { request: new Request(`https://game.test/napoleon/${path}`), respondWith: (r: Promise<Response>) => { response = r; } };
    w.handlers.get("fetch")!(event);
    expect(await (await response)!.text()).toBe("new-layout");
    w.fetch.mockRejectedValue(new Error("offline"));
    w.handlers.get("fetch")!(event);
    expect(await (await response)!.text()).toBe("old-layout");
  });
  it("clears the previous layout cache", async () => {
    const w = worker();
    let activate: Promise<unknown> | undefined;
    w.handlers.get("activate")!({ waitUntil: (p: Promise<unknown>) => { activate = p; } });
    await activate;
    expect(w.del).toHaveBeenCalledWith("napoleon-web-v1");
  });
});
