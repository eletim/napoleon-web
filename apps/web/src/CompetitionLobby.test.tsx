// @vitest-environment happy-dom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CompetitionLobby } from "./CompetitionLobby";
import * as api from "./api";
vi.mock("./api",()=>({getCpuOpponents:vi.fn(),getMyRating:vi.fn(),getMatchmaking:vi.fn(),getGame:vi.fn(),joinMatchmaking:vi.fn(),cancelMatchmaking:vi.fn(),startCpuMatch:vi.fn()}));
Object.assign(globalThis,{IS_REACT_ACT_ENVIRONMENT:true});
let root:Root;let container:HTMLDivElement;
beforeEach(()=>{vi.useFakeTimers();vi.mocked(api.getCpuOpponents).mockResolvedValue({cpus:[{id:"cpu1",displayName:"CPU",rating:1600,games:8}]});
  vi.mocked(api.getMyRating).mockResolvedValue({id:"me",displayName:"Me",rating:1450,games:4});
  vi.mocked(api.getMatchmaking).mockResolvedValue({status:"idle"});
  container=document.createElement("div");document.body.append(container);root=createRoot(container);
});
afterEach(async()=>{await act(async()=>root.unmount());container.remove();vi.clearAllTimers();vi.useRealTimers();vi.resetAllMocks();});
async function click(label:string){await act(async()=>{[...container.querySelectorAll("button")].find(b=>b.textContent===label)!.click();});}
describe("rated match lobby",()=>{
  it("displays ratings and starts the selected CPU match",async()=>{
    const start=vi.fn();const session={gameId:"rated-1"} as Awaited<ReturnType<typeof api.startCpuMatch>>;
    vi.mocked(api.startCpuMatch).mockResolvedValue(session);
    await act(async()=>root.render(<CompetitionLobby onStart={start}/>));
    expect(container.textContent).toContain("1450");expect(container.textContent).toContain("1600");
    await click("CPUとすぐ対戦");expect(api.startCpuMatch).toHaveBeenCalledWith("cpu1");expect(start).toHaveBeenCalledWith(session);
  });
  it("joins and cancels matchmaking",async()=>{
    vi.mocked(api.joinMatchmaking).mockResolvedValue({status:"waiting",waitedMs:0,fillAfterMs:10000});
    vi.mocked(api.cancelMatchmaking).mockResolvedValue({status:"idle"});
    await act(async()=>root.render(<CompetitionLobby onStart={vi.fn()}/>));
    await click("マッチング開始");expect(container.textContent).toContain("対戦相手を探しています");
    await click("キャンセル");expect(api.cancelMatchmaking).toHaveBeenCalledTimes(1);
    expect(container.textContent).toContain("マッチング開始");
  });
  it("reconnects to an existing match",async()=>{
    const start=vi.fn();const session={gameId:"rated-resume"} as Awaited<ReturnType<typeof api.getGame>>;
    vi.mocked(api.getMatchmaking).mockResolvedValue({status:"matched",gameId:"rated-resume"});vi.mocked(api.getGame).mockResolvedValue(session);
    await act(async()=>root.render(<CompetitionLobby onStart={start}/>));
    expect(api.getGame).toHaveBeenCalledWith("rated-resume");expect(start).toHaveBeenCalledWith(session);
  });
});
