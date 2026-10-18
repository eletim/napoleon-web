// @vitest-environment happy-dom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import type Keycloak from "keycloak-js";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AccountGate } from "./AccountGate";
import { initializeAuth } from "./auth";
import { usePermissions } from "./permissions";
vi.mock("./auth",()=>({initializeAuth:vi.fn(),authHeaders:vi.fn(async()=>({Authorization:"Bearer token"}))}));
Object.assign(globalThis,{IS_REACT_ACT_ENVIRONMENT:true});
const roles=new Set<string>();
const client={authenticated:true,hasRealmRole:(role:string)=>roles.has(role),
  onAuthRefreshSuccess:undefined as (()=>void)|undefined,onAuthLogout:undefined as (()=>void)|undefined,
  onTokenExpired:undefined as (()=>void)|undefined,updateToken:vi.fn(),clearToken:vi.fn(),login:vi.fn(),register:vi.fn(),logout:vi.fn()};
function Child(){const {isAdmin}=usePermissions();return <span>{isAdmin?"管理画面":"通常画面"}</span>;}
let root:Root;let container:HTMLDivElement;
beforeEach(()=>{
  roles.clear();roles.add("user");roles.add("admin");client.authenticated=true;
  client.updateToken.mockImplementation(async()=>{client.onAuthRefreshSuccess?.();return true;});
  client.clearToken.mockImplementation(()=>{client.authenticated=false;client.onAuthLogout?.();});
  vi.mocked(initializeAuth).mockResolvedValue({client:client as unknown as Keycloak});
  vi.stubGlobal("fetch",vi.fn(async()=>new Response(JSON.stringify({id:"self",displayName:"自分"}),{status:200})));
  container=document.createElement("div");document.body.append(container);root=createRoot(container);
});
afterEach(async()=>{await act(async()=>root.unmount());container.remove();vi.unstubAllGlobals();vi.clearAllMocks();});
describe("account permissions lifecycle",()=>{
  it("removes admin UI after an expired token is refreshed with revoked roles",async()=>{
    await act(async()=>root.render(<AccountGate><Child/></AccountGate>));expect(container.textContent).toContain("管理画面");
    roles.delete("admin");await act(async()=>client.onTokenExpired?.());
    expect(client.updateToken).toHaveBeenCalledWith(30);expect(container.textContent).toContain("通常画面");
    expect(container.textContent).not.toContain("管理画面");
    roles.delete("user");await act(async()=>client.onTokenExpired?.());
    expect(container.textContent).toContain("利用権限がありません");expect(container.textContent).not.toContain("通常画面");
  });
  it("removes privileged content and returns to login when refresh fails",async()=>{
    await act(async()=>root.render(<AccountGate><Child/></AccountGate>));
    client.updateToken.mockRejectedValueOnce(new Error("session expired"));await act(async()=>client.onTokenExpired?.());
    expect(client.clearToken).toHaveBeenCalledTimes(1);expect(container.textContent).toContain("ログインしてプレイ");
    expect(container.textContent).not.toContain("管理画面");expect(container.textContent).not.toContain("自分");
  });
});
