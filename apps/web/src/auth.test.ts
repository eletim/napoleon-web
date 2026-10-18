// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from "vitest";
const mocks = vi.hoisted(() => ({
  init: vi.fn(), updateToken: vi.fn(), clearToken: vi.fn(),
  constructor: vi.fn(), token: "access-token"
}));
vi.mock("keycloak-js", () => ({ default: class {
  constructor(config: unknown) { mocks.constructor(config); }
  init = mocks.init;
  updateToken = mocks.updateToken;
  clearToken = mocks.clearToken;
  get token() { return mocks.token; }
} }));
afterEach(() => { vi.unstubAllGlobals(); vi.resetAllMocks(); vi.resetModules(); mocks.token = "access-token"; });
function config(enabled: boolean) {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ enabled,
    url: "https://auth.example.com", realm: "napoleon", clientId: "napoleon-web" }))));
}
describe("browser authentication", () => {
  it("initializes once across repeated mounts and uses PKCE without storing tokens", async () => {
    config(true); mocks.init.mockResolvedValue(true);
    const auth = await import("./auth");
    const first = auth.initializeAuth();
    expect(auth.initializeAuth()).toBe(first);
    await first;
    expect(mocks.init).toHaveBeenCalledTimes(1);
    expect(mocks.init).toHaveBeenCalledWith(expect.objectContaining({ pkceMethod: "S256", onLoad: "check-sso", checkLoginIframe: false }));
    expect(window.localStorage.length).toBe(0);
    expect(await auth.authHeaders()).toEqual({ Authorization: "Bearer access-token" });
    expect(mocks.updateToken).toHaveBeenCalledWith(30);
  });
  it("keeps the existing development mode available when authentication is disabled", async () => {
    config(false);
    const auth = await import("./auth");
    expect(await auth.initializeAuth()).toEqual({ client: null });
    expect(await auth.authHeaders()).toEqual({});
    expect(mocks.constructor).not.toHaveBeenCalled();
  });
  it("rejects requests when token refresh fails", async () => {
    config(true); mocks.updateToken.mockRejectedValue(new Error("expired"));
    const auth = await import("./auth"); await auth.initializeAuth();
    await expect(auth.authHeaders()).rejects.toThrow("再度ログイン");
    expect(mocks.clearToken).toHaveBeenCalledTimes(1);
  });
  it("does not enable development mode when fetching authentication config fails", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("unavailable", { status: 503 })));
    const auth = await import("./auth");
    await expect(auth.initializeAuth()).rejects.toThrow("認証設定");
    expect(mocks.constructor).not.toHaveBeenCalled();
  });
});
