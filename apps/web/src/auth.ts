import Keycloak from "keycloak-js";
import { resolveAppPath } from "./appPath";

export interface AuthSession { client: Keycloak | null; }
let initialization: Promise<AuthSession> | undefined;
let client: Keycloak | null = null;

export function initializeAuth(): Promise<AuthSession> {
  initialization ??= (async () => {
    const response = await fetch(resolveAppPath("api/auth/config"), { cache: "no-store" });
    if (!response.ok) throw new Error("認証設定を取得できませんでした。");
    const config = await response.json() as { enabled: boolean; url: string; realm: string; clientId: string };
    if (config.enabled === false) return { client: null };
    if (config.enabled !== true || typeof config.url !== "string" ||
        typeof config.realm !== "string" || typeof config.clientId !== "string") {
      throw new Error("認証設定が不正です。");
    }
    client = new Keycloak({ url: config.url, realm: config.realm, clientId: config.clientId });
    await client.init({ onLoad: "check-sso", pkceMethod: "S256", checkLoginIframe: false,
      redirectUri: new URL(import.meta.env.BASE_URL, window.location.origin).href });
    return { client };
  })();
  return initialization;
}

export async function authHeaders(): Promise<Record<string, string>> {
  if (client === null) return {};
  try {
    await client.updateToken(30);
    if (!client.token) throw new Error("Missing token");
    return { Authorization: `Bearer ${client.token}` };
  } catch {
    client.clearToken();
    throw new Error("セッションの有効期限が切れました。再度ログインしてください。");
  }
}
