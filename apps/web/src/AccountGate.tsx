import { useEffect, useState, type ReactNode } from "react";
import { authHeaders, initializeAuth, type AuthSession } from "./auth";
import { PermissionsContext, type Permissions } from "./permissions";
import { resolveAppPath } from "./appPath";
import "./account.css";

interface Profile { id: string; displayName: string; }
export function AccountGate({ children }: { children: ReactNode }) {
  const [permissions, setPermissions] = useState<Permissions>({ canPlay: false, isAdmin: false });
  const [session, setSession] = useState<AuthSession>();
  const [authenticated, setAuthenticated] = useState(false);
  const [profile, setProfile] = useState<Profile>();
  const [name, setName] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let active = true;
    let cleanup = () => {};
    initializeAuth().then(async (value) => {
      if (!active) return;
      setSession(value);
      const client = value.client;
      setAuthenticated(client?.authenticated ?? false);
      if (client === null) return;
      const updatePermissions = () => {
        if (!active) return;
        const isAdmin = client.hasRealmRole("admin");
        setPermissions({ isAdmin, canPlay: isAdmin || client.hasRealmRole("user") });
      };
      updatePermissions();
      client.onAuthRefreshSuccess = updatePermissions;
      client.onAuthLogout = () => { if (active) { setAuthenticated(false); setProfile(undefined); } };
      cleanup = () => { client.onAuthLogout = undefined; client.onAuthRefreshSuccess = undefined; };
      if (client.authenticated && (client.hasRealmRole("user") || client.hasRealmRole("admin"))) {
        const result = await profileRequest();
        if (active) { setProfile(result); setName(result.displayName); }
      }
    }).catch((error: unknown) => { if (active) setMessage(error instanceof Error ? error.message : "認証に失敗しました。"); });
    return () => { active = false; cleanup(); };
  }, []);

  async function action(work: () => Promise<unknown>) {
    setBusy(true); setMessage("");
    try { await work(); }
    catch (error) { setMessage(error instanceof Error ? error.message : "処理に失敗しました。"); }
    finally { setBusy(false); }
  }
  if (session?.client === null) return children;
  const client = session?.client;
  if (!client) return <main className="account-screen"><p role="status">{message || "読み込み中…"}</p>
    {message && <button onClick={() => window.location.reload()}>再読み込み</button>}</main>;
  const redirectUri = new URL(import.meta.env.BASE_URL, window.location.origin).href;
  if (!authenticated) return <main className="account-screen"><h1>Napoleon Web</h1>
    <p>ログインしてプレイを始めましょう。</p>
    <button disabled={busy} onClick={() => void action(() => client.login({ redirectUri }))}>ログイン</button>
    <button disabled={busy} onClick={() => void action(() => client.register({ redirectUri }))}>新規登録</button>
    {message && <p role="alert">{message}</p>}</main>;
  return <>
    <section className="account-bar" aria-label="アカウント">
      <details><summary>{profile?.displayName ?? "アカウント"}</summary>
        <form onSubmit={(event) => { event.preventDefault(); void action(async () => {
          const result = await profileRequest(name); setProfile(result); setName(result.displayName); setMessage("保存しました。");
        }); }}>
          <label>プレイヤー名<input value={name} required maxLength={60} onChange={(event) => setName(event.target.value)} /></label>
          <button disabled={busy}>保存</button>
        </form>
      </details>
      <button disabled={busy} onClick={() => void action(() => client.logout({ redirectUri }))}>ログアウト</button>
      {message && <p role="status">{message}</p>}
    </section>
    <PermissionsContext.Provider value={permissions}>
      {permissions.canPlay ? children : <p className="account-screen" role="alert">このアカウントには利用権限がありません。</p>}
    </PermissionsContext.Provider>
  </>;
}

async function profileRequest(displayName?: string): Promise<Profile> {
  const response = await fetch(resolveAppPath("api/me"), {
    method: displayName === undefined ? "GET" : "PUT",
    headers: { ...await authHeaders(), "Content-Type": "application/json" },
    ...(displayName === undefined ? {} : { body: JSON.stringify({ displayName }) })
  });
  if (!response.ok) {
    const body = await response.json();
    throw new Error(body.error?.message ?? "プレイヤー情報を取得できませんでした。");
  }
  return response.json() as Promise<Profile>;
}
