import { readFileSync, writeFileSync, chmodSync } from "node:fs";
const raw = readFileSync(".env.production", "utf8");
const line = raw.split(/\r?\n/).find(l => l.trim().startsWith("DOMAIN="));
const domain = line?.trim().slice(7).replace(/^['"]|['"]$/g, "");
if (!domain || !/^(?=.{1,253}$)[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?$/.test(domain) || domain.includes("..") || domain === "game.example.com") {
  throw new Error("Set DOMAIN to the actual DNS hostname in .env.production.");
}
const passwords = ["PLAYERS_DB_PASSWORD", "KEYCLOAK_DB_PASSWORD", "KEYCLOAK_ADMIN_PASSWORD"].map(key => {
  const entry = raw.split(/\r?\n/).find(l => l.trim().startsWith(`${key}=`));
  const value = entry?.trim().slice(key.length + 1).replace(/^['"]|['"]$/g, "");
  if (!value || value.length < 16 || value.startsWith("replace-")) throw new Error(`Replace ${key} with a strong password.`);
  return value;
});
if (new Set(passwords).size !== 3) throw new Error("Use independent database and administrator passwords.");
const realm = JSON.parse(readFileSync("infra/keycloak/napoleon-realm.json", "utf8"));
const client = realm.clients.find(c => c.clientId === "napoleon-web");
client.redirectUris = [`https://${domain}/napoleon/`];
client.webOrigins = [`https://${domain}`];
client.attributes["post.logout.redirect.uris"] = `https://${domain}/napoleon/`;
writeFileSync("infra/keycloak/production-realm.json", JSON.stringify(realm, null, 2) + "\n", { mode: 0o644 });
chmodSync("infra/keycloak/production-realm.json", 0o644);
console.log("Production realm prepared.");
