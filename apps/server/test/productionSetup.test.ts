import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, statSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
const root=fileURLToPath(new URL("../../../",import.meta.url));
describe("production preparation and database operations",()=>{
  it("generates exact production redirect URLs and rejects sample credentials",()=>{
    const dir=mkdtempSync(join(tmpdir(),"napoleon-production-"));
    try {
      mkdirSync(join(dir,"infra/keycloak"),{recursive:true});
      writeFileSync(join(dir,"infra/keycloak/napoleon-realm.json"),readFileSync(join(root,"infra/keycloak/napoleon-realm.json")));
      writeFileSync(join(dir,".env.production"),"DOMAIN=play.example.org\nPLAYERS_DB_PASSWORD=unique-player-password-123\nKEYCLOAK_DB_PASSWORD=unique-keycloak-password-456\nKEYCLOAK_ADMIN_PASSWORD=unique-administrator-password-789\n");
      execFileSync(process.execPath,[join(root,"scripts/prepare-production.mjs")],{cwd:dir});
      const realm=JSON.parse(readFileSync(join(dir,"infra/keycloak/production-realm.json"),"utf8"));
      expect(realm.clients[0].redirectUris).toEqual(["https://play.example.org/napoleon/"]);
      expect(realm.clients[0].webOrigins).toEqual(["https://play.example.org"]);
      expect(statSync(join(dir,"infra/keycloak/production-realm.json")).mode & 0o777).toBe(0o644);
      writeFileSync(join(dir,".env.production"),readFileSync(join(root,".env.production.example")));
      expect(()=>execFileSync(process.execPath,[join(root,"scripts/prepare-production.mjs")],{cwd:dir,stdio:"pipe"})).toThrow();
    } finally {rmSync(dir,{recursive:true,force:true});}
  });
  it("backs up both databases and restores them using a fixture Docker executable",()=>{
    const dir=mkdtempSync(join(tmpdir(),"napoleon-backup-"));const log=join(dir,"commands");
    try {
      const bin=join(dir,"bin");mkdirSync(bin);
      writeFileSync(join(bin,"docker"),'#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_DOCKER_LOG"\ncase "$*" in *" pg_dump "*) printf "fixture-archive";; esac\n',{mode:0o755});
      const env={...process.env,PATH:`${bin}:${process.env.PATH}`,TEST_DOCKER_LOG:log};
      const output=join(dir,"backup");
      execFileSync("bash",[join(root,"scripts/database-backup.sh"),output],{env});
      expect(readFileSync(join(output,"players.dump"),"utf8")).toBe("fixture-archive");
      expect(readFileSync(join(output,"keycloak.dump"),"utf8")).toBe("fixture-archive");
      expect(statSync(join(output,"players.dump")).mode & 0o077).toBe(0);
      expect(()=>execFileSync("bash",[join(root,"scripts/database-backup.sh"),output],{env,stdio:"pipe"})).toThrow();
      execFileSync("bash",[join(root,"scripts/database-restore.sh"),output],{env});
      const calls=readFileSync(log,"utf8");
      expect(calls).toContain("stop server keycloak");expect(calls).toContain("start keycloak server");
      expect(calls).toContain("stop nginx server keycloak");expect(calls).toContain("pg_restore --exit-on-error --clean --if-exists --no-owner");
      expect(calls).toContain("up -d keycloak server nginx");
    } finally {rmSync(dir,{recursive:true,force:true});}
  });
});
