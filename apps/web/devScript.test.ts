/// <reference types="node" />

import { spawnSync } from "node:child_process";
import type { SpawnSyncReturns } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const scriptPath = fileURLToPath(new URL("../../scripts/dev.sh", import.meta.url));
const viteConfigPath = fileURLToPath(new URL("./vite.config.ts", import.meta.url));

describe("scripts/dev.sh", () => {
  it("loads VITE_ALLOWED_HOSTS from an existing .env.local", () => {
    const root = createTempRoot();
    const envFile = join(root, "apps/web/.env.local");
    writeFileSync(
      envFile,
      [
        "# Local development settings",
        "VITE_ALLOWED_HOSTS=old.example",
        "OTHER_VALUE=ignored",
        "VITE_ALLOWED_HOSTS=test-host.example"
      ].join("\n") + "\n"
    );

    const result = runDevScript(root);

    expect(result.status).toBe(0);
    expect(result.stdout).toContain("env_file_exists=true");
    expect(result.stdout).toContain("VITE_ALLOWED_HOSTS=test-host.example");
    expect(result.stdout).not.toContain("生成しますか");
    rmSync(root, { recursive: true, force: true });
  });

  it("does not prompt or create .env.local without a TTY", () => {
    const root = createTempRoot();
    const envFile = join(root, "apps/web/.env.local");

    const result = runDevScript(root);

    expect(result.status).toBe(0);
    expect(result.stdout).toContain("env_file_exists=false");
    expect(result.stdout).toContain("VITE_ALLOWED_HOSTS=\n");
    expect(result.stdout).not.toContain("生成しますか");
    expect(existsSync(envFile)).toBe(false);
    rmSync(root, { recursive: true, force: true });
  });

  it("creates a normalized .env.local for interactive Yes input", () => {
    const root = createTempRoot();
    const envFile = join(root, "apps/web/.env.local");

    const result = runDevScript(root, {
      forceInteractive: true,
      input: "y\nhost-a.example, host-b.example,,host-c.example\n"
    });

    expect(result.status).toBe(0);
    expect(readFileSync(envFile, "utf8")).toBe(
      "VITE_ALLOWED_HOSTS=host-a.example,host-b.example,host-c.example\n"
    );
    expect(result.stdout).toContain("VITE_ALLOWED_HOSTS=host-a.example,host-b.example,host-c.example");
    rmSync(root, { recursive: true, force: true });
  });

  it("does not create .env.local for interactive No input", () => {
    const root = createTempRoot();
    const envFile = join(root, "apps/web/.env.local");

    const result = runDevScript(root, {
      forceInteractive: true,
      input: "n\n"
    });

    expect(result.status).toBe(0);
    expect(existsSync(envFile)).toBe(false);
    expect(result.stdout).toContain("env_file_exists=false");
    rmSync(root, { recursive: true, force: true });
  });

  it("does not create .env.local when interactive Yes input has no hosts", () => {
    const root = createTempRoot();
    const envFile = join(root, "apps/web/.env.local");

    const result = runDevScript(root, {
      forceInteractive: true,
      input: "Y\n , , \n"
    });

    expect(result.status).toBe(0);
    expect(existsSync(envFile)).toBe(false);
    expect(result.stdout).toContain("VITE_ALLOWED_HOSTS=\n");
    rmSync(root, { recursive: true, force: true });
  });

  it("preserves multiple host input order", () => {
    const root = createTempRoot();
    const envFile = join(root, "apps/web/.env.local");

    const result = runDevScript(root, {
      forceInteractive: true,
      input: "y\nz.example, a.example, m.example\n"
    });

    expect(result.status).toBe(0);
    expect(readFileSync(envFile, "utf8")).toBe(
      "VITE_ALLOWED_HOSTS=z.example,a.example,m.example\n"
    );
    rmSync(root, { recursive: true, force: true });
  });

  it("does not overwrite or prompt for an existing .env.local", () => {
    const root = createTempRoot();
    const envFile = join(root, "apps/web/.env.local");
    writeFileSync(envFile, "VITE_ALLOWED_HOSTS=existing.example\n");

    const result = runDevScript(root, {
      forceInteractive: true,
      input: "y\nnew.example\n"
    });

    expect(result.status).toBe(0);
    expect(readFileSync(envFile, "utf8")).toBe("VITE_ALLOWED_HOSTS=existing.example\n");
    expect(result.stdout).not.toContain("生成しますか");
    rmSync(root, { recursive: true, force: true });
  });

  it("does not use Vite env-file loading", () => {
    expect(readFileSync(viteConfigPath, "utf8")).not.toContain(["load", "Env"].join(""));
  });
});

function createTempRoot(): string {
  const root = mkdtempSync(join(tmpdir(), "napoleon-dev-"));
  mkdirSync(join(root, "apps/web"), { recursive: true });
  return root;
}

function runDevScript(
  root: string,
  options: { forceInteractive?: boolean; input?: string } = {}
): SpawnSyncReturns<string> {
  const env = { ...process.env };
  delete env.VITE_ALLOWED_HOSTS;
  env.NAPOLEON_DEV_ROOT = root;
  env.NAPOLEON_DEV_DRY_RUN = "1";

  if (options.forceInteractive === true) {
    env.NAPOLEON_DEV_FORCE_INTERACTIVE = "1";
  } else {
    delete env.NAPOLEON_DEV_FORCE_INTERACTIVE;
  }

  return spawnSync("bash", [scriptPath], {
    cwd: root,
    encoding: "utf8",
    env,
    input: options.input ?? ""
  });
}
