/// <reference types="node" />

import { spawnSync } from "node:child_process";
import type { SpawnSyncReturns } from "node:child_process";
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const scriptPath = fileURLToPath(new URL("../../start-dev.sh", import.meta.url));
const packageJsonPath = fileURLToPath(new URL("../../package.json", import.meta.url));
const viteConfigPath = fileURLToPath(new URL("./vite.config.ts", import.meta.url));

describe("start-dev.sh", () => {
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

  it("configures Tailscale Serve once when hosts are configured", () => {
    const root = createTempRoot();
    const fakeTailscale = createFakeTailscale();
    writeFileSync(join(root, "apps/web/.env.local"), "VITE_ALLOWED_HOSTS=host.example\n");

    const result = runDevScript(root, {
      dryRun: false,
      testMode: true,
      path: fakeTailscale.path
    });

    expect(result.status).toBe(0);
    expect(result.stdout).toContain("dev_server_started=true");
    expect(readFakeTailscaleLog(fakeTailscale.logPath)).toEqual([
      "status",
      "serve --bg --http=5173 http://127.0.0.1:5173"
    ]);
    rmSync(root, { recursive: true, force: true });
    rmSync(fakeTailscale.root, { recursive: true, force: true });
  });

  it("does not call Tailscale when hosts are not configured", () => {
    const root = createTempRoot();
    const fakeTailscale = createFakeTailscale();

    const result = runDevScript(root, {
      dryRun: false,
      testMode: true,
      path: fakeTailscale.path
    });

    expect(result.status).toBe(0);
    expect(result.stdout).toContain("dev_server_started=true");
    expect(existsSync(fakeTailscale.logPath)).toBe(false);
    rmSync(root, { recursive: true, force: true });
    rmSync(fakeTailscale.root, { recursive: true, force: true });
  });

  it("runs Tailscale Serve once for multiple allowed hosts", () => {
    const root = createTempRoot();
    const fakeTailscale = createFakeTailscale();
    writeFileSync(join(root, "apps/web/.env.local"), "VITE_ALLOWED_HOSTS=a.example,b.example,c.example\n");

    const result = runDevScript(root, {
      dryRun: false,
      testMode: true,
      path: fakeTailscale.path
    });

    expect(result.status).toBe(0);
    expect(readFakeTailscaleLog(fakeTailscale.logPath).filter((line) => line.startsWith("serve "))).toHaveLength(1);
    rmSync(root, { recursive: true, force: true });
    rmSync(fakeTailscale.root, { recursive: true, force: true });
  });

  it("fails before starting dev when tailscale is not installed and hosts are configured", () => {
    const root = createTempRoot();
    writeFileSync(join(root, "apps/web/.env.local"), "VITE_ALLOWED_HOSTS=host.example\n");

    const result = runDevScript(root, {
      dryRun: false,
      testMode: true,
      path: createPathWithoutTailscale()
    });

    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain("エラー: tailscaleコマンドが見つかりません。");
    expect(result.stdout).not.toContain("dev_server_started=true");
    rmSync(root, { recursive: true, force: true });
  });

  it("fails before Serve or dev startup when tailscale status fails", () => {
    const root = createTempRoot();
    const fakeTailscale = createFakeTailscale({ statusExitCode: 1 });
    writeFileSync(join(root, "apps/web/.env.local"), "VITE_ALLOWED_HOSTS=host.example\n");

    const result = runDevScript(root, {
      dryRun: false,
      testMode: true,
      path: fakeTailscale.path
    });

    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain("エラー: Tailscaleが接続されていません。");
    expect(result.stdout).not.toContain("dev_server_started=true");
    expect(readFakeTailscaleLog(fakeTailscale.logPath)).toEqual(["status"]);
    rmSync(root, { recursive: true, force: true });
    rmSync(fakeTailscale.root, { recursive: true, force: true });
  });

  it("fails before dev startup when tailscale serve fails", () => {
    const root = createTempRoot();
    const fakeTailscale = createFakeTailscale({ serveExitCode: 1 });
    writeFileSync(join(root, "apps/web/.env.local"), "VITE_ALLOWED_HOSTS=host.example\n");

    const result = runDevScript(root, {
      dryRun: false,
      testMode: true,
      path: fakeTailscale.path
    });

    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain("エラー: Tailscale Serveの設定に失敗しました。");
    expect(result.stdout).not.toContain("dev_server_started=true");
    expect(readFakeTailscaleLog(fakeTailscale.logPath)).toEqual([
      "status",
      "serve --bg --http=5173 http://127.0.0.1:5173"
    ]);
    rmSync(root, { recursive: true, force: true });
    rmSync(fakeTailscale.root, { recursive: true, force: true });
  });

  it("does not call Tailscale during dry-run and prints the planned Serve command", () => {
    const root = createTempRoot();
    const fakeTailscale = createFakeTailscale();
    writeFileSync(join(root, "apps/web/.env.local"), "VITE_ALLOWED_HOSTS=my-machine.example.ts.net\n");

    const result = runDevScript(root, {
      path: fakeTailscale.path
    });

    expect(result.status).toBe(0);
    expect(result.stdout).toContain("tailscale_serve_enabled=true");
    expect(result.stdout).toContain(
      "tailscale_serve_command=tailscale serve --bg --http=5173 http://127.0.0.1:5173"
    );
    expect(result.stdout).toContain("dev_server_started=false");
    expect(existsSync(fakeTailscale.logPath)).toBe(false);
    rmSync(root, { recursive: true, force: true });
    rmSync(fakeTailscale.root, { recursive: true, force: true });
  });

  it("prints disabled Tailscale state during dry-run without hosts", () => {
    const root = createTempRoot();

    const result = runDevScript(root);

    expect(result.status).toBe(0);
    expect(result.stdout).toContain("tailscale_serve_enabled=false");
    expect(result.stdout).toContain("dev_server_started=false");
    rmSync(root, { recursive: true, force: true });
  });

  it("keeps pnpm dev on the root startup script and dev:raw non-recursive", () => {
    const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf8")) as {
      scripts: Record<string, string>;
    };
    const script = readFileSync(scriptPath, "utf8");

    expect(packageJson.scripts.dev).toBe("./start-dev.sh");
    expect(packageJson.scripts["dev:raw"]).toBe(
      "pnpm --parallel --filter @napoleon/server --filter @napoleon/web dev"
    );
    expect(packageJson.scripts["dev:raw"]).not.toContain("start-dev.sh");
    expect(script.trimEnd().endsWith("exec pnpm dev:raw")).toBe(true);
  });

  it("disables Vite env-file loading", () => {
    const config = readFileSync(viteConfigPath, "utf8");

    expect(config).not.toContain(["load", "Env"].join(""));
    expect(config).toContain("envDir: false");
  });
});

function createTempRoot(): string {
  const root = mkdtempSync(join(tmpdir(), "napoleon-dev-"));
  mkdirSync(join(root, "apps/web"), { recursive: true });
  return root;
}

function runDevScript(
  root: string,
  options: {
    dryRun?: boolean;
    forceInteractive?: boolean;
    input?: string;
    path?: string;
    testMode?: boolean;
  } = {}
): SpawnSyncReturns<string> {
  const env = { ...process.env };
  delete env.VITE_ALLOWED_HOSTS;
  env.NAPOLEON_DEV_ROOT = root;
  if (options.dryRun !== false) {
    env.NAPOLEON_DEV_DRY_RUN = "1";
  } else {
    delete env.NAPOLEON_DEV_DRY_RUN;
  }
  if (options.testMode === true) {
    env.NAPOLEON_DEV_TEST_MODE = "1";
  } else {
    delete env.NAPOLEON_DEV_TEST_MODE;
  }
  if (options.path !== undefined) {
    env.PATH = options.path;
  }

  if (options.forceInteractive === true) {
    env.NAPOLEON_DEV_FORCE_INTERACTIVE = "1";
  } else {
    delete env.NAPOLEON_DEV_FORCE_INTERACTIVE;
  }

  return spawnSync("/bin/bash", [scriptPath], {
    cwd: root,
    encoding: "utf8",
    env,
    input: options.input ?? ""
  });
}

function createFakeTailscale(
  options: { serveExitCode?: number; statusExitCode?: number } = {}
): { logPath: string; path: string; root: string } {
  const root = mkdtempSync(join(tmpdir(), "napoleon-tailscale-"));
  const binDir = join(root, "bin");
  const logPath = join(root, "tailscale.log");
  const tailscalePath = join(binDir, "tailscale");
  mkdirSync(binDir, { recursive: true });
  writeFileSync(
    tailscalePath,
    [
      "#!/bin/bash",
      `printf "%s\\n" "$*" >> "${logPath}"`,
      'if [[ "$1" == "status" ]]; then',
      `  exit ${options.statusExitCode ?? 0}`,
      "fi",
      'if [[ "$1" == "serve" ]]; then',
      `  exit ${options.serveExitCode ?? 0}`,
      "fi",
      "exit 0"
    ].join("\n") + "\n"
  );
  chmodSync(tailscalePath, 0o755);

  return {
    logPath,
    path: `${binDir}:${createPathWithoutTailscale()}`,
    root
  };
}

function createPathWithoutTailscale(): string {
  const root = mkdtempSync(join(tmpdir(), "napoleon-path-"));
  const binDir = join(root, "bin");
  mkdirSync(binDir, { recursive: true });
  for (const command of ["dirname", "grep", "pwd", "tail"]) {
    symlinkSync(findCommand(command), join(binDir, command));
  }
  return binDir;
}

function findCommand(command: string): string {
  for (const directory of (process.env.PATH ?? "").split(":")) {
    const candidate = join(directory, command);
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  throw new Error(`Unable to find required test command: ${command}`);
}

function readFakeTailscaleLog(logPath: string): string[] {
  return readFileSync(logPath, "utf8").trim().split("\n");
}
