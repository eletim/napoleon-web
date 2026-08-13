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
const envSamplePath = fileURLToPath(new URL("../../.env.sample", import.meta.url));

describe("start-dev.sh", () => {
  it("creates .env from .env.sample on first run", () => {
    const root = createTempRoot();
    const sample = readFileSync(envSamplePath, "utf8");
    const envFile = join(root, ".env");
    writeFileSync(join(root, ".env.sample"), sample);

    const result = runDevScript(root);

    expect(result.status).toBe(0);
    expect(readFileSync(envFile, "utf8")).toBe(sample);
    expect(result.stdout).toContain(".env を .env.sample から生成しました。");
    expect(result.stdout).toContain("root_env_file_exists=true");
    expect(result.stdout).toContain("learned_policy_slots_configured=1");
    expect(result.stdout).toContain("learned_policy_slots_incomplete=0");
    expect(result.stdout).not.toContain("/home/eletim/napoleon_runs");
    rmSync(root, { recursive: true, force: true });
  });

  it("never overwrites an existing .env when .env.sample is present", () => {
    const root = createTempRoot();
    const envFile = join(root, ".env");
    const existingEnv = [
      "NAPOLEON_POLICY_1_DISPLAY_NAME=Local RL",
      "NAPOLEON_POLICY_1_ONNX_PATH=/private/local/policy.onnx",
      "NAPOLEON_POLICY_1_METADATA_PATH=/private/local/policy.json"
    ].join("\n") + "\n";
    writeFileSync(join(root, ".env.sample"), readFileSync(envSamplePath, "utf8"));
    writeFileSync(envFile, existingEnv);

    const result = runDevScript(root);

    expect(result.status).toBe(0);
    expect(readFileSync(envFile, "utf8")).toBe(existingEnv);
    expect(result.stdout).not.toContain(".env を .env.sample から生成しました。");
    expect(result.stdout).toContain("learned_policy_slots_configured=1");
    rmSync(root, { recursive: true, force: true });
  });

  it("loads learned ONNX policy slots from the root .env without printing local paths", () => {
    const root = createTempRoot();
    writeFileSync(
      join(root, ".env"),
      [
        "NAPOLEON_POLICY_1_DISPLAY_NAME=RL v900",
        "NAPOLEON_POLICY_1_ONNX_PATH=/private/models/v900.onnx",
        "NAPOLEON_POLICY_1_METADATA_PATH=/private/models/v900.json",
        "NAPOLEON_POLICY_2_DISPLAY_NAME=",
        "NAPOLEON_POLICY_2_ONNX_PATH=/private/models/unused.onnx",
        "NAPOLEON_POLICY_2_METADATA_PATH=/private/models/unused.json",
        "NAPOLEON_POLICY_5_DISPLAY_NAME=RL v1400",
        "NAPOLEON_POLICY_5_ONNX_PATH=/private/models/v1400.onnx",
        "NAPOLEON_POLICY_5_METADATA_PATH=/private/models/v1400.json"
      ].join("\n") + "\n"
    );

    const result = runDevScript(root);

    expect(result.status).toBe(0);
    expect(result.stdout).toContain("root_env_file_exists=true");
    expect(result.stdout).toContain("learned_policy_slots_configured=2");
    expect(result.stdout).toContain("learned_policy_slots_incomplete=0");
    expect(result.stdout).not.toContain("/private/models");
    rmSync(root, { recursive: true, force: true });
  });

  it("reports incomplete learned ONNX slots during dry-run", () => {
    const root = createTempRoot();
    writeFileSync(
      join(root, ".env"),
      [
        "NAPOLEON_POLICY_1_DISPLAY_NAME=Broken RL",
        "NAPOLEON_POLICY_1_ONNX_PATH=/private/models/broken.onnx"
      ].join("\n") + "\n"
    );

    const result = runDevScript(root);

    expect(result.status).toBe(0);
    expect(result.stdout).toContain("learned_policy_slots_configured=0");
    expect(result.stdout).toContain("learned_policy_slots_incomplete=1");
    rmSync(root, { recursive: true, force: true });
  });

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
    expect(result.stdout).toContain("NAPOLEON_WEB_BASE_PATH=/napoleon/");
    expect(result.stdout).not.toContain("生成しますか");
    rmSync(root, { recursive: true, force: true });
  });

  it("automatically creates .env.local from Tailscale Self.DNSName", () => {
    const root = createTempRoot();
    const envFile = join(root, "apps/web/.env.local");
    const fakeTailscale = createFakeTailscale({
      selfDnsName: "e-ryzen.tail6bc726.ts.net."
    });

    const result = runDevScript(root, {
      path: fakeTailscale.path
    });

    expect(result.status).toBe(0);
    expect(readFileSync(envFile, "utf8")).toBe(
      "VITE_ALLOWED_HOSTS=e-ryzen.tail6bc726.ts.net\n"
    );
    expect(result.stdout).toContain("apps/web/.env.local をTailscale DNS名から生成しました。");
    expect(result.stdout).toContain("env_file_exists=true");
    expect(result.stdout).toContain("VITE_ALLOWED_HOSTS=e-ryzen.tail6bc726.ts.net");
    expect(result.stdout).not.toContain("生成しますか");
    expect(result.stdout).not.toContain("許可するホスト名を入力してください");
    expect(readFakeTailscaleLog(fakeTailscale.logPath)).toEqual(["status --json"]);
    rmSync(root, { recursive: true, force: true });
    rmSync(fakeTailscale.root, { recursive: true, force: true });
  });

  it("falls back to localhost-only when Tailscale is not installed", () => {
    const root = createTempRoot();
    const envFile = join(root, "apps/web/.env.local");

    const result = runDevScript(root, {
      path: createPathWithoutTailscale()
    });

    expect(result.status).toBe(0);
    expect(existsSync(envFile)).toBe(false);
    expect(result.stdout).toContain("env_file_exists=false");
    expect(result.stdout).toContain("VITE_ALLOWED_HOSTS=\n");
    expect(result.stdout).toContain("NAPOLEON_WEB_BASE_PATH=\n");
    expect(result.stdout).toContain("tailscale_serve_enabled=false");
    expect(result.stdout).not.toContain("生成しますか");
    rmSync(root, { recursive: true, force: true });
  });

  it("falls back to localhost-only when Tailscale status json fails", () => {
    const root = createTempRoot();
    const envFile = join(root, "apps/web/.env.local");
    const fakeTailscale = createFakeTailscale({ statusExitCode: 1 });

    const result = runDevScript(root, {
      dryRun: false,
      testMode: true,
      path: fakeTailscale.path
    });

    expect(result.status).toBe(0);
    expect(existsSync(envFile)).toBe(false);
    expect(result.stdout).toContain("Tailscale DNS名を取得できないため");
    expect(result.stdout).toContain("dev_server_started=true");
    expect(readFakeTailscaleLog(fakeTailscale.logPath)).toEqual(["status --json"]);
    rmSync(root, { recursive: true, force: true });
    rmSync(fakeTailscale.root, { recursive: true, force: true });
  });

  it("falls back to localhost-only when Tailscale is disconnected", () => {
    const root = createTempRoot();
    const envFile = join(root, "apps/web/.env.local");
    const fakeTailscale = createFakeTailscale({ backendState: "NeedsLogin" });

    const result = runDevScript(root, {
      dryRun: false,
      testMode: true,
      path: fakeTailscale.path
    });

    expect(result.status).toBe(0);
    expect(existsSync(envFile)).toBe(false);
    expect(result.stdout).toContain("Tailscale DNS名を取得できないため");
    expect(result.stdout).toContain("dev_server_started=true");
    expect(readFakeTailscaleLog(fakeTailscale.logPath)).toEqual(["status --json"]);
    rmSync(root, { recursive: true, force: true });
    rmSync(fakeTailscale.root, { recursive: true, force: true });
  });

  it("falls back to localhost-only when Tailscale Self.DNSName is missing", () => {
    const root = createTempRoot();
    const envFile = join(root, "apps/web/.env.local");
    const fakeTailscale = createFakeTailscale({ selfDnsName: null });

    const result = runDevScript(root, {
      dryRun: false,
      testMode: true,
      path: fakeTailscale.path
    });

    expect(result.status).toBe(0);
    expect(existsSync(envFile)).toBe(false);
    expect(result.stdout).toContain("Tailscale DNS名を取得できないため");
    expect(result.stdout).toContain("dev_server_started=true");
    expect(readFakeTailscaleLog(fakeTailscale.logPath)).toEqual(["status --json"]);
    rmSync(root, { recursive: true, force: true });
    rmSync(fakeTailscale.root, { recursive: true, force: true });
  });

  it("does not overwrite or prompt for an existing .env.local", () => {
    const root = createTempRoot();
    const envFile = join(root, "apps/web/.env.local");
    const fakeTailscale = createFakeTailscale();
    writeFileSync(envFile, "VITE_ALLOWED_HOSTS=existing.example\n");

    const result = runDevScript(root, {
      path: fakeTailscale.path
    });

    expect(result.status).toBe(0);
    expect(readFileSync(envFile, "utf8")).toBe("VITE_ALLOWED_HOSTS=existing.example\n");
    expect(result.stdout).not.toContain("生成しますか");
    expect(existsSync(fakeTailscale.logPath)).toBe(false);
    rmSync(root, { recursive: true, force: true });
    rmSync(fakeTailscale.root, { recursive: true, force: true });
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
      "serve --bg --https=443 --set-path=/napoleon/ http://127.0.0.1:5173"
    ]);
    rmSync(root, { recursive: true, force: true });
    rmSync(fakeTailscale.root, { recursive: true, force: true });
  });

  it("starts localhost-only when automatic host configuration is unavailable", () => {
    const root = createTempRoot();

    const result = runDevScript(root, {
      dryRun: false,
      testMode: true,
      path: createPathWithoutTailscale()
    });

    expect(result.status).toBe(0);
    expect(result.stdout).toContain("dev_server_started=true");
    expect(result.stdout).toContain("Tailscale DNS名を取得できないため");
    rmSync(root, { recursive: true, force: true });
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
      "serve --bg --https=443 --set-path=/napoleon/ http://127.0.0.1:5173"
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
    expect(result.stdout).toContain("NAPOLEON_WEB_BASE_PATH=/napoleon/");
    expect(result.stdout).toContain(
      "tailscale_serve_command=tailscale serve --bg --https=443 --set-path=/napoleon/ http://127.0.0.1:5173"
    );
    expect(result.stdout).toContain("dev_server_started=false");
    expect(existsSync(fakeTailscale.logPath)).toBe(false);
    rmSync(root, { recursive: true, force: true });
    rmSync(fakeTailscale.root, { recursive: true, force: true });
  });

  it("normalizes a custom Tailscale dev base path override", () => {
    const root = createTempRoot();
    const fakeTailscale = createFakeTailscale();
    writeFileSync(join(root, "apps/web/.env.local"), "VITE_ALLOWED_HOSTS=my-machine.example.ts.net\n");

    const result = runDevScript(root, {
      basePath: "custom-dev",
      path: fakeTailscale.path
    });

    expect(result.status).toBe(0);
    expect(result.stdout).toContain("NAPOLEON_WEB_BASE_PATH=/custom-dev/");
    expect(result.stdout).toContain(
      "tailscale_serve_command=tailscale serve --bg --https=443 --set-path=/custom-dev/ http://127.0.0.1:5173"
    );
    rmSync(root, { recursive: true, force: true });
    rmSync(fakeTailscale.root, { recursive: true, force: true });
  });

  it("prints disabled Tailscale state during dry-run without hosts", () => {
    const root = createTempRoot();

    const result = runDevScript(root);

    expect(result.status).toBe(0);
    expect(result.stdout).toContain("tailscale_serve_enabled=false");
    expect(result.stdout).toContain("NAPOLEON_WEB_BASE_PATH=\n");
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
    basePath?: string;
    dryRun?: boolean;
    input?: string;
    path?: string;
    testMode?: boolean;
  } = {}
): SpawnSyncReturns<string> {
  const env = { ...process.env };
  delete env.VITE_ALLOWED_HOSTS;
  for (let slot = 1; slot <= 5; slot += 1) {
    delete env[`NAPOLEON_POLICY_${slot}_DISPLAY_NAME`];
    delete env[`NAPOLEON_POLICY_${slot}_ONNX_PATH`];
    delete env[`NAPOLEON_POLICY_${slot}_METADATA_PATH`];
  }
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
  if (options.basePath !== undefined) {
    env.NAPOLEON_DEV_BASE_PATH = options.basePath;
  } else {
    delete env.NAPOLEON_DEV_BASE_PATH;
  }
  env.PATH = options.path ?? createPathWithoutTailscale();

  delete env.NAPOLEON_DEV_FORCE_INTERACTIVE;

  return spawnSync("/bin/bash", [scriptPath], {
    cwd: root,
    encoding: "utf8",
    env,
    input: options.input ?? ""
  });
}

function createFakeTailscale(
  options: {
    backendState?: string;
    selfDnsName?: string | null;
    serveExitCode?: number;
    statusExitCode?: number;
  } = {}
): { logPath: string; path: string; root: string } {
  const root = mkdtempSync(join(tmpdir(), "napoleon-tailscale-"));
  const binDir = join(root, "bin");
  const logPath = join(root, "tailscale.log");
  const tailscalePath = join(binDir, "tailscale");
  const statusJson =
    options.selfDnsName === null
      ? JSON.stringify({ BackendState: options.backendState ?? "Running", Self: {} })
      : JSON.stringify({
          BackendState: options.backendState ?? "Running",
          Self: { DNSName: options.selfDnsName ?? "host.example.ts.net." }
        });
  mkdirSync(binDir, { recursive: true });
  writeFileSync(
    tailscalePath,
    [
      "#!/bin/bash",
      `printf "%s\\n" "$*" >> "${logPath}"`,
      'if [[ "$1" == "status" ]]; then',
      '  if [[ "$2" == "--json" ]]; then',
      `    printf '%s\\n' '${statusJson}'`,
      "  fi",
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
  for (const command of ["cp", "dirname", "grep", "mkdir", "node", "pwd", "tail"]) {
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
