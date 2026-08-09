import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";

const supportedLocalEnvKeys = new Set([
  "PORT",
  "HOST",
  ...[1, 2, 3, 4, 5].flatMap((slotNumber) => [
    `NAPOLEON_POLICY_${slotNumber}_DISPLAY_NAME`,
    `NAPOLEON_POLICY_${slotNumber}_ONNX_PATH`,
    `NAPOLEON_POLICY_${slotNumber}_METADATA_PATH`
  ])
]);

export function loadLocalEnvFile(
  env: NodeJS.ProcessEnv = process.env,
  cwd = process.cwd()
): void {
  const envPath = join(findWorkspaceRoot(cwd), ".env");

  if (!existsSync(envPath)) {
    return;
  }

  for (const line of readFileSync(envPath, "utf8").split(/\r?\n/)) {
    const parsed = parseEnvLine(line);

    if (parsed === undefined || !supportedLocalEnvKeys.has(parsed.key)) {
      continue;
    }

    env[parsed.key] ??= parsed.value;
  }
}

function findWorkspaceRoot(cwd: string): string {
  let current = cwd;

  for (;;) {
    if (existsSync(join(current, "pnpm-workspace.yaml"))) {
      return current;
    }

    const parent = dirname(current);

    if (parent === current) {
      return cwd;
    }

    current = parent;
  }
}

function parseEnvLine(line: string): { key: string; value: string } | undefined {
  const trimmed = line.trim();

  if (trimmed.length === 0 || trimmed.startsWith("#")) {
    return undefined;
  }

  const withoutExport = trimmed.startsWith("export ") ? trimmed.slice("export ".length) : trimmed;
  const separatorIndex = withoutExport.indexOf("=");

  if (separatorIndex <= 0) {
    return undefined;
  }

  const key = withoutExport.slice(0, separatorIndex).trim();
  const rawValue = withoutExport.slice(separatorIndex + 1).trim();

  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
    return undefined;
  }

  return {
    key,
    value: stripMatchingQuotes(rawValue)
  };
}

function stripMatchingQuotes(value: string): string {
  if (
    (value.startsWith("\"") && value.endsWith("\"")) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1);
  }

  return value;
}
