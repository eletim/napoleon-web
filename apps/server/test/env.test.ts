import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadLocalEnvFile } from "../src/env.js";

describe("local .env loading", () => {
  it("loads server ONNX policy settings from a root .env file", () => {
    const root = mkdtempSync(join(tmpdir(), "napoleon-env-"));
    const env: NodeJS.ProcessEnv = {
      NAPOLEON_POLICY_1_DISPLAY_NAME: "already exported"
    };
    writeFileSync(
      join(root, ".env"),
      [
        "NAPOLEON_POLICY_1_DISPLAY_NAME=RL v900",
        "NAPOLEON_POLICY_1_ONNX_PATH=/models/v900.onnx",
        "NAPOLEON_POLICY_1_METADATA_PATH='/models/v900.json'",
        "NAPOLEON_POLICY_5_DISPLAY_NAME=\"RL v1400\"",
        "NAPOLEON_POLICY_5_ONNX_PATH=/models/v1400.onnx",
        "NAPOLEON_POLICY_5_METADATA_PATH=/models/v1400.json",
        "IGNORED_KEY=ignored"
      ].join("\n") + "\n"
    );

    loadLocalEnvFile(env, root);

    expect(env.NAPOLEON_POLICY_1_DISPLAY_NAME).toBe("already exported");
    expect(env.NAPOLEON_POLICY_1_ONNX_PATH).toBe("/models/v900.onnx");
    expect(env.NAPOLEON_POLICY_1_METADATA_PATH).toBe("/models/v900.json");
    expect(env.NAPOLEON_POLICY_5_DISPLAY_NAME).toBe("RL v1400");
    expect(env.NAPOLEON_POLICY_5_ONNX_PATH).toBe("/models/v1400.onnx");
    expect(env.NAPOLEON_POLICY_5_METADATA_PATH).toBe("/models/v1400.json");
    expect(env.IGNORED_KEY).toBeUndefined();
    rmSync(root, { recursive: true, force: true });
  });
});
