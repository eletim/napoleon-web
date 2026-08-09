import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { learnedPolicyEnvKeys } from "../src/agentEnv.js";
import {
  PLAYING_POLICY_ONNX_AGENT_ID,
  readPlayingPolicyOnnxAgentConfigs
} from "../src/agentRegistry.js";
import { loadLocalEnvFile } from "../src/env.js";

const envSamplePath = fileURLToPath(new URL("../../../.env.sample", import.meta.url));

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

  it("loads every shared learned policy environment key", () => {
    const root = mkdtempSync(join(tmpdir(), "napoleon-env-"));
    const env: NodeJS.ProcessEnv = {};
    writeFileSync(
      join(root, ".env"),
      learnedPolicyEnvKeys.map((key) => `${key}=value-for-${key}`).join("\n") + "\n"
    );

    loadLocalEnvFile(env, root);

    for (const key of learnedPolicyEnvKeys) {
      expect(env[key]).toBe(`value-for-${key}`);
    }

    rmSync(root, { recursive: true, force: true });
  });

  it("keeps the committed .env.sample in sync with supported learned policy keys", () => {
    const keys = readFileSync(envSamplePath, "utf8")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0 && !line.startsWith("#"))
      .map((line) => line.split("=")[0]);

    expect(keys).toEqual(learnedPolicyEnvKeys);
  });

  it("uses the committed .env.sample to enable only RL v740 by default", () => {
    const env = Object.fromEntries(
      readFileSync(envSamplePath, "utf8")
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line.length > 0 && !line.startsWith("#"))
        .map((line) => {
          const separatorIndex = line.indexOf("=");

          return [line.slice(0, separatorIndex), line.slice(separatorIndex + 1)];
        })
    );

    expect(readPlayingPolicyOnnxAgentConfigs(env)).toEqual([
      {
        id: PLAYING_POLICY_ONNX_AGENT_ID,
        displayName: "RL v740",
        onnxPath: "benchmarks/playing-policies/rl-v740/policy.onnx",
        metadataPath: "benchmarks/playing-policies/rl-v740/policy.json"
      }
    ]);
  });
});
