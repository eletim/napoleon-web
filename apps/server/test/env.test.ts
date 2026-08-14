import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { fullPolicyEnvKeys, learnedPolicyEnvKeys } from "../src/agentEnv.js";
import {
  FULL_POLICY_ONNX_AGENT_ID,
  PLAYING_POLICY_ONNX_AGENT_ID,
  readFullPolicyOnnxAgentConfigs,
  readPlayingPolicyOnnxAgentConfigs
} from "../src/agentRegistry.js";
import { loadLocalEnvFile } from "../src/env.js";

const envSamplePath = fileURLToPath(new URL("../../../.env.sample", import.meta.url));
const workspaceRoot = fileURLToPath(new URL("../../..", import.meta.url));

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
        "NAPOLEON_FULL_POLICY_1_DISPLAY_NAME=Full v1",
        "NAPOLEON_FULL_POLICY_1_PLAYING_ONNX_PATH=/models/full/playing.onnx",
        "NAPOLEON_FULL_POLICY_1_PLAYING_METADATA_PATH=/models/full/playing.json",
        "NAPOLEON_FULL_POLICY_1_BIDDING_ONNX_PATH=/models/full/bidding.onnx",
        "NAPOLEON_FULL_POLICY_1_BIDDING_METADATA_PATH=/models/full/bidding.json",
        "NAPOLEON_FULL_POLICY_1_ADJUTANT_ONNX_PATH=/models/full/adjutant.onnx",
        "NAPOLEON_FULL_POLICY_1_ADJUTANT_METADATA_PATH=/models/full/adjutant.json",
        "NAPOLEON_FULL_POLICY_1_EXCHANGE_ONNX_PATH=/models/full/exchange.onnx",
        "NAPOLEON_FULL_POLICY_1_EXCHANGE_METADATA_PATH=/models/full/exchange.json",
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
    expect(env.NAPOLEON_FULL_POLICY_1_DISPLAY_NAME).toBe("Full v1");
    expect(env.NAPOLEON_FULL_POLICY_1_EXCHANGE_METADATA_PATH).toBe("/models/full/exchange.json");
    expect(env.IGNORED_KEY).toBeUndefined();
    rmSync(root, { recursive: true, force: true });
  });

  it("loads every shared learned policy environment key", () => {
    const root = mkdtempSync(join(tmpdir(), "napoleon-env-"));
    const env: NodeJS.ProcessEnv = {};
    writeFileSync(
      join(root, ".env"),
      [...learnedPolicyEnvKeys, ...fullPolicyEnvKeys]
        .map((key) => `${key}=value-for-${key}`)
        .join("\n") + "\n"
    );

    loadLocalEnvFile(env, root);

    for (const key of [...learnedPolicyEnvKeys, ...fullPolicyEnvKeys]) {
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

    expect(keys).toEqual([...learnedPolicyEnvKeys, ...fullPolicyEnvKeys]);
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
        onnxPath: join(workspaceRoot, "benchmarks/playing-policies/rl-v740/policy.onnx"),
        metadataPath: join(workspaceRoot, "benchmarks/playing-policies/rl-v740/policy.json")
      }
    ]);
    expect(readFullPolicyOnnxAgentConfigs(env)).toEqual([]);
  });

  it("parses a complete full-policy slot from env", () => {
    const env = {
      NAPOLEON_FULL_POLICY_1_DISPLAY_NAME: "Full policy",
      NAPOLEON_FULL_POLICY_1_PLAYING_ONNX_PATH: "artifacts/playing.onnx",
      NAPOLEON_FULL_POLICY_1_PLAYING_METADATA_PATH: "artifacts/playing.json",
      NAPOLEON_FULL_POLICY_1_BIDDING_ONNX_PATH: "artifacts/bidding.onnx",
      NAPOLEON_FULL_POLICY_1_BIDDING_METADATA_PATH: "artifacts/bidding.json",
      NAPOLEON_FULL_POLICY_1_ADJUTANT_ONNX_PATH: "artifacts/adjutant.onnx",
      NAPOLEON_FULL_POLICY_1_ADJUTANT_METADATA_PATH: "artifacts/adjutant.json",
      NAPOLEON_FULL_POLICY_1_EXCHANGE_ONNX_PATH: "artifacts/exchange.onnx",
      NAPOLEON_FULL_POLICY_1_EXCHANGE_METADATA_PATH: "artifacts/exchange.json"
    };

    expect(readFullPolicyOnnxAgentConfigs(env, workspaceRoot)).toEqual([
      {
        id: FULL_POLICY_ONNX_AGENT_ID,
        displayName: "Full policy",
        playing: {
          onnxPath: join(workspaceRoot, "artifacts/playing.onnx"),
          metadataPath: join(workspaceRoot, "artifacts/playing.json")
        },
        bidding: {
          onnxPath: join(workspaceRoot, "artifacts/bidding.onnx"),
          metadataPath: join(workspaceRoot, "artifacts/bidding.json")
        },
        adjutant: {
          onnxPath: join(workspaceRoot, "artifacts/adjutant.onnx"),
          metadataPath: join(workspaceRoot, "artifacts/adjutant.json")
        },
        exchange: {
          onnxPath: join(workspaceRoot, "artifacts/exchange.onnx"),
          metadataPath: join(workspaceRoot, "artifacts/exchange.json")
        }
      }
    ]);
  });
});
