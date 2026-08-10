import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { PolicyOnnxCompatibilityError } from "./errors.js";
import { loadPolicyOnnxModel } from "./policyOnnx.js";
import type { PolicyOnnxModel } from "./policyOnnx.js";
import { validatePolicyOnnxMetadata } from "./metadata.js";
import type { PolicyOnnxInferenceDevice, PolicyOnnxMetadata } from "./types.js";

export const RL_V740_BENCHMARK_POLICY_ID = "rl-v740" as const;

export type RepoManagedPlayingPolicyBenchmarkId = typeof RL_V740_BENCHMARK_POLICY_ID;

export interface PlayingPolicyArtifactReference {
  id: string;
  displayName: string;
  onnxPath: string;
  metadataPath: string;
  provenancePath?: string;
  onnxSha256: string;
  metadataSha256: string;
}

export interface LoadedPlayingPolicyBenchmark {
  artifact: PlayingPolicyArtifactReference;
  policy: PolicyOnnxModel;
}

const rlV740Artifact = {
  id: RL_V740_BENCHMARK_POLICY_ID,
  displayName: "RL v740",
  onnxPath: benchmarkPath("rl-v740/policy.onnx"),
  metadataPath: benchmarkPath("rl-v740/policy.json"),
  provenancePath: benchmarkPath("rl-v740/provenance.json"),
  onnxSha256: "73578d864bfdb2e368754ded8f852c7683409085f991d59251c9c4e8deec10f7",
  metadataSha256: "ae86a8d3949076d352e5caad1e45c6717bbe1785de56462ac988b65639909a91"
} as const satisfies PlayingPolicyArtifactReference;

export function getRepoManagedPlayingPolicyBenchmark(
  id: RepoManagedPlayingPolicyBenchmarkId
): PlayingPolicyArtifactReference {
  switch (id) {
    case RL_V740_BENCHMARK_POLICY_ID:
      return { ...rlV740Artifact };
  }
}

export async function loadRepoManagedPlayingPolicyBenchmark(
  id: RepoManagedPlayingPolicyBenchmarkId,
  options: { inferenceDevice?: PolicyOnnxInferenceDevice } = {}
): Promise<LoadedPlayingPolicyBenchmark> {
  const artifact = getRepoManagedPlayingPolicyBenchmark(id);
  await validatePlayingPolicyArtifactReference(artifact);
  return {
    artifact,
    policy: await loadPolicyOnnxModel({
      onnxPath: artifact.onnxPath,
      metadataPath: artifact.metadataPath,
      inferenceDevice: options.inferenceDevice
    })
  };
}

export async function validatePlayingPolicyArtifactReference(
  artifact: PlayingPolicyArtifactReference
): Promise<PolicyOnnxMetadata> {
  const [onnxSha256, metadataBytes] = await Promise.all([
    calculateFileSha256(artifact.onnxPath),
    readFile(artifact.metadataPath)
  ]);
  const metadataSha256 = sha256(metadataBytes);

  if (onnxSha256 !== artifact.onnxSha256) {
    throw new PolicyOnnxCompatibilityError(
      `playing benchmark artifact ${artifact.id} ONNX SHA256 mismatch: ` +
      `expected ${artifact.onnxSha256}, got ${onnxSha256}.`
    );
  }
  if (metadataSha256 !== artifact.metadataSha256) {
    throw new PolicyOnnxCompatibilityError(
      `playing benchmark artifact ${artifact.id} metadata SHA256 mismatch: ` +
      `expected ${artifact.metadataSha256}, got ${metadataSha256}.`
    );
  }

  const metadata = JSON.parse(new TextDecoder().decode(metadataBytes)) as unknown;
  validatePolicyOnnxMetadata(metadata);
  return metadata;
}

async function calculateFileSha256(path: string): Promise<string> {
  return sha256(await readFile(path));
}

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function benchmarkPath(relativePath: string): string {
  return fileURLToPath(new URL(`../../../benchmarks/playing-policies/${relativePath}`, import.meta.url));
}
