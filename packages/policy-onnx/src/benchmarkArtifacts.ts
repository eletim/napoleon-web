import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { PolicyOnnxCompatibilityError } from "./errors.js";
import { loadPolicyCriticOnnxModel, loadPolicyOnnxModel } from "./policyOnnx.js";
import type { PolicyCriticOnnxModel, PolicyOnnxModel } from "./policyOnnx.js";
import { validatePolicyCriticOnnxMetadata, validatePolicyOnnxMetadata } from "./metadata.js";
import type { PolicyOnnxInferenceDevice, PolicyOnnxMetadata } from "./types.js";

export const RL_V740_BENCHMARK_POLICY_ID = "rl-v740" as const;
export const PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID = "ppo-separated-v1000" as const;

export type RepoManagedPlayingPolicyBenchmarkId =
  | typeof RL_V740_BENCHMARK_POLICY_ID
  | typeof PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID;

export interface PlayingPolicyArtifactReference {
  id: string;
  displayName: string;
  onnxPath: string;
  metadataPath: string;
  provenancePath?: string;
  checkpointPath?: string;
  checkpointSha256?: string;
  criticOnnxPath?: string;
  criticMetadataPath?: string;
  criticOnnxSha256?: string;
  criticMetadataSha256?: string;
  onnxSha256: string;
  metadataSha256: string;
}

export interface LoadedPlayingPolicyBenchmark {
  artifact: PlayingPolicyArtifactReference;
  policy: PolicyOnnxModel;
  critic?: PolicyCriticOnnxModel;
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

const ppoSeparatedV1000Artifact = {
  id: PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID,
  displayName: "PPO separated v1000",
  onnxPath: benchmarkPath("ppo-separated-v1000/policy.onnx"),
  metadataPath: benchmarkPath("ppo-separated-v1000/policy.json"),
  provenancePath: benchmarkPath("ppo-separated-v1000/provenance.json"),
  checkpointPath: benchmarkPath("ppo-separated-v1000/checkpoint.pt"),
  criticOnnxPath: benchmarkPath("ppo-separated-v1000/critic.onnx"),
  criticMetadataPath: benchmarkPath("ppo-separated-v1000/critic.json"),
  checkpointSha256: "36c543b8e3026283269fd40b382abf12aeb085296a8de52e52d3bf65b4c24376",
  criticOnnxSha256: "3055882f3e63e2a096ee7cedee341bc97e033572bcb59f36f3f68e3d89f134d9",
  criticMetadataSha256: "7d5e4d6d785666c1da26bc5e59bb3f92c8b7dd592e2a91dd747ddefc3ea7cdaa",
  onnxSha256: "54d7ba29222a12e99a91ab61ee7aa253fe3fab73200d78167d64bf9e7bb8887e",
  metadataSha256: "54f0f2837f0e0bad81c778114ab996259b5f3a05bda338a12d0fb32b1fb50616"
} as const satisfies PlayingPolicyArtifactReference;

export function getRepoManagedPlayingPolicyBenchmark(
  id: RepoManagedPlayingPolicyBenchmarkId
): PlayingPolicyArtifactReference {
  switch (id) {
    case RL_V740_BENCHMARK_POLICY_ID:
      return { ...rlV740Artifact };
    case PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID:
      return { ...ppoSeparatedV1000Artifact };
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
    }),
    ...(artifact.criticOnnxPath === undefined || artifact.criticMetadataPath === undefined
      ? {}
      : {
          critic: await loadPolicyCriticOnnxModel({
            onnxPath: artifact.criticOnnxPath,
            metadataPath: artifact.criticMetadataPath,
            inferenceDevice: options.inferenceDevice
          })
        })
  };
}

export async function validatePlayingPolicyArtifactReference(
  artifact: PlayingPolicyArtifactReference
): Promise<PolicyOnnxMetadata> {
  const [onnxSha256, metadataBytes, checkpointSha256, criticOnnxSha256, criticMetadataBytes] = await Promise.all([
    calculateFileSha256(artifact.onnxPath),
    readFile(artifact.metadataPath),
    artifact.checkpointPath === undefined ? Promise.resolve(undefined) : calculateFileSha256(artifact.checkpointPath),
    artifact.criticOnnxPath === undefined ? Promise.resolve(undefined) : calculateFileSha256(artifact.criticOnnxPath),
    artifact.criticMetadataPath === undefined ? Promise.resolve(undefined) : readFile(artifact.criticMetadataPath)
  ]);
  const metadataSha256 = sha256(metadataBytes);
  const criticMetadataSha256 = criticMetadataBytes === undefined ? undefined : sha256(criticMetadataBytes);

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
  if (
    artifact.checkpointSha256 !== undefined &&
    checkpointSha256 !== artifact.checkpointSha256
  ) {
    throw new PolicyOnnxCompatibilityError(
      `playing benchmark artifact ${artifact.id} checkpoint SHA256 mismatch: ` +
      `expected ${artifact.checkpointSha256}, got ${checkpointSha256}.`
    );
  }
  if (
    artifact.criticOnnxSha256 !== undefined &&
    criticOnnxSha256 !== artifact.criticOnnxSha256
  ) {
    throw new PolicyOnnxCompatibilityError(
      `playing benchmark artifact ${artifact.id} critic ONNX SHA256 mismatch: ` +
      `expected ${artifact.criticOnnxSha256}, got ${criticOnnxSha256}.`
    );
  }
  if (
    artifact.criticMetadataSha256 !== undefined &&
    criticMetadataSha256 !== artifact.criticMetadataSha256
  ) {
    throw new PolicyOnnxCompatibilityError(
      `playing benchmark artifact ${artifact.id} critic metadata SHA256 mismatch: ` +
      `expected ${artifact.criticMetadataSha256}, got ${criticMetadataSha256}.`
    );
  }

  const metadata = JSON.parse(new TextDecoder().decode(metadataBytes)) as unknown;
  validatePolicyOnnxMetadata(metadata);
  if (criticMetadataBytes !== undefined) {
    const criticMetadata = JSON.parse(new TextDecoder().decode(criticMetadataBytes)) as unknown;
    validatePolicyCriticOnnxMetadata(criticMetadata);
  }
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
