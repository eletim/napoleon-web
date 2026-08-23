import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { PolicyOnnxCompatibilityError } from "./errors.js";
import { loadBiddingMarginOnnxModel, loadPolicyCriticOnnxModel, loadPolicyOnnxModel } from "./policyOnnx.js";
import type { BiddingMarginOnnxModel, PolicyCriticOnnxModel, PolicyOnnxModel } from "./policyOnnx.js";
import {
  validateBiddingMarginOnnxMetadata,
  validatePolicyCriticOnnxMetadata,
  validatePolicyOnnxMetadata
} from "./metadata.js";
import type { BiddingMarginOnnxMetadata, PolicyOnnxInferenceDevice, PolicyOnnxMetadata } from "./types.js";

export const RL_V740_BENCHMARK_POLICY_ID = "rl-v740" as const;
export const PPO_SEPARATED_V1000_BENCHMARK_POLICY_ID = "ppo-separated-v1000" as const;
export const ISSUE427_T1_BIDDING_MARGIN_POLICY_ID = "issue427-t1-strong-raise-ft" as const;
export const FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID = "frozen-raise-v1" as const;

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

export interface BiddingMarginPolicyArtifactReference {
  id: string;
  displayName: string;
  onnxPath: string;
  metadataPath: string;
  externalDataPath?: string;
  exportReportPath?: string;
  manifestPath?: string;
  onnxSha256: string;
  metadataSha256: string;
  externalDataSha256?: string;
  exportReportSha256?: string;
  manifestSha256?: string;
}

export interface LoadedBiddingMarginPolicyBenchmark {
  artifact: BiddingMarginPolicyArtifactReference;
  model: BiddingMarginOnnxModel;
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

const issue427T1BiddingMarginArtifact = {
  id: ISSUE427_T1_BIDDING_MARGIN_POLICY_ID,
  displayName: "Issue #427 T1 history-consistent raise FT",
  onnxPath: biddingMarginBenchmarkPath("issue427-t1-strong-raise-ft/margin.onnx"),
  metadataPath: biddingMarginBenchmarkPath("issue427-t1-strong-raise-ft/margin.json"),
  externalDataPath: biddingMarginBenchmarkPath("issue427-t1-strong-raise-ft/margin.onnx.data"),
  exportReportPath: biddingMarginBenchmarkPath("issue427-t1-strong-raise-ft/export-report.json"),
  onnxSha256: "c6e51c0f45d62436964a328c9aa2989eaf8cd2822d8aa4bd3648604bca75d14c",
  metadataSha256: "6f7fde46269b77aac7165adc5f5a0bd7276e725c5ca47e5c52cff0aba9a3b519",
  externalDataSha256: "aee857e51e8dcdaebe6449bfc1984fde5c661f51395512c0c62dde23d261bed0",
  exportReportSha256: "00906b86fb8f48240826574c11da15ebb41d57e6e3862dc415b93f948bbdc905"
} as const satisfies BiddingMarginPolicyArtifactReference;

const frozenRaiseV1BiddingMarginArtifact = {
  id: FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID,
  displayName: "Frozen bidding margin v1 (Issue #431 50k)",
  onnxPath: biddingMarginBenchmarkPath("frozen-raise-v1/margin.onnx"),
  metadataPath: biddingMarginBenchmarkPath("frozen-raise-v1/margin.json"),
  externalDataPath: biddingMarginBenchmarkPath("frozen-raise-v1/margin.onnx.data"),
  exportReportPath: biddingMarginBenchmarkPath("frozen-raise-v1/export-report.json"),
  manifestPath: biddingMarginBenchmarkPath("frozen-raise-v1/frozen-manifest.json"),
  onnxSha256: "f3454cca5d2ef667942431296b1260da114f255ce9a03594d32720b180c9c623",
  metadataSha256: "6b3eb4f1d5d5f6f1d6986a7e960f0770292ae8b74dc184d8c237e476e17ce479",
  externalDataSha256: "37f5c882cbdf3b0a4deb4635a81d0eaa76d0987f305663451e8488ed92401872",
  exportReportSha256: "8c871120927f3e99ef4e37926733f73184cd4bce12a0bdda117a3e8962ce33f3",
  manifestSha256: "bc79cbd7895c6a2228d98bae55983fe1a96cd8ae5b655866e8fe373c51098c56"
} as const satisfies BiddingMarginPolicyArtifactReference;

export type RepoManagedBiddingMarginPolicyBenchmarkId =
  | typeof ISSUE427_T1_BIDDING_MARGIN_POLICY_ID
  | typeof FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID;

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

export function getRepoManagedBiddingMarginPolicyBenchmark(
  id: RepoManagedBiddingMarginPolicyBenchmarkId
): BiddingMarginPolicyArtifactReference {
  switch (id) {
    case ISSUE427_T1_BIDDING_MARGIN_POLICY_ID:
      return { ...issue427T1BiddingMarginArtifact };
    case FROZEN_RAISE_V1_BIDDING_MARGIN_POLICY_ID:
      return { ...frozenRaiseV1BiddingMarginArtifact };
  }
}

export async function loadRepoManagedBiddingMarginPolicyBenchmark(
  id: RepoManagedBiddingMarginPolicyBenchmarkId,
  options: { inferenceDevice?: PolicyOnnxInferenceDevice } = {}
): Promise<LoadedBiddingMarginPolicyBenchmark> {
  const artifact = getRepoManagedBiddingMarginPolicyBenchmark(id);
  await validateBiddingMarginPolicyArtifactReference(artifact);
  return {
    artifact,
    model: await loadBiddingMarginOnnxModel({
      onnxPath: artifact.onnxPath,
      metadataPath: artifact.metadataPath,
      inferenceDevice: options.inferenceDevice
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

export async function validateBiddingMarginPolicyArtifactReference(
  artifact: BiddingMarginPolicyArtifactReference
): Promise<BiddingMarginOnnxMetadata> {
  const [onnxSha256, metadataBytes, externalDataSha256, exportReportSha256, manifestSha256] = await Promise.all([
    calculateFileSha256(artifact.onnxPath),
    readFile(artifact.metadataPath),
    artifact.externalDataPath === undefined ? Promise.resolve(undefined) : calculateFileSha256(artifact.externalDataPath),
    artifact.exportReportPath === undefined ? Promise.resolve(undefined) : calculateFileSha256(artifact.exportReportPath),
    artifact.manifestPath === undefined ? Promise.resolve(undefined) : calculateFileSha256(artifact.manifestPath)
  ]);
  const metadataSha256 = sha256(metadataBytes);
  if (onnxSha256 !== artifact.onnxSha256) {
    throw new PolicyOnnxCompatibilityError(
      `bidding margin artifact ${artifact.id} ONNX SHA256 mismatch: ` +
      `expected ${artifact.onnxSha256}, got ${onnxSha256}.`
    );
  }
  if (metadataSha256 !== artifact.metadataSha256) {
    throw new PolicyOnnxCompatibilityError(
      `bidding margin artifact ${artifact.id} metadata SHA256 mismatch: ` +
      `expected ${artifact.metadataSha256}, got ${metadataSha256}.`
    );
  }
  if (
    artifact.externalDataSha256 !== undefined &&
    externalDataSha256 !== artifact.externalDataSha256
  ) {
    throw new PolicyOnnxCompatibilityError(
      `bidding margin artifact ${artifact.id} external data SHA256 mismatch: ` +
      `expected ${artifact.externalDataSha256}, got ${externalDataSha256}.`
    );
  }
  if (
    artifact.exportReportSha256 !== undefined &&
    exportReportSha256 !== artifact.exportReportSha256
  ) {
    throw new PolicyOnnxCompatibilityError(
      `bidding margin artifact ${artifact.id} export report SHA256 mismatch: ` +
      `expected ${artifact.exportReportSha256}, got ${exportReportSha256}.`
    );
  }
  if (
    artifact.manifestSha256 !== undefined &&
    manifestSha256 !== artifact.manifestSha256
  ) {
    throw new PolicyOnnxCompatibilityError(
      `bidding margin artifact ${artifact.id} manifest SHA256 mismatch: ` +
      `expected ${artifact.manifestSha256}, got ${manifestSha256}.`
    );
  }

  const metadata = JSON.parse(new TextDecoder().decode(metadataBytes)) as unknown;
  validateBiddingMarginOnnxMetadata(metadata);
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

function biddingMarginBenchmarkPath(relativePath: string): string {
  return fileURLToPath(new URL(`../../../benchmarks/bidding-margin-policies/${relativePath}`, import.meta.url));
}
