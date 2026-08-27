import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  PARAMETERIZED_ADJUTANT_FEATURE_COUNT,
  PARAMETERIZED_EXCHANGE_FEATURE_COUNT,
  PARAMETERIZED_NON_PLAYING_FEATURE_SCHEMA_VERSION,
  PARAMETERIZED_NON_PLAYING_PARAMETER_COUNT
} from "@napoleon/ai";
import type { ParameterizedNonPlayingParameters } from "@napoleon/ai";

export const PARAMETERIZED_ADJUTANT_EXCHANGE_V1_POLICY_ID =
  "parameterized-adjutant-exchange-v1" as const;
export const PARAMETERIZED_ADJUTANT_EXCHANGE_V1_POLICY_PATH = fileURLToPath(
  new URL(
    "../../../benchmarks/non-playing-policies/parameterized-adjutant-exchange-v1/policy.json",
    import.meta.url
  )
);
export const PARAMETERIZED_ADJUTANT_EXCHANGE_V1_SCHEMA_PATH = fileURLToPath(
  new URL(
    "../../../benchmarks/non-playing-policies/parameterized-adjutant-exchange-v1/feature-schema.json",
    import.meta.url
  )
);
const REPOSITORY_ROOT = fileURLToPath(new URL("../../../", import.meta.url));

const expectedPolicyFileSha256 =
  "71b417b9011907ee02e5b3f7521e5479ca260101937dab05ee5cb91a8634c9fa";
const expectedSchemaFileSha256 =
  "fe217987adac8d4671e849da795f0028516b709291f340a30ad45dd9d7aee098";
const expectedLogicalArtifactSha256 =
  "a6e97b72160338d3f0ce831f5b1422f86dafb419ff8e458b79741440b2433faa";
const expectedParameterSha256 =
  "d364aef0c48a1832bd6602d254d0440f6cb2e2cb50492cfb53934e0378a84d69";
const expectedWeightVectorSha256 =
  "c84fbf7012ee043b2f3451213ecf30cdf596eaec41c6aeff7381ff7fccc9a4f0";

export interface ParameterizedPolicyArtifactProvenance {
  policyId: typeof PARAMETERIZED_ADJUTANT_EXCHANGE_V1_POLICY_ID;
  policyPath: string;
  policyFileSha256: string;
  schemaPath: string;
  schemaFileSha256: string;
  logicalArtifactSha256: string;
  parameterSha256: string;
  weightVectorSha256: string;
  optimizerIssue: "452";
  verificationIssue: "454";
  biddingDependencySha256: string;
  playingDependencySha256: string;
  playingCriticDependencySha256: string;
  evaluatorDependencySha256: string;
  sourceEvaluatorDependencySha256: string;
  verificationReportPath: string;
  verificationReportFileSha256: string;
  verificationSeedManifestPath: string;
  verificationSeedManifestFileSha256: string;
  verificationSeedManifestSha256: string;
}

export interface LoadedParameterizedPolicyArtifact {
  parameters: ParameterizedNonPlayingParameters;
  provenance: ParameterizedPolicyArtifactProvenance;
}

interface WeightRow {
  index: number;
  block: "adjutant" | "exchange";
  name: string;
  scale: number;
  description: string;
  weight: number;
}

interface FeatureRow {
  index: number;
  block: "adjutant" | "exchange";
  name: string;
  scale: number;
  description: string;
}

export function loadParameterizedPolicyArtifact(
  policyPath = PARAMETERIZED_ADJUTANT_EXCHANGE_V1_POLICY_PATH,
  schemaPath = PARAMETERIZED_ADJUTANT_EXCHANGE_V1_SCHEMA_PATH,
  options: { validateRepoManagedFileHashes?: boolean } = {}
): LoadedParameterizedPolicyArtifact {
  let policyBytes: Buffer;
  let schemaBytes: Buffer;
  try {
    policyBytes = readFileSync(policyPath);
  } catch (error) {
    throw new Error(`Parameterized non-playing artifact is missing or unreadable: ${policyPath}: ${message(error)}`);
  }
  try {
    schemaBytes = readFileSync(schemaPath);
  } catch (error) {
    throw new Error(`Parameterized non-playing feature schema is missing or unreadable: ${schemaPath}: ${message(error)}`);
  }

  const policy = parseJsonObject(policyBytes, "artifact", policyPath);
  const schema = parseJsonObject(schemaBytes, "feature schema", schemaPath);
  const featureRows = validateFeatureSchema(schema);
  const weightRows = validatePolicy(policy, featureRows);
  const weights = weightRows.map((row) => row.weight);
  const weightVectorSha256 = sha256(JSON.stringify(weights));
  if (weightVectorSha256 !== expectedWeightVectorSha256) {
    throw new Error(
      `Parameterized non-playing weight vector SHA256 mismatch: expected ${expectedWeightVectorSha256}, got ${weightVectorSha256}.`
    );
  }

  const validateFileHashes = options.validateRepoManagedFileHashes ??
    (policyPath === PARAMETERIZED_ADJUTANT_EXCHANGE_V1_POLICY_PATH &&
      schemaPath === PARAMETERIZED_ADJUTANT_EXCHANGE_V1_SCHEMA_PATH);
  const policyFileSha256 = sha256(policyBytes);
  const schemaFileSha256 = sha256(schemaBytes);
  if (validateFileHashes && policyFileSha256 !== expectedPolicyFileSha256) {
    throw new Error(
      `Parameterized non-playing artifact file SHA256 mismatch: expected ${expectedPolicyFileSha256}, got ${policyFileSha256}.`
    );
  }
  if (validateFileHashes && schemaFileSha256 !== expectedSchemaFileSha256) {
    throw new Error(
      `Parameterized non-playing feature schema file SHA256 mismatch: expected ${expectedSchemaFileSha256}, got ${schemaFileSha256}.`
    );
  }

  const dependencies = requireObject(policy.dependencyProvenance, "dependencyProvenance");
  expectString(
    dependencies.biddingMargin,
    "dependencyProvenance.biddingMargin",
    "f3454cca5d2ef667942431296b1260da114f255ce9a03594d32720b180c9c623"
  );
  expectString(
    dependencies.playingPolicy,
    "dependencyProvenance.playingPolicy",
    "54d7ba29222a12e99a91ab61ee7aa253fe3fab73200d78167d64bf9e7bb8887e"
  );
  expectString(
    dependencies.playingCritic,
    "dependencyProvenance.playingCritic",
    "3055882f3e63e2a096ee7cedee341bc97e033572bcb59f36f3f68e3d89f134d9"
  );
  const optimizer = requireObject(policy.optimizerProvenance, "optimizerProvenance");
  const verification = requireObject(policy.verificationProvenance, "verificationProvenance");
  expectNumber(optimizer.sourceIssue, "optimizerProvenance.sourceIssue", 452);
  expectString(
    requireObject(optimizer.parameterArtifact, "optimizerProvenance.parameterArtifact").optimizer,
    "optimizerProvenance.parameterArtifact.optimizer",
    "pycma.CMAEvolutionStrategy"
  );
  expectNumber(verification.sourceIssue, "verificationProvenance.sourceIssue", 454);
  expectNumber(verification.games, "verificationProvenance.games", 10_000);
  const verificationReportPath = readString(
    verification.report,
    "verificationProvenance.report"
  );
  const verificationReportFileSha256 = readSha256(
    verification.reportFileSha256,
    "verificationProvenance.reportFileSha256"
  );
  const verificationSeedManifestPath = readString(
    verification.seedManifest,
    "verificationProvenance.seedManifest"
  );
  const verificationSeedManifestFileSha256 = readSha256(
    verification.seedManifestFileSha256,
    "verificationProvenance.seedManifestFileSha256"
  );
  const verificationSeedManifestSha256 = readSha256(
    verification.seedManifestSha256,
    "verificationProvenance.seedManifestSha256"
  );
  const evaluatorDependencySha256 = readSha256(
    dependencies.evaluator,
    "dependencyProvenance.evaluator"
  );
  const sourceEvaluatorDependencySha256 = readSha256(
    dependencies.sourceIssue452Evaluator,
    "dependencyProvenance.sourceIssue452Evaluator"
  );
  expectString(
    dependencies.verificationEvaluator,
    "dependencyProvenance.verificationEvaluator",
    evaluatorDependencySha256
  );
  if (validateFileHashes) {
    validateReferencedFileHash(verificationReportPath, verificationReportFileSha256);
    validateReferencedFileHash(
      verificationSeedManifestPath,
      verificationSeedManifestFileSha256
    );
  }

  return {
    parameters: {
      featureSchemaVersion: 1,
      adjutantWeights: weights.slice(0, PARAMETERIZED_ADJUTANT_FEATURE_COUNT),
      exchangeWeights: weights.slice(PARAMETERIZED_ADJUTANT_FEATURE_COUNT)
    },
    provenance: {
      policyId: PARAMETERIZED_ADJUTANT_EXCHANGE_V1_POLICY_ID,
      policyPath,
      policyFileSha256,
      schemaPath,
      schemaFileSha256,
      logicalArtifactSha256: expectedLogicalArtifactSha256,
      parameterSha256: expectedParameterSha256,
      weightVectorSha256,
      optimizerIssue: "452",
      verificationIssue: "454",
      biddingDependencySha256: dependencies.biddingMargin as string,
      playingDependencySha256: dependencies.playingPolicy as string,
      playingCriticDependencySha256: dependencies.playingCritic as string,
      evaluatorDependencySha256,
      sourceEvaluatorDependencySha256,
      verificationReportPath,
      verificationReportFileSha256,
      verificationSeedManifestPath,
      verificationSeedManifestFileSha256,
      verificationSeedManifestSha256
    }
  };
}

function validateFeatureSchema(schema: Record<string, unknown>): readonly FeatureRow[] {
  expectNumber(
    schema.schemaVersion,
    "feature schema schemaVersion",
    PARAMETERIZED_NON_PLAYING_FEATURE_SCHEMA_VERSION
  );
  expectNumber(
    schema.adjutantFeatureCount,
    "feature schema adjutantFeatureCount",
    PARAMETERIZED_ADJUTANT_FEATURE_COUNT
  );
  expectNumber(
    schema.exchangeFeatureCount,
    "feature schema exchangeFeatureCount",
    PARAMETERIZED_EXCHANGE_FEATURE_COUNT
  );
  expectNumber(
    schema.parameterCount,
    "feature schema parameterCount",
    PARAMETERIZED_NON_PLAYING_PARAMETER_COUNT
  );
  if (!Array.isArray(schema.features) || schema.features.length !== PARAMETERIZED_NON_PLAYING_PARAMETER_COUNT) {
    throw new Error(
      `Parameterized feature schema must contain ${PARAMETERIZED_NON_PLAYING_PARAMETER_COUNT} features.`
    );
  }
  return schema.features.map((value, index) => validateFeatureRow(value, index));
}

function validatePolicy(
  policy: Record<string, unknown>,
  features: readonly FeatureRow[]
): readonly WeightRow[] {
  expectString(
    policy.artifactType,
    "artifactType",
    "parameterized-adjutant-exchange-policy-candidate"
  );
  expectNumber(policy.artifactVersion, "artifactVersion", 1);
  expectNumber(
    policy.featureSchemaVersion,
    "featureSchemaVersion",
    PARAMETERIZED_NON_PLAYING_FEATURE_SCHEMA_VERSION
  );
  expectNumber(policy.parameterCount, "parameterCount", PARAMETERIZED_NON_PLAYING_PARAMETER_COUNT);
  expectString(policy.sha256, "sha256", expectedLogicalArtifactSha256);
  expectString(policy.parameterSha256, "parameterSha256", expectedParameterSha256);
  expectString(policy.weightVectorSha256, "weightVectorSha256", expectedWeightVectorSha256);
  if (!Array.isArray(policy.weights) || policy.weights.length !== PARAMETERIZED_NON_PLAYING_PARAMETER_COUNT) {
    throw new Error(
      `Parameterized artifact must contain ${PARAMETERIZED_NON_PLAYING_PARAMETER_COUNT} weights.`
    );
  }
  const rows = policy.weights.map((value, index) => validateWeightRow(value, index));
  rows.forEach((row, index) => {
    const feature = features[index];
    for (const key of ["index", "block", "name", "scale", "description"] as const) {
      if (row[key] !== feature[key]) {
        throw new Error(`Parameterized artifact feature parity mismatch at weights[${index}].${key}.`);
      }
    }
  });
  const adjutantCount = rows.filter((row) => row.block === "adjutant").length;
  const exchangeCount = rows.filter((row) => row.block === "exchange").length;
  if (
    adjutantCount !== PARAMETERIZED_ADJUTANT_FEATURE_COUNT ||
    exchangeCount !== PARAMETERIZED_EXCHANGE_FEATURE_COUNT
  ) {
    throw new Error(
      `Parameterized artifact block counts mismatch: adjutant=${adjutantCount}, exchange=${exchangeCount}.`
    );
  }
  return rows;
}

function validateFeatureRow(value: unknown, expectedIndex: number): FeatureRow {
  const row = requireObject(value, `features[${expectedIndex}]`);
  const block = readBlock(row.block, `features[${expectedIndex}].block`);
  expectNumber(row.index, `features[${expectedIndex}].index`, expectedIndex);
  return {
    index: expectedIndex,
    block,
    name: readString(row.name, `features[${expectedIndex}].name`),
    scale: readFiniteNumber(row.scale, `features[${expectedIndex}].scale`),
    description: readString(row.description, `features[${expectedIndex}].description`)
  };
}

function validateWeightRow(value: unknown, expectedIndex: number): WeightRow {
  const row = requireObject(value, `weights[${expectedIndex}]`);
  const feature = validateFeatureRow(row, expectedIndex);
  return {
    ...feature,
    weight: readFiniteNumber(row.weight, `weights[${expectedIndex}].weight`)
  };
}

function parseJsonObject(bytes: Buffer, label: string, path: string): Record<string, unknown> {
  try {
    return requireObject(JSON.parse(bytes.toString("utf8")), label);
  } catch (error) {
    throw new Error(`Parameterized non-playing ${label} is malformed (${path}): ${message(error)}`);
  }
}

function requireObject(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function readBlock(value: unknown, label: string): "adjutant" | "exchange" {
  if (value !== "adjutant" && value !== "exchange") {
    throw new Error(`${label} must be adjutant or exchange.`);
  }
  return value;
}

function readString(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${label} must be a non-empty string.`);
  }
  return value;
}

function readFiniteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number.`);
  }
  return value;
}

function readSha256(value: unknown, label: string): string {
  const result = readString(value, label);
  if (!/^[0-9a-f]{64}$/.test(result)) {
    throw new Error(`${label} must be a lowercase SHA256 digest.`);
  }
  return result;
}

function validateReferencedFileHash(repositoryRelativePath: string, expected: string): void {
  if (repositoryRelativePath.startsWith("/") || repositoryRelativePath.includes("..")) {
    throw new Error(`Artifact provenance path must be repository-relative: ${repositoryRelativePath}.`);
  }
  const path = `${REPOSITORY_ROOT}${repositoryRelativePath}`;
  let bytes: Buffer;
  try {
    bytes = readFileSync(path);
  } catch (error) {
    throw new Error(`Artifact provenance file is missing or unreadable: ${path}: ${message(error)}`);
  }
  const actual = sha256(bytes);
  if (actual !== expected) {
    throw new Error(
      `Artifact provenance file SHA256 mismatch for ${repositoryRelativePath}: expected ${expected}, got ${actual}.`
    );
  }
}

function expectString(value: unknown, label: string, expected: string): void {
  if (value !== expected) {
    throw new Error(`${label} mismatch: expected ${expected}, got ${String(value)}.`);
  }
}

function expectNumber(value: unknown, label: string, expected: number): void {
  if (value !== expected) {
    throw new Error(`${label} mismatch: expected ${expected}, got ${String(value)}.`);
  }
}

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
