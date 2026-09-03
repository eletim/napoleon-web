import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  PARAMETERIZED_ADJUTANT_EXCHANGE_V1_POLICY_PATH,
  PARAMETERIZED_ADJUTANT_EXCHANGE_V1_SCHEMA_PATH,
  loadParameterizedPolicyArtifact
} from "../src/parameterizedPolicyArtifact.js";

describe("parameterized non-playing artifact", () => {
  // Normal loading (the default, and what server startup uses) never reads
  // the historical Issue #454 verification report / seed manifest - those
  // large audit files were deliberately removed from the repo (see the
  // artifact's own dependencyProvenance/verificationProvenance metadata,
  // which is still validated below) and are not needed to actually run the
  // policy. This must keep working even in a fresh checkout that never had
  // them.
  it("loads the repo-managed human-readable source of truth with fixed provenance", () => {
    const loaded = loadParameterizedPolicyArtifact();

    expect(loaded.parameters.featureSchemaVersion).toBe(1);
    expect(loaded.parameters.adjutantWeights).toHaveLength(35);
    expect(loaded.parameters.exchangeWeights).toHaveLength(60);
    expect(loaded.provenance).toMatchObject({
      logicalArtifactSha256: "a6e97b72160338d3f0ce831f5b1422f86dafb419ff8e458b79741440b2433faa",
      parameterSha256: "d364aef0c48a1832bd6602d254d0440f6cb2e2cb50492cfb53934e0378a84d69",
      weightVectorSha256: "c84fbf7012ee043b2f3451213ecf30cdf596eaec41c6aeff7381ff7fccc9a4f0",
      optimizerIssue: "452",
      verificationIssue: "454",
      biddingDependencySha256: "f3454cca5d2ef667942431296b1260da114f255ce9a03594d32720b180c9c623",
      playingDependencySha256: "54d7ba29222a12e99a91ab61ee7aa253fe3fab73200d78167d64bf9e7bb8887e",
      playingCriticDependencySha256: "3055882f3e63e2a096ee7cedee341bc97e033572bcb59f36f3f68e3d89f134d9"
    });
  });

  // The dedicated entry point for re-verifying the historical Issue #454
  // audit trail (never used by normal server startup - see above). In this
  // repository the audit files were deliberately removed as oversized
  // research artifacts, so opting in must fail loudly rather than silently
  // skip the check.
  it("detects a missing historical provenance audit file when explicitly requested", () => {
    let thrown: unknown;
    try {
      loadParameterizedPolicyArtifact(undefined, undefined, {
        validateRepoManagedFileHashes: true
      });
    } catch (error) {
      thrown = error;
    }

    expect(thrown).toBeInstanceOf(Error);
    const thrownMessage = (thrown as Error).message;
    expect(thrownMessage).toContain("Artifact provenance file is missing or unreadable");
    expect(thrownMessage).toContain("verification-report.json");
  });

  it("rejects missing and malformed artifacts without fallback", () => {
    expect(() =>
      loadParameterizedPolicyArtifact("/definitely/missing/policy.json", PARAMETERIZED_ADJUTANT_EXCHANGE_V1_SCHEMA_PATH)
    ).toThrow("artifact is missing or unreadable");

    const fixture = createFixture();
    writeFileSync(fixture.policyPath, "{not-json", "utf8");
    expect(() => loadParameterizedPolicyArtifact(fixture.policyPath, fixture.schemaPath)).toThrow(
      "artifact is malformed"
    );
  });

  it("rejects feature schema and weight mutations", () => {
    const schemaFixture = createFixture();
    const schema = JSON.parse(readFileSync(schemaFixture.schemaPath, "utf8")) as {
      schemaVersion: number;
    };
    schema.schemaVersion = 2;
    writeFileSync(schemaFixture.schemaPath, JSON.stringify(schema), "utf8");
    expect(() =>
      loadParameterizedPolicyArtifact(schemaFixture.policyPath, schemaFixture.schemaPath)
    ).toThrow("feature schema schemaVersion mismatch");

    const weightFixture = createFixture();
    const policy = JSON.parse(readFileSync(weightFixture.policyPath, "utf8")) as {
      weights: { weight: number }[];
    };
    policy.weights[0].weight += 1;
    writeFileSync(weightFixture.policyPath, JSON.stringify(policy), "utf8");
    expect(() =>
      loadParameterizedPolicyArtifact(weightFixture.policyPath, weightFixture.schemaPath)
    ).toThrow("weight vector SHA256 mismatch");
  });
});

function createFixture(): { policyPath: string; schemaPath: string } {
  const directory = mkdtempSync(join(tmpdir(), "napoleon-parameterized-artifact-"));
  const policyPath = join(directory, "policy.json");
  const schemaPath = join(directory, "feature-schema.json");
  writeFileSync(policyPath, readFileSync(PARAMETERIZED_ADJUTANT_EXCHANGE_V1_POLICY_PATH));
  writeFileSync(schemaPath, readFileSync(PARAMETERIZED_ADJUTANT_EXCHANGE_V1_SCHEMA_PATH));
  return { policyPath, schemaPath };
}
