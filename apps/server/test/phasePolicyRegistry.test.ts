import { describe, expect, it } from "vitest";
import { AgentUnavailableError } from "../src/agentErrors.js";
import { createPhasePolicyRegistry } from "../src/phasePolicyRegistry.js";

describe("formal phase policy availability", () => {
  it("keeps frozen bidding available when the unused playing actor fails preflight", async () => {
    const registry = createPhasePolicyRegistry({
      loadPlayingPolicy: async () => {
        throw new Error("actor artifact missing");
      }
    });

    await registry.initialize();

    expect(findAvailability(registry, "playing", "ppo-separated-v1000")).toBe(false);
    expect(findAvailability(registry, "bidding", "frozen-raise-v1")).toBe(true);
    expect(() => registry.createAgent({
      playing: "rule-based",
      bidding: "frozen-raise-v1",
      nonPlaying: "rule-based"
    })).not.toThrow();
    expect(() => registry.createAgent({
      playing: "ppo-separated-v1000",
      bidding: "rule-based",
      nonPlaying: "rule-based"
    })).toThrow(AgentUnavailableError);
  });

  it("keeps formal playing available when the independently loaded critic fails", async () => {
    const registry = createPhasePolicyRegistry({
      loadPlayingCritic: async () => {
        throw new Error("critic artifact missing");
      }
    });

    await registry.initialize();

    expect(findAvailability(registry, "playing", "ppo-separated-v1000")).toBe(true);
    expect(findAvailability(registry, "bidding", "frozen-raise-v1")).toBe(false);
    expect(() => registry.createAgent({
      playing: "ppo-separated-v1000",
      bidding: "rule-based",
      nonPlaying: "rule-based"
    })).not.toThrow();
    expect(() => registry.createAgent({
      playing: "rule-based",
      bidding: "frozen-raise-v1",
      nonPlaying: "rule-based"
    })).toThrow(AgentUnavailableError);
  });

  // Mirrors the two tests above: a single non-playing policy's own load
  // failure (e.g. the historical Issue #454 provenance audit files this
  // artifact used to require, now absent from a fresh checkout) must not
  // take rule-based, playing, or bidding down with it.
  it("keeps rule-based and the other phases available when the parameterized artifact fails to load", async () => {
    const registry = createPhasePolicyRegistry({
      loadParameterizedArtifact: () => {
        throw new Error("verification report artifact missing");
      }
    });

    await registry.initialize();

    expect(findAvailability(registry, "nonPlaying", "parameterized-adjutant-exchange-v1")).toBe(false);
    expect(findAvailability(registry, "playing", "ppo-separated-v1000")).toBe(true);
    expect(findAvailability(registry, "bidding", "frozen-raise-v1")).toBe(true);
    expect(() => registry.createAgent({
      playing: "rule-based",
      bidding: "rule-based",
      nonPlaying: "rule-based"
    })).not.toThrow();
    expect(() => registry.createAgent({
      playing: "ppo-separated-v1000",
      bidding: "frozen-raise-v1",
      nonPlaying: "rule-based"
    })).not.toThrow();
    expect(() => registry.createAgent({
      playing: "rule-based",
      bidding: "rule-based",
      nonPlaying: "parameterized-adjutant-exchange-v1"
    })).toThrow(AgentUnavailableError);
  });
});

function findAvailability(
  registry: ReturnType<typeof createPhasePolicyRegistry>,
  phase: "playing" | "bidding" | "nonPlaying",
  policyId: string
): boolean | undefined {
  return registry.describe()[phase].find((policy) => policy.id === policyId)?.isAvailable;
}
