import { describe, expect, it } from "vitest";
import { parseAllowedHosts } from "./viteAllowedHosts";

describe("parseAllowedHosts", () => {
  it("returns an empty list for undefined", () => {
    expect(parseAllowedHosts(undefined)).toEqual([]);
  });

  it("returns an empty list for an empty string", () => {
    expect(parseAllowedHosts("")).toEqual([]);
  });

  it("parses a single host", () => {
    expect(parseAllowedHosts("host-a.example.test")).toEqual(["host-a.example.test"]);
  });

  it("parses multiple hosts", () => {
    expect(parseAllowedHosts("host-a.example.test,host-b.example.test")).toEqual([
      "host-a.example.test",
      "host-b.example.test"
    ]);
  });

  it("trims surrounding whitespace", () => {
    expect(parseAllowedHosts(" host-a.example.test, host-b.example.test ")).toEqual([
      "host-a.example.test",
      "host-b.example.test"
    ]);
  });

  it("removes empty entries", () => {
    expect(parseAllowedHosts("host-a.example.test,,host-c.example.test,")).toEqual([
      "host-a.example.test",
      "host-c.example.test"
    ]);
  });

  it("preserves input order", () => {
    expect(parseAllowedHosts("host-a.example.test, host-b.example.test,,host-c.example.test ")).toEqual([
      "host-a.example.test",
      "host-b.example.test",
      "host-c.example.test"
    ]);
  });
});
