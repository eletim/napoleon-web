import { beforeEach, describe, expect, it, vi } from "vitest";

const generateRuleBasedDataset = vi.hoisted(() => vi.fn());

vi.mock("@napoleon/training-data", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@napoleon/training-data")>()),
  generateRuleBasedDataset
}));

const { runCli } = await import("../src/runCli.js");

function createIo() {
  return {
    stdout: { write: vi.fn() },
    stderr: { write: vi.fn() }
  };
}

describe("runCli", () => {
  beforeEach(() => {
    generateRuleBasedDataset.mockReset();
  });

  it("prints help to stdout and exits 0", async () => {
    const io = createIo();
    const code = await runCli(["--help"], io);

    expect(code).toBe(0);
    expect(io.stdout.write.mock.calls.join("\n")).toContain("Usage:");
    expect(io.stderr.write).not.toHaveBeenCalled();
    expect(generateRuleBasedDataset).not.toHaveBeenCalled();
  });

  it("prints parser errors to stderr and exits 1", async () => {
    const io = createIo();
    const code = await runCli(["--games", "1", "--output", "./out"], io);

    expect(code).toBe(1);
    expect(io.stderr.write.mock.calls.join("\n")).toContain("--start-seed is required");
    expect(io.stdout.write).not.toHaveBeenCalled();
    expect(generateRuleBasedDataset).not.toHaveBeenCalled();
  });

  it("prints generation errors to stderr and exits 1", async () => {
    generateRuleBasedDataset.mockRejectedValueOnce(new Error("generation failed"));
    const io = createIo();
    const code = await runCli([
      "--start-seed",
      "0",
      "--games",
      "1",
      "--output",
      "./out"
    ], io);

    expect(code).toBe(1);
    expect(io.stderr.write.mock.calls.join("\n")).toContain("generation failed");
    expect(io.stdout.write).not.toHaveBeenCalled();
  });
});
