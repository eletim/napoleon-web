import { describe, expect, it } from "vitest";
import { CARD_IDS } from "@napoleon/ai-observation";
import { calculateCardIdsSha256, serializeManifest, sha256Utf8 } from "../src/index.js";

describe("serialization", () => {
  it("hashes card ids as JSON.stringify(CARD_IDS) UTF-8", () => {
    expect(calculateCardIdsSha256()).toBe(sha256Utf8(JSON.stringify(CARD_IDS)));
  });

  it("serializes manifest with two-space JSON and one trailing newline", () => {
    expect(serializeManifest({ b: 1, a: 2 })).toBe("{\n  \"b\": 1,\n  \"a\": 2\n}\n");
  });
});
