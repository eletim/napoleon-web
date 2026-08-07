import { createHash } from "node:crypto";
import { CARD_IDS } from "@napoleon/ai-observation";

export function calculateCardIdsSha256(): string {
  return createHash("sha256").update(JSON.stringify(CARD_IDS), "utf8").digest("hex");
}
