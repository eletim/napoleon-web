export const learnedPolicySlotNumbers = [1, 2, 3, 4, 5] as const;
export const learnedPolicyEnvFields = [
  "DISPLAY_NAME",
  "ONNX_PATH",
  "METADATA_PATH"
] as const;
export const fullPolicyEnvFields = [
  "DISPLAY_NAME",
  "PLAYING_ONNX_PATH",
  "PLAYING_METADATA_PATH",
  "BIDDING_ONNX_PATH",
  "BIDDING_METADATA_PATH",
  "ADJUTANT_ONNX_PATH",
  "ADJUTANT_METADATA_PATH",
  "EXCHANGE_ONNX_PATH",
  "EXCHANGE_METADATA_PATH"
] as const;

export type LearnedPolicySlotNumber = (typeof learnedPolicySlotNumbers)[number];
export type LearnedPolicyEnvField = (typeof learnedPolicyEnvFields)[number];
export type FullPolicyEnvField = (typeof fullPolicyEnvFields)[number];

export function createLearnedPolicyEnvKey(
  slotNumber: LearnedPolicySlotNumber,
  field: LearnedPolicyEnvField
): string {
  return `NAPOLEON_POLICY_${slotNumber}_${field}`;
}

export function createFullPolicyEnvKey(
  slotNumber: LearnedPolicySlotNumber,
  field: FullPolicyEnvField
): string {
  return `NAPOLEON_FULL_POLICY_${slotNumber}_${field}`;
}

export const learnedPolicyEnvKeys = learnedPolicySlotNumbers.flatMap((slotNumber) =>
  learnedPolicyEnvFields.map((field) => createLearnedPolicyEnvKey(slotNumber, field))
);

export const fullPolicyEnvKeys = learnedPolicySlotNumbers.flatMap((slotNumber) =>
  fullPolicyEnvFields.map((field) => createFullPolicyEnvKey(slotNumber, field))
);
