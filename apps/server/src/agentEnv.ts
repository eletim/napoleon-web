export const learnedPolicySlotNumbers = [1, 2, 3, 4, 5] as const;
export const learnedPolicyEnvFields = [
  "DISPLAY_NAME",
  "ONNX_PATH",
  "METADATA_PATH"
] as const;

export type LearnedPolicySlotNumber = (typeof learnedPolicySlotNumbers)[number];
export type LearnedPolicyEnvField = (typeof learnedPolicyEnvFields)[number];

export function createLearnedPolicyEnvKey(
  slotNumber: LearnedPolicySlotNumber,
  field: LearnedPolicyEnvField
): string {
  return `NAPOLEON_POLICY_${slotNumber}_${field}`;
}

export const learnedPolicyEnvKeys = learnedPolicySlotNumbers.flatMap((slotNumber) =>
  learnedPolicyEnvFields.map((field) => createLearnedPolicyEnvKey(slotNumber, field))
);
