import type { EncodedAdjutantObservation } from "./encodeAdjutantObservation.js";
import { validateEncodedAdjutantObservation } from "./encodeAdjutantObservation.js";
import type { EncodedBiddingHistory } from "./encodeBiddingHistory.js";
import type { EncodedBiddingObservation } from "./encodeBiddingObservation.js";
import { validateEncodedBiddingObservation } from "./encodeBiddingObservation.js";
import type { EncodedCompleteInfoPlayingObservation } from "./encodeCompleteInfoPlayingObservation.js";
import { validateEncodedCompleteInfoPlayingObservation } from "./encodeCompleteInfoPlayingObservation.js";
import type { EncodedExchangeObservation } from "./encodeExchangeObservation.js";
import { validateEncodedExchangeObservation } from "./encodeExchangeObservation.js";
import type { EncodedPlayingObservation } from "./encodePlayingObservation.js";
import { validateEncodedPlayingObservation } from "./encodePlayingObservation.js";
import {
  ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
  ADJUTANT_MODEL_INPUT_SCHEMA_VERSION,
  BIDDING_ACTION_COUNT,
  BIDDING_HISTORY_SUIT_ORDER,
  BIDDING_MODEL_INPUT_FEATURE_COUNT,
  BIDDING_MODEL_INPUT_SCHEMA_VERSION,
  CARD_COUNT,
  CARDS_PER_TRICK,
  COMPLETE_INFO_PLAYING_MODEL_INPUT_FEATURE_COUNT,
  COMPLETE_INFO_PLAYING_MODEL_INPUT_SCHEMA_VERSION,
  EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
  EXCHANGE_MODEL_INPUT_SCHEMA_VERSION,
  FLAT_OBSERVATION_FEATURE_COUNT,
  MAX_BIDDING_ACTION_COUNT,
  MAX_BIDDING_TARGET_POINT_CARDS,
  MIN_BIDDING_TARGET_POINT_CARDS,
  MIN_CONTRACT_TARGET_POINT_CARDS,
  MODEL_INPUT_FEATURE_COUNT,
  PLAYER_COUNT,
  SELF_ROLE_COUNT,
  TRICK_COUNT
} from "./schema.js";

const COMPLETED_TRICK_CARD_SLOT_COUNT = TRICK_COUNT * CARDS_PER_TRICK;
const SPECIAL_CARD_INDEX_COUNT = 4;
const BIDDING_ACTION_TYPE_CLASS_COUNT = 2;
const COMPLETE_INFO_PLAYING_OWNER_CLASS_COUNT = PLAYER_COUNT + 1;
const BIDDING_TARGET_POINT_CARDS_CLASS_COUNT =
  MAX_BIDDING_TARGET_POINT_CARDS - MIN_BIDDING_TARGET_POINT_CARDS + 1;
const CONTRACT_TARGET_POINT_CARDS_CLASS_COUNT =
  MAX_BIDDING_TARGET_POINT_CARDS - MIN_CONTRACT_TARGET_POINT_CARDS + 1;
const CONSECUTIVE_PASS_COUNT_CLASS_COUNT = PLAYER_COUNT + 1;

interface OneHotIndexField {
  name: string;
  indices: readonly number[];
  slotCount: number;
  classCount: number;
  minValue: number;
  emptyValues: readonly number[];
}

export interface ModelInputFeatureSlice {
  name: string;
  start: number;
  stop: number;
  shape: readonly number[];
  dtype: "float32";
}

export interface PlayingModelInput {
  modelInput: Float32Array;
  legalPlayMask: readonly number[];
}

export interface CompleteInfoPlayingModelInput {
  modelInput: Float32Array;
  legalPlayMask: readonly number[];
}

export interface BiddingModelInput {
  modelInput: Float32Array;
  legalBidMask: readonly number[];
}

export interface ExchangeModelInput {
  modelInput: Float32Array;
  legalDiscardCardMask: readonly number[];
}

export interface AdjutantModelInput {
  modelInput: Float32Array;
  legalAdjutantMask: readonly number[];
}

const FLAT_LAYOUT_SPEC: readonly (readonly [string, readonly number[]])[] = [
  ["trumpSuitOneHot", [4]],
  ["napoleonPlayerOneHot", [PLAYER_COUNT]],
  ["revealedAdjutantPlayerOneHot", [PLAYER_COUNT + 1]],
  ["calledAdjutantCardMask", [CARD_COUNT]],
  ["selfHandMask", [CARD_COUNT]],
  ["legalPlayMask", [CARD_COUNT]],
  ["handCountByPlayer", [PLAYER_COUNT]],
  ["capturedPointCardMaskByPlayer", [PLAYER_COUNT, CARD_COUNT]],
  ["currentTrickSlotMask", [CARDS_PER_TRICK]],
  ["completedTrickSlotMask", [COMPLETED_TRICK_CARD_SLOT_COUNT]],
  ["completedTrickMask", [TRICK_COUNT]],
  ["biddingHistoryActionMask", [MAX_BIDDING_ACTION_COUNT]],
  ["latestBuriedEventPointCardMask", [CARD_COUNT]],
  ["trickNumber", [1]],
  ["completedTrickCount", [1]],
  ["contractTargetPointCards", [1]],
  ["latestBuriedEventHiddenNonPointCount", [1]],
  ["latestBuriedEventPresent", [1]]
];

const MODEL_INPUT_ONEHOT_SPEC: readonly (readonly [string, readonly number[]])[] = [
  ["specialCardIndicesOneHot", [SPECIAL_CARD_INDEX_COUNT, CARD_COUNT]],
  ["currentTrickCardIndicesOneHot", [CARDS_PER_TRICK, CARD_COUNT]],
  ["completedTrickCardIndicesOneHot", [COMPLETED_TRICK_CARD_SLOT_COUNT, CARD_COUNT]],
  ["currentTrickPlayerIndicesOneHot", [CARDS_PER_TRICK, PLAYER_COUNT]],
  ["completedTrickPlayerIndicesOneHot", [COMPLETED_TRICK_CARD_SLOT_COUNT, PLAYER_COUNT]],
  ["completedTrickWinnerIndicesOneHot", [TRICK_COUNT, PLAYER_COUNT]],
  ["biddingHistoryActionTypeIndicesOneHot", [MAX_BIDDING_ACTION_COUNT, BIDDING_ACTION_TYPE_CLASS_COUNT]],
  ["biddingHistoryPlayerIndicesOneHot", [MAX_BIDDING_ACTION_COUNT, PLAYER_COUNT]],
  ["biddingHistorySuitIndicesOneHot", [MAX_BIDDING_ACTION_COUNT, BIDDING_HISTORY_SUIT_ORDER.length]],
  ["biddingHistoryTargetPointCardsOneHot", [MAX_BIDDING_ACTION_COUNT, BIDDING_TARGET_POINT_CARDS_CLASS_COUNT]],
  ["selfRoleOneHot", [SELF_ROLE_COUNT]]
];

const COMPLETE_INFO_PLAYING_MODEL_INPUT_SPEC: readonly (readonly [string, readonly number[]])[] = [
  ["cardOwnerClassByCardOneHot", [CARD_COUNT, COMPLETE_INFO_PLAYING_OWNER_CLASS_COUNT]],
  ["capturedPointCardCountByPlayer", [PLAYER_COUNT]],
  ["trumpSuitOneHot", [BIDDING_HISTORY_SUIT_ORDER.length]],
  ["contractTargetPointCards", [1]],
  ["napoleonPlayerOneHot", [PLAYER_COUNT]],
  ["revealedAdjutantPlayerOneHot", [PLAYER_COUNT + 1]],
  ["selfRoleOneHot", [SELF_ROLE_COUNT]],
  ["calledAdjutantCardIndex", [1]],
  ["specialCardIndices", [SPECIAL_CARD_INDEX_COUNT]],
  ["currentTrickSlotMask", [CARDS_PER_TRICK]],
  ["currentTrickCardIndices", [CARDS_PER_TRICK]],
  ["currentTrickPlayerIndicesOneHot", [CARDS_PER_TRICK, PLAYER_COUNT]],
  ["trickNumber", [1]],
  ["completedTrickCount", [1]]
];

const BIDDING_HISTORY_ONEHOT_SPEC: readonly (readonly [string, readonly number[]])[] = [
  ["biddingHistoryActionTypeIndicesOneHot", [MAX_BIDDING_ACTION_COUNT, BIDDING_ACTION_TYPE_CLASS_COUNT]],
  ["biddingHistoryPlayerIndicesOneHot", [MAX_BIDDING_ACTION_COUNT, PLAYER_COUNT]],
  ["biddingHistorySuitIndicesOneHot", [MAX_BIDDING_ACTION_COUNT, BIDDING_HISTORY_SUIT_ORDER.length]],
  ["biddingHistoryTargetPointCardsOneHot", [MAX_BIDDING_ACTION_COUNT, BIDDING_TARGET_POINT_CARDS_CLASS_COUNT]]
];

const BIDDING_MODEL_INPUT_SPEC: readonly (readonly [string, readonly number[]])[] = [
  ["selfHandMask", [CARD_COUNT]],
  ["legalBidMask", [BIDDING_ACTION_COUNT]],
  ["starterPlayerOneHot", [PLAYER_COUNT]],
  ["highestBidPresent", [1]],
  ["highestBidPlayerOneHot", [PLAYER_COUNT]],
  ["highestBidSuitOneHot", [BIDDING_HISTORY_SUIT_ORDER.length]],
  ["highestBidTargetPointCardsOneHot", [BIDDING_TARGET_POINT_CARDS_CLASS_COUNT]],
  ["consecutivePassCountOneHot", [CONSECUTIVE_PASS_COUNT_CLASS_COUNT]],
  ["biddingHistoryActionMask", [MAX_BIDDING_ACTION_COUNT]],
  ...BIDDING_HISTORY_ONEHOT_SPEC
];

const EXCHANGE_MODEL_INPUT_SPEC: readonly (readonly [string, readonly number[]])[] = [
  ["trumpSuitOneHot", [BIDDING_HISTORY_SUIT_ORDER.length]],
  ["selfHandMask", [CARD_COUNT]],
  ["legalDiscardCardMask", [CARD_COUNT]],
  ["calledAdjutantCardMask", [CARD_COUNT]],
  ["contractTargetPointCardsOneHot", [CONTRACT_TARGET_POINT_CARDS_CLASS_COUNT]],
  ["handCountByPlayer", [PLAYER_COUNT]],
  ["specialCardIndicesOneHot", [SPECIAL_CARD_INDEX_COUNT, CARD_COUNT]],
  ["biddingHistoryActionMask", [MAX_BIDDING_ACTION_COUNT]],
  ...BIDDING_HISTORY_ONEHOT_SPEC
];

const ADJUTANT_MODEL_INPUT_SPEC: readonly (readonly [string, readonly number[]])[] = [
  ["trumpSuitOneHot", [BIDDING_HISTORY_SUIT_ORDER.length]],
  ["selfHandMask", [CARD_COUNT]],
  ["legalAdjutantMask", [CARD_COUNT]],
  ["contractTargetPointCardsOneHot", [CONTRACT_TARGET_POINT_CARDS_CLASS_COUNT]],
  ["specialCardIndicesOneHot", [SPECIAL_CARD_INDEX_COUNT, CARD_COUNT]],
  ["biddingHistoryActionMask", [MAX_BIDDING_ACTION_COUNT]],
  ...BIDDING_HISTORY_ONEHOT_SPEC
];

export const FLAT_OBSERVATION_LAYOUT: readonly ModelInputFeatureSlice[] = buildLayout(
  FLAT_LAYOUT_SPEC,
  0
);
export const MODEL_INPUT_ONEHOT_LAYOUT: readonly ModelInputFeatureSlice[] = buildLayout(
  MODEL_INPUT_ONEHOT_SPEC,
  FLAT_OBSERVATION_FEATURE_COUNT
);
export const MODEL_INPUT_LAYOUT: readonly ModelInputFeatureSlice[] = [
  ...FLAT_OBSERVATION_LAYOUT,
  ...MODEL_INPUT_ONEHOT_LAYOUT
];
export const COMPLETE_INFO_PLAYING_MODEL_INPUT_LAYOUT: readonly ModelInputFeatureSlice[] =
  buildLayout(COMPLETE_INFO_PLAYING_MODEL_INPUT_SPEC, 0);
export const BIDDING_MODEL_INPUT_LAYOUT: readonly ModelInputFeatureSlice[] = buildLayout(
  BIDDING_MODEL_INPUT_SPEC,
  0
);
export const EXCHANGE_MODEL_INPUT_LAYOUT: readonly ModelInputFeatureSlice[] = buildLayout(
  EXCHANGE_MODEL_INPUT_SPEC,
  0
);
export const ADJUTANT_MODEL_INPUT_LAYOUT: readonly ModelInputFeatureSlice[] = buildLayout(
  ADJUTANT_MODEL_INPUT_SPEC,
  0
);

validateLayout(FLAT_OBSERVATION_LAYOUT, 0, FLAT_OBSERVATION_FEATURE_COUNT, "FLAT_OBSERVATION_LAYOUT");
validateLayout(
  MODEL_INPUT_ONEHOT_LAYOUT,
  FLAT_OBSERVATION_FEATURE_COUNT,
  MODEL_INPUT_FEATURE_COUNT,
  "MODEL_INPUT_ONEHOT_LAYOUT"
);
validateLayout(MODEL_INPUT_LAYOUT, 0, MODEL_INPUT_FEATURE_COUNT, "MODEL_INPUT_LAYOUT");
validateLayout(
  COMPLETE_INFO_PLAYING_MODEL_INPUT_LAYOUT,
  0,
  COMPLETE_INFO_PLAYING_MODEL_INPUT_FEATURE_COUNT,
  "COMPLETE_INFO_PLAYING_MODEL_INPUT_LAYOUT"
);
validateLayout(
  BIDDING_MODEL_INPUT_LAYOUT,
  0,
  BIDDING_MODEL_INPUT_FEATURE_COUNT,
  "BIDDING_MODEL_INPUT_LAYOUT"
);
validateLayout(
  EXCHANGE_MODEL_INPUT_LAYOUT,
  0,
  EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
  "EXCHANGE_MODEL_INPUT_LAYOUT"
);
validateLayout(
  ADJUTANT_MODEL_INPUT_LAYOUT,
  0,
  ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
  "ADJUTANT_MODEL_INPUT_LAYOUT"
);

if (
  COMPLETE_INFO_PLAYING_MODEL_INPUT_SCHEMA_VERSION !== 1 ||
  BIDDING_MODEL_INPUT_SCHEMA_VERSION !== 1 ||
  EXCHANGE_MODEL_INPUT_SCHEMA_VERSION !== 1 ||
  ADJUTANT_MODEL_INPUT_SCHEMA_VERSION !== 1
) {
  throw new Error("Complete-info/non-playing model_input schema versions must match schema v1.");
}

export function createPlayingModelInput(
  observation: EncodedPlayingObservation
): PlayingModelInput {
  return {
    modelInput: encodePlayingModelInput(observation),
    legalPlayMask: observation.legalPlayMask
  };
}

export function encodePlayingModelInput(
  observation: EncodedPlayingObservation
): Float32Array {
  validateEncodedPlayingObservation(observation);

  const flatObservation = flattenObservation(observation);
  const modelInputParts: number[] = [...flatObservation];

  for (const field of createOneHotFields(observation)) {
    appendOneHotIndexField(modelInputParts, field);
  }
  append(modelInputParts, observation.selfRoleOneHot);

  if (modelInputParts.length !== MODEL_INPUT_FEATURE_COUNT) {
    throw new Error(
      `model_input length must be ${MODEL_INPUT_FEATURE_COUNT}, got ${modelInputParts.length}.`
    );
  }

  return Float32Array.from(modelInputParts);
}

export function createCompleteInfoPlayingModelInput(
  observation: EncodedCompleteInfoPlayingObservation
): CompleteInfoPlayingModelInput {
  return {
    modelInput: encodeCompleteInfoPlayingModelInput(observation),
    legalPlayMask: observation.legalPlayMask
  };
}

export function encodeCompleteInfoPlayingModelInput(
  observation: EncodedCompleteInfoPlayingObservation
): Float32Array {
  validateEncodedCompleteInfoPlayingObservation(observation);

  const modelInputParts: number[] = [];

  appendOneHotIndexField(modelInputParts, {
    name: "cardOwnerClassByCardOneHot",
    indices: observation.cardOwnerClassByCard,
    slotCount: CARD_COUNT,
    classCount: COMPLETE_INFO_PLAYING_OWNER_CLASS_COUNT,
    minValue: 0,
    emptyValues: []
  });
  append(modelInputParts, observation.capturedPointCardCountByPlayer);
  append(modelInputParts, observation.trumpSuitOneHot);
  modelInputParts.push(observation.contractTargetPointCards);
  append(modelInputParts, observation.napoleonPlayerOneHot);
  append(modelInputParts, observation.revealedAdjutantPlayerOneHot);
  append(modelInputParts, observation.selfRoleOneHot);
  modelInputParts.push(observation.calledAdjutantCardIndex);
  modelInputParts.push(
    observation.specialCardIndices.oruma,
    observation.specialCardIndices.yoromeki,
    observation.specialCardIndices.seiJack,
    observation.specialCardIndices.uraJack
  );
  append(modelInputParts, observation.currentTrickSlotMask);
  append(modelInputParts, observation.currentTrickCardIndices);
  appendOneHotIndexField(modelInputParts, {
    name: "currentTrickPlayerIndicesOneHot",
    indices: observation.currentTrickPlayerIndices,
    slotCount: CARDS_PER_TRICK,
    classCount: PLAYER_COUNT,
    minValue: 0,
    emptyValues: [-1]
  });
  modelInputParts.push(observation.trickNumber, observation.completedTrickCount);

  return toCheckedFloat32Array(
    modelInputParts,
    COMPLETE_INFO_PLAYING_MODEL_INPUT_FEATURE_COUNT,
    "complete-info playing model_input"
  );
}

export function createBiddingModelInput(
  observation: EncodedBiddingObservation
): BiddingModelInput {
  return {
    modelInput: encodeBiddingModelInput(observation),
    legalBidMask: observation.legalBidMask
  };
}

export function encodeBiddingModelInput(
  observation: EncodedBiddingObservation
): Float32Array {
  validateEncodedBiddingObservation(observation);

  const modelInputParts: number[] = [];

  append(modelInputParts, observation.selfHandMask);
  append(modelInputParts, observation.legalBidMask);
  appendOneHotIndexField(modelInputParts, {
    name: "starterPlayerOneHot",
    indices: [observation.starterPlayerIndex],
    slotCount: 1,
    classCount: PLAYER_COUNT,
    minValue: 0,
    emptyValues: []
  });
  modelInputParts.push(observation.highestBidPresent);
  appendOneHotIndexField(modelInputParts, {
    name: "highestBidPlayerOneHot",
    indices: [observation.highestBidPlayerIndex],
    slotCount: 1,
    classCount: PLAYER_COUNT,
    minValue: 0,
    emptyValues: [-1]
  });
  appendOneHotIndexField(modelInputParts, {
    name: "highestBidSuitOneHot",
    indices: [observation.highestBidSuitIndex],
    slotCount: 1,
    classCount: BIDDING_HISTORY_SUIT_ORDER.length,
    minValue: 0,
    emptyValues: [-1]
  });
  appendOneHotIndexField(modelInputParts, {
    name: "highestBidTargetPointCardsOneHot",
    indices: [observation.highestBidTargetPointCards],
    slotCount: 1,
    classCount: BIDDING_TARGET_POINT_CARDS_CLASS_COUNT,
    minValue: MIN_BIDDING_TARGET_POINT_CARDS,
    emptyValues: [0]
  });
  appendOneHotIndexField(modelInputParts, {
    name: "consecutivePassCountOneHot",
    indices: [observation.consecutivePassCount],
    slotCount: 1,
    classCount: CONSECUTIVE_PASS_COUNT_CLASS_COUNT,
    minValue: 0,
    emptyValues: []
  });
  appendBiddingHistoryModelInputParts(modelInputParts, observation.biddingHistory);

  return toCheckedFloat32Array(
    modelInputParts,
    BIDDING_MODEL_INPUT_FEATURE_COUNT,
    "bidding model_input"
  );
}

export function createExchangeModelInput(
  observation: EncodedExchangeObservation
): ExchangeModelInput {
  return {
    modelInput: encodeExchangeModelInput(observation),
    legalDiscardCardMask: observation.legalDiscardCardMask
  };
}

export function encodeExchangeModelInput(
  observation: EncodedExchangeObservation
): Float32Array {
  validateEncodedExchangeObservation(observation);

  const modelInputParts: number[] = [];

  append(modelInputParts, observation.trumpSuitOneHot);
  append(modelInputParts, observation.selfHandMask);
  append(modelInputParts, observation.legalDiscardCardMask);
  append(modelInputParts, observation.calledAdjutantCardMask);
  appendOneHotIndexField(modelInputParts, {
    name: "contractTargetPointCardsOneHot",
    indices: [observation.contractTargetPointCards],
    slotCount: 1,
    classCount: CONTRACT_TARGET_POINT_CARDS_CLASS_COUNT,
    minValue: MIN_CONTRACT_TARGET_POINT_CARDS,
    emptyValues: []
  });
  append(modelInputParts, observation.handCountByPlayer);
  appendSpecialCardIndicesOneHot(modelInputParts, observation.specialCardIndices, "exchange");
  appendBiddingHistoryModelInputParts(modelInputParts, observation.biddingHistory);

  return toCheckedFloat32Array(
    modelInputParts,
    EXCHANGE_MODEL_INPUT_FEATURE_COUNT,
    "exchange model_input"
  );
}

export function createAdjutantModelInput(
  observation: EncodedAdjutantObservation
): AdjutantModelInput {
  return {
    modelInput: encodeAdjutantModelInput(observation),
    legalAdjutantMask: observation.legalAdjutantMask
  };
}

export function encodeAdjutantModelInput(
  observation: EncodedAdjutantObservation
): Float32Array {
  validateEncodedAdjutantObservation(observation);

  const modelInputParts: number[] = [];

  append(modelInputParts, observation.trumpSuitOneHot);
  append(modelInputParts, observation.selfHandMask);
  append(modelInputParts, observation.legalAdjutantMask);
  appendOneHotIndexField(modelInputParts, {
    name: "contractTargetPointCardsOneHot",
    indices: [observation.contractTargetPointCards],
    slotCount: 1,
    classCount: CONTRACT_TARGET_POINT_CARDS_CLASS_COUNT,
    minValue: MIN_CONTRACT_TARGET_POINT_CARDS,
    emptyValues: []
  });
  appendSpecialCardIndicesOneHot(modelInputParts, observation.specialCardIndices, "adjutant");
  appendBiddingHistoryModelInputParts(modelInputParts, observation.biddingHistory);

  return toCheckedFloat32Array(
    modelInputParts,
    ADJUTANT_MODEL_INPUT_FEATURE_COUNT,
    "adjutant model_input"
  );
}

function flattenObservation(observation: EncodedPlayingObservation): readonly number[] {
  const values: number[] = [];

  append(values, observation.trumpSuitOneHot);
  append(values, observation.napoleonPlayerOneHot);
  append(values, observation.revealedAdjutantPlayerOneHot);
  append(values, observation.calledAdjutantCardMask);
  append(values, observation.selfHandMask);
  append(values, observation.legalPlayMask);
  append(values, observation.handCountByPlayer);
  for (const mask of observation.capturedPointCardMaskByPlayer) {
    append(values, mask);
  }
  append(values, observation.currentTrickSlotMask);
  append(values, observation.completedTrickSlotMask);
  append(values, observation.completedTrickMask);
  append(values, observation.biddingHistory.actionMask);
  append(values, observation.latestBuriedEventPointCardMask);
  values.push(
    observation.trickNumber,
    observation.completedTrickCount,
    observation.contractTargetPointCards,
    observation.latestBuriedEventHiddenNonPointCount,
    observation.latestBuriedEventPresent
  );

  if (values.length !== FLAT_OBSERVATION_FEATURE_COUNT) {
    throw new Error(
      `flat_observation length must be ${FLAT_OBSERVATION_FEATURE_COUNT}, got ${values.length}.`
    );
  }

  return values;
}

function appendBiddingHistoryModelInputParts(
  target: number[],
  biddingHistory: EncodedBiddingHistory
): void {
  append(target, biddingHistory.actionMask);
  appendOneHotIndexField(target, {
    name: "biddingHistoryActionTypeIndicesOneHot",
    indices: biddingHistory.actionTypeIndices,
    slotCount: MAX_BIDDING_ACTION_COUNT,
    classCount: BIDDING_ACTION_TYPE_CLASS_COUNT,
    minValue: 0,
    emptyValues: [-1]
  });
  appendOneHotIndexField(target, {
    name: "biddingHistoryPlayerIndicesOneHot",
    indices: biddingHistory.playerIndices,
    slotCount: MAX_BIDDING_ACTION_COUNT,
    classCount: PLAYER_COUNT,
    minValue: 0,
    emptyValues: [-1]
  });
  appendOneHotIndexField(target, {
    name: "biddingHistorySuitIndicesOneHot",
    indices: biddingHistory.suitIndices,
    slotCount: MAX_BIDDING_ACTION_COUNT,
    classCount: BIDDING_HISTORY_SUIT_ORDER.length,
    minValue: 0,
    emptyValues: [-1]
  });
  appendOneHotIndexField(target, {
    name: "biddingHistoryTargetPointCardsOneHot",
    indices: biddingHistory.targetPointCards,
    slotCount: MAX_BIDDING_ACTION_COUNT,
    classCount: BIDDING_TARGET_POINT_CARDS_CLASS_COUNT,
    minValue: MIN_BIDDING_TARGET_POINT_CARDS,
    emptyValues: [0]
  });
}

function appendSpecialCardIndicesOneHot(
  target: number[],
  specialCardIndices:
    | EncodedExchangeObservation["specialCardIndices"]
    | EncodedAdjutantObservation["specialCardIndices"],
  phase: "exchange" | "adjutant"
): void {
  appendOneHotIndexField(target, {
    name: `${phase} specialCardIndicesOneHot`,
    indices: [
      specialCardIndices.oruma,
      specialCardIndices.yoromeki,
      specialCardIndices.seiJack,
      specialCardIndices.uraJack
    ],
    slotCount: SPECIAL_CARD_INDEX_COUNT,
    classCount: CARD_COUNT,
    minValue: 0,
    emptyValues: []
  });
}

function createOneHotFields(
  observation: EncodedPlayingObservation
): readonly OneHotIndexField[] {
  return [
    {
      name: "specialCardIndicesOneHot",
      indices: [
        observation.specialCardIndices.oruma,
        observation.specialCardIndices.yoromeki,
        observation.specialCardIndices.seiJack,
        observation.specialCardIndices.uraJack
      ],
      slotCount: SPECIAL_CARD_INDEX_COUNT,
      classCount: CARD_COUNT,
      minValue: 0,
      emptyValues: [-1]
    },
    {
      name: "currentTrickCardIndicesOneHot",
      indices: observation.currentTrickCardIndices,
      slotCount: CARDS_PER_TRICK,
      classCount: CARD_COUNT,
      minValue: 0,
      emptyValues: [-1]
    },
    {
      name: "completedTrickCardIndicesOneHot",
      indices: observation.completedTrickCardIndices,
      slotCount: COMPLETED_TRICK_CARD_SLOT_COUNT,
      classCount: CARD_COUNT,
      minValue: 0,
      emptyValues: [-1]
    },
    {
      name: "currentTrickPlayerIndicesOneHot",
      indices: observation.currentTrickPlayerIndices,
      slotCount: CARDS_PER_TRICK,
      classCount: PLAYER_COUNT,
      minValue: 0,
      emptyValues: [-1]
    },
    {
      name: "completedTrickPlayerIndicesOneHot",
      indices: observation.completedTrickPlayerIndices,
      slotCount: COMPLETED_TRICK_CARD_SLOT_COUNT,
      classCount: PLAYER_COUNT,
      minValue: 0,
      emptyValues: [-1]
    },
    {
      name: "completedTrickWinnerIndicesOneHot",
      indices: observation.completedTrickWinnerIndices,
      slotCount: TRICK_COUNT,
      classCount: PLAYER_COUNT,
      minValue: 0,
      emptyValues: [-1]
    },
    {
      name: "biddingHistoryActionTypeIndicesOneHot",
      indices: observation.biddingHistory.actionTypeIndices,
      slotCount: MAX_BIDDING_ACTION_COUNT,
      classCount: BIDDING_ACTION_TYPE_CLASS_COUNT,
      minValue: 0,
      emptyValues: [-1]
    },
    {
      name: "biddingHistoryPlayerIndicesOneHot",
      indices: observation.biddingHistory.playerIndices,
      slotCount: MAX_BIDDING_ACTION_COUNT,
      classCount: PLAYER_COUNT,
      minValue: 0,
      emptyValues: [-1]
    },
    {
      name: "biddingHistorySuitIndicesOneHot",
      indices: observation.biddingHistory.suitIndices,
      slotCount: MAX_BIDDING_ACTION_COUNT,
      classCount: BIDDING_HISTORY_SUIT_ORDER.length,
      minValue: 0,
      emptyValues: [-1]
    },
    {
      name: "biddingHistoryTargetPointCardsOneHot",
      indices: observation.biddingHistory.targetPointCards,
      slotCount: MAX_BIDDING_ACTION_COUNT,
      classCount: BIDDING_TARGET_POINT_CARDS_CLASS_COUNT,
      minValue: MIN_BIDDING_TARGET_POINT_CARDS,
      emptyValues: [0]
    }
  ];
}

function appendOneHotIndexField(target: number[], field: OneHotIndexField): void {
  if (field.indices.length !== field.slotCount) {
    throw new Error(`${field.name} must have ${field.slotCount} indices.`);
  }

  for (const indexValue of field.indices) {
    const classIndex = indexValue - field.minValue;
    if (
      (classIndex < 0 || classIndex >= field.classCount) &&
      !field.emptyValues.includes(indexValue)
    ) {
      throw new Error(
        `${field.name} index must be empty or between ${field.minValue} and ${field.minValue + field.classCount - 1}, got ${indexValue}.`
      );
    }

    for (let candidate = 0; candidate < field.classCount; candidate += 1) {
      target.push(classIndex === candidate ? 1 : 0);
    }
  }
}

function append(target: number[], values: readonly number[]): void {
  for (const value of values) {
    target.push(value);
  }
}

function toCheckedFloat32Array(
  modelInputParts: readonly number[],
  featureCount: number,
  label: string
): Float32Array {
  if (modelInputParts.length !== featureCount) {
    throw new Error(`${label} length must be ${featureCount}, got ${modelInputParts.length}.`);
  }

  return Float32Array.from(modelInputParts);
}

function buildLayout(
  spec: readonly (readonly [string, readonly number[]])[],
  start: number
): readonly ModelInputFeatureSlice[] {
  const slices: ModelInputFeatureSlice[] = [];
  let offset = start;

  for (const [name, shape] of spec) {
    const length = shape.reduce((product, dimension) => product * dimension, 1);
    slices.push({
      name,
      start: offset,
      stop: offset + length,
      shape,
      dtype: "float32"
    });
    offset += length;
  }

  return slices;
}

function validateLayout(
  layout: readonly ModelInputFeatureSlice[],
  expectedStart: number,
  expectedStop: number,
  label: string
): void {
  let offset = expectedStart;
  const names = new Set<string>();

  for (const feature of layout) {
    if (feature.start !== offset) {
      throw new Error(`${label} has a gap or overlap before ${feature.name}.`);
    }
    if (feature.stop <= feature.start) {
      throw new Error(`${label} slice ${feature.name} is empty.`);
    }
    if (names.has(feature.name)) {
      throw new Error(`${label} contains duplicate feature name ${feature.name}.`);
    }

    names.add(feature.name);
    offset = feature.stop;
  }

  if (offset !== expectedStop) {
    throw new Error(`${label} must stop at ${expectedStop}, got ${offset}.`);
  }
}
