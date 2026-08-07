import type { EncodedPlayingObservation } from "./encodePlayingObservation.js";
import { validateEncodedPlayingObservation } from "./encodePlayingObservation.js";
import {
  BIDDING_HISTORY_SUIT_ORDER,
  CARD_COUNT,
  CARDS_PER_TRICK,
  FLAT_OBSERVATION_FEATURE_COUNT,
  MAX_BIDDING_ACTION_COUNT,
  MAX_BIDDING_TARGET_POINT_CARDS,
  MIN_BIDDING_TARGET_POINT_CARDS,
  MODEL_INPUT_FEATURE_COUNT,
  PLAYER_COUNT,
  TRICK_COUNT
} from "./schema.js";

const COMPLETED_TRICK_CARD_SLOT_COUNT = TRICK_COUNT * CARDS_PER_TRICK;
const SPECIAL_CARD_INDEX_COUNT = 4;
const BIDDING_ACTION_TYPE_CLASS_COUNT = 2;
const BIDDING_TARGET_POINT_CARDS_CLASS_COUNT =
  MAX_BIDDING_TARGET_POINT_CARDS - MIN_BIDDING_TARGET_POINT_CARDS + 1;

interface OneHotIndexField {
  name: string;
  indices: readonly number[];
  slotCount: number;
  classCount: number;
  minValue: number;
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
  ["biddingHistoryTargetPointCardsOneHot", [MAX_BIDDING_ACTION_COUNT, BIDDING_TARGET_POINT_CARDS_CLASS_COUNT]]
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

validateLayout(FLAT_OBSERVATION_LAYOUT, 0, FLAT_OBSERVATION_FEATURE_COUNT, "FLAT_OBSERVATION_LAYOUT");
validateLayout(
  MODEL_INPUT_ONEHOT_LAYOUT,
  FLAT_OBSERVATION_FEATURE_COUNT,
  MODEL_INPUT_FEATURE_COUNT,
  "MODEL_INPUT_ONEHOT_LAYOUT"
);
validateLayout(MODEL_INPUT_LAYOUT, 0, MODEL_INPUT_FEATURE_COUNT, "MODEL_INPUT_LAYOUT");

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

  if (modelInputParts.length !== MODEL_INPUT_FEATURE_COUNT) {
    throw new Error(
      `model_input length must be ${MODEL_INPUT_FEATURE_COUNT}, got ${modelInputParts.length}.`
    );
  }

  return Float32Array.from(modelInputParts);
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
      minValue: 0
    },
    {
      name: "currentTrickCardIndicesOneHot",
      indices: observation.currentTrickCardIndices,
      slotCount: CARDS_PER_TRICK,
      classCount: CARD_COUNT,
      minValue: 0
    },
    {
      name: "completedTrickCardIndicesOneHot",
      indices: observation.completedTrickCardIndices,
      slotCount: COMPLETED_TRICK_CARD_SLOT_COUNT,
      classCount: CARD_COUNT,
      minValue: 0
    },
    {
      name: "currentTrickPlayerIndicesOneHot",
      indices: observation.currentTrickPlayerIndices,
      slotCount: CARDS_PER_TRICK,
      classCount: PLAYER_COUNT,
      minValue: 0
    },
    {
      name: "completedTrickPlayerIndicesOneHot",
      indices: observation.completedTrickPlayerIndices,
      slotCount: COMPLETED_TRICK_CARD_SLOT_COUNT,
      classCount: PLAYER_COUNT,
      minValue: 0
    },
    {
      name: "completedTrickWinnerIndicesOneHot",
      indices: observation.completedTrickWinnerIndices,
      slotCount: TRICK_COUNT,
      classCount: PLAYER_COUNT,
      minValue: 0
    },
    {
      name: "biddingHistoryActionTypeIndicesOneHot",
      indices: observation.biddingHistory.actionTypeIndices,
      slotCount: MAX_BIDDING_ACTION_COUNT,
      classCount: BIDDING_ACTION_TYPE_CLASS_COUNT,
      minValue: 0
    },
    {
      name: "biddingHistoryPlayerIndicesOneHot",
      indices: observation.biddingHistory.playerIndices,
      slotCount: MAX_BIDDING_ACTION_COUNT,
      classCount: PLAYER_COUNT,
      minValue: 0
    },
    {
      name: "biddingHistorySuitIndicesOneHot",
      indices: observation.biddingHistory.suitIndices,
      slotCount: MAX_BIDDING_ACTION_COUNT,
      classCount: BIDDING_HISTORY_SUIT_ORDER.length,
      minValue: 0
    },
    {
      name: "biddingHistoryTargetPointCardsOneHot",
      indices: observation.biddingHistory.targetPointCards,
      slotCount: MAX_BIDDING_ACTION_COUNT,
      classCount: BIDDING_TARGET_POINT_CARDS_CLASS_COUNT,
      minValue: MIN_BIDDING_TARGET_POINT_CARDS
    }
  ];
}

function appendOneHotIndexField(target: number[], field: OneHotIndexField): void {
  if (field.indices.length !== field.slotCount) {
    throw new Error(`${field.name} must have ${field.slotCount} indices.`);
  }

  for (const indexValue of field.indices) {
    const classIndex = indexValue - field.minValue;

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
