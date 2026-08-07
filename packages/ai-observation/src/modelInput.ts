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

export interface PlayingModelInput {
  modelInput: Float32Array;
  legalPlayMask: readonly number[];
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
