import type { ChooseAdjutantAction, GameAction, PlayerId } from "@napoleon/game-core";
import { getCardId, getCardIndex } from "./cardIndex.js";
import { CARD_COUNT } from "./schema.js";

export type DecodedAdjutantAction = ChooseAdjutantAction;

export function encodeAdjutantAction(
  action: GameAction,
  legalAdjutantMask?: readonly number[],
  expectedPlayerId?: PlayerId
): number {
  if (action.type !== "choose-adjutant") {
    throw new Error(`encodeAdjutantAction requires a choose-adjutant action, got ${action.type}.`);
  }

  if (expectedPlayerId !== undefined && action.playerId !== expectedPlayerId) {
    throw new Error(
      `Adjutant action playerId must match the acting player: ${action.playerId} !== ${expectedPlayerId}`
    );
  }

  const actionIndex = getCardIndex(action.cardId);

  if (legalAdjutantMask !== undefined) {
    validateLegalAdjutantMask(legalAdjutantMask);

    if (legalAdjutantMask[actionIndex] !== 1) {
      throw new Error(`Adjutant actorTarget ${actionIndex} is not legal for this observation.`);
    }
  }

  return actionIndex;
}

export function decodeAdjutantAction(
  actionIndex: number,
  playerId: PlayerId
): DecodedAdjutantAction {
  expectIntegerInRange("adjutant action index", actionIndex, 0, CARD_COUNT - 1);

  return {
    type: "choose-adjutant",
    playerId,
    cardId: getCardId(actionIndex)
  };
}

export function encodeLegalAdjutantMask(
  legalActions: readonly GameAction[],
  expectedPlayerId: PlayerId
): readonly number[] {
  const mask = Array(CARD_COUNT).fill(0);

  for (const action of legalActions) {
    if (action.type !== "choose-adjutant") {
      throw new Error(
        `Adjutant legalActions must contain only choose-adjutant actions, got ${action.type}.`
      );
    }

    if (action.playerId !== expectedPlayerId) {
      throw new Error(
        `Legal adjutant action playerId must match observation playerId: ${action.playerId} !== ${expectedPlayerId}`
      );
    }

    mask[encodeAdjutantAction(action)] = 1;
  }

  return mask;
}

export function validateLegalAdjutantMask(mask: readonly number[]): void {
  expectLength("legalAdjutantMask", mask, CARD_COUNT);
  validateMask("legalAdjutantMask", mask);
}

function expectLength(name: string, value: readonly unknown[], expectedLength: number): void {
  if (value.length !== expectedLength) {
    throw new Error(`${name} must have length ${expectedLength}, got ${value.length}.`);
  }
}

function validateMask(name: string, mask: readonly number[]): void {
  for (const value of mask) {
    if (!Number.isFinite(value) || !Number.isInteger(value) || (value !== 0 && value !== 1)) {
      throw new Error(`${name} must contain only 0/1 values.`);
    }
  }
}

function expectIntegerInRange(name: string, value: number, min: number, max: number): void {
  if (!Number.isFinite(value) || !Number.isInteger(value)) {
    throw new Error(`${name} must be a finite integer.`);
  }

  if (value < min || value > max) {
    throw new Error(`${name} must be between ${min} and ${max}, got ${value}.`);
  }
}
