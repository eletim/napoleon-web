import type { GameAction } from "@napoleon/game-core";
import { getCardIndex } from "./cardIndex.js";
import { CARD_COUNT } from "./schema.js";

export interface EncodedExchangeAction {
  discardTargetMask: readonly number[];
}

export function encodeDiscardAction(
  action: GameAction,
  legalDiscardCardMask: readonly number[],
  playerId?: string
): EncodedExchangeAction {
  validateLegalDiscardCardMask(legalDiscardCardMask);

  if (action.type !== "discard-cards") {
    throw new Error(`encodeDiscardAction requires a discard-cards action, got ${action.type}.`);
  }

  if (playerId !== undefined && action.playerId !== playerId) {
    throw new Error(
      `Discard action playerId must match the acting player: ${action.playerId} !== ${playerId}`
    );
  }

  if (action.cardIds.length !== 3) {
    throw new Error(`Discard action must contain exactly 3 card ids, got ${action.cardIds.length}.`);
  }

  if (new Set(action.cardIds).size !== action.cardIds.length) {
    throw new Error("Discard action card ids must be distinct.");
  }

  const discardTargetMask = createMask(action.cardIds);
  const encoded: EncodedExchangeAction = { discardTargetMask };

  validateEncodedExchangeAction(encoded, legalDiscardCardMask);

  return encoded;
}

export function validateEncodedExchangeAction(
  action: EncodedExchangeAction,
  legalDiscardCardMask: readonly number[]
): void {
  validateLegalDiscardCardMask(legalDiscardCardMask);
  expectLength("discardTargetMask", action.discardTargetMask, CARD_COUNT);
  validateMask("discardTargetMask", action.discardTargetMask);
  expectSum("discardTargetMask", action.discardTargetMask, 3);
  validateMaskIsSubset(
    "discardTargetMask",
    action.discardTargetMask,
    "legalDiscardCardMask",
    legalDiscardCardMask
  );
}

export function validateLegalDiscardCardMask(mask: readonly number[]): void {
  expectLength("legalDiscardCardMask", mask, CARD_COUNT);
  validateMask("legalDiscardCardMask", mask);
}

function createMask(cardIds: readonly string[]): readonly number[] {
  const mask = Array(CARD_COUNT).fill(0);

  for (const cardId of cardIds) {
    mask[getCardIndex(cardId)] = 1;
  }

  return mask;
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

function expectSum(name: string, values: readonly number[], expectedSum: number): void {
  const actualSum = values.reduce((total, value) => total + value, 0);

  if (actualSum !== expectedSum) {
    throw new Error(`${name} must sum to ${expectedSum}, got ${actualSum}.`);
  }
}

function validateMaskIsSubset(
  childName: string,
  child: readonly number[],
  parentName: string,
  parent: readonly number[]
): void {
  child.forEach((value, index) => {
    if (value === 1 && parent[index] !== 1) {
      throw new Error(`${childName}[${index}] must be within ${parentName}.`);
    }
  });
}
