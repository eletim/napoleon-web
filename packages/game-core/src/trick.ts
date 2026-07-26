import { GameRuleError } from "./errors.js";
import { getRankValue } from "./ranks.js";
import {
  isJokerCard,
  isOrumaCard,
  isSeiJackCard,
  isStandardCard,
  isUraJackCard,
  isYoromekiCard
} from "./cards.js";
import type { Card, PlayedCard, PlayerId, Suit } from "./types.js";

export type TrickCardCategory = "trump" | "lead" | "other";

export interface TrickContext {
  trumpSuit: Suit | null;
}

export interface TrickCardStrength {
  category: TrickCardCategory;
  categoryPriority: number;
  rankPriority: number;
}

export interface DetermineTrickWinnerOptions {
  trickNumber?: number;
}

const categoryPriorities: Record<TrickCardCategory, number> = {
  other: 0,
  lead: 1,
  trump: 2
};
const jokerRankPriority = 1;

export function getLeadSuit(
  currentTrick: readonly PlayedCard[],
  context: TrickContext
): Suit | undefined {
  const leadCard = currentTrick[0]?.card;

  if (leadCard === undefined) {
    return undefined;
  }

  if (isStandardCard(leadCard)) {
    return leadCard.suit;
  }

  if (context.trumpSuit === null) {
    throw new GameRuleError(
      "TRUMP_NOT_SET",
      "Trump suit must be set when joker leads a trick."
    );
  }

  return context.trumpSuit;
}

export function getPlayableCards(
  hand: readonly Card[],
  currentTrick: readonly PlayedCard[],
  context: TrickContext
): readonly Card[] {
  const leadSuit = getLeadSuit(currentTrick, context);

  if (leadSuit === undefined) {
    return hand;
  }

  const hasFollowCards = hand.some((card) => isStandardCard(card) && card.suit === leadSuit);
  return hasFollowCards
    ? hand.filter((card) => isJokerCard(card) || (isStandardCard(card) && card.suit === leadSuit))
    : hand;
}

export function getTrickCardStrength(
  card: Card,
  leadSuit: Suit,
  context: TrickContext,
  options: { isLeadCard?: boolean } = {}
): TrickCardStrength {
  const category = getTrickCardCategory(card, leadSuit, context, options);

  return {
    category,
    categoryPriority: categoryPriorities[category],
    rankPriority: isJokerCard(card) ? jokerRankPriority : getRankValue(card.rank)
  };
}

export function determineTrickWinner(
  trick: readonly PlayedCard[],
  context: TrickContext,
  options: DetermineTrickWinnerOptions = {}
): PlayerId {
  const sameTwoWinner = findSameTwoWinner(trick, context, options.trickNumber);

  if (sameTwoWinner !== undefined) {
    return sameTwoWinner;
  }

  const specialWinner = determineSpecialCardWinner(trick, context);

  if (specialWinner !== undefined) {
    return specialWinner;
  }

  const leadSuit = getLeadSuit(trick, context);

  if (leadSuit === undefined) {
    throw new GameRuleError("TRICK_NOT_COMPLETE", "Cannot determine a winner for an empty trick.");
  }

  const winningCard = trick.reduce((winner, candidate, candidateIndex) =>
    comparePlayedCards(
      candidate,
      winner,
      leadSuit,
      context,
      candidateIndex,
      trick.indexOf(winner)
    ) > 0
      ? candidate
      : winner
  );

  return winningCard.playerId;
}

function findSameTwoWinner(
  trick: readonly PlayedCard[],
  context: TrickContext,
  trickNumber: number | undefined
): PlayerId | undefined {
  if (trickNumber === undefined || trickNumber <= 1 || trick.length !== 5) {
    return undefined;
  }

  if (trick.some((playedCard) => isJokerCard(playedCard.card))) {
    return undefined;
  }

  const standardCards = trick.map((playedCard) => playedCard.card).filter(isStandardCard);

  if (standardCards.length !== trick.length) {
    return undefined;
  }

  const sameTwoSuit = standardCards[0]?.suit;

  if (sameTwoSuit === undefined || !standardCards.every((card) => card.suit === sameTwoSuit)) {
    return undefined;
  }

  if (standardCards.some((card) => isOrumaCard(card))) {
    return undefined;
  }

  if (context.trumpSuit === null && standardCards.some((card) => card.rank === "J")) {
    return undefined;
  }

  if (context.trumpSuit !== null) {
    const trumpSuit = context.trumpSuit;

    if (
      standardCards.some(
        (card) => isSeiJackCard(card, trumpSuit) || isUraJackCard(card, trumpSuit)
      )
    ) {
      return undefined;
    }
  }

  return trick.find(
    (playedCard) =>
      isStandardCard(playedCard.card) &&
      playedCard.card.suit === sameTwoSuit &&
      playedCard.card.rank === "2"
  )?.playerId;
}

function determineSpecialCardWinner(
  trick: readonly PlayedCard[],
  context: TrickContext
): PlayerId | undefined {
  const orumaPlay = trick.find((playedCard) => isOrumaCard(playedCard.card));
  const yoromekiPlay = trick.find((playedCard) => isYoromekiCard(playedCard.card));

  if (orumaPlay !== undefined && yoromekiPlay !== undefined) {
    return yoromekiPlay.playerId;
  }

  if (orumaPlay !== undefined) {
    return orumaPlay.playerId;
  }

  if (trick.some((playedCard) => isStandardCard(playedCard.card) && playedCard.card.rank === "J")) {
    if (context.trumpSuit === null) {
      throw new GameRuleError(
        "TRUMP_NOT_SET",
        "Trump suit must be set to determine sei jack and ura jack."
      );
    }

    const trumpSuit = context.trumpSuit;
    const seiJackPlay = trick.find((playedCard) =>
      isSeiJackCard(playedCard.card, trumpSuit)
    );

    if (seiJackPlay !== undefined) {
      return seiJackPlay.playerId;
    }

    const uraJackPlay = trick.find((playedCard) =>
      isUraJackCard(playedCard.card, trumpSuit)
    );

    if (uraJackPlay !== undefined) {
      return uraJackPlay.playerId;
    }
  }

  return undefined;
}

function getTrickCardCategory(
  card: Card,
  leadSuit: Suit,
  context: TrickContext,
  options: { isLeadCard?: boolean }
): TrickCardCategory {
  if (isJokerCard(card)) {
    return options.isLeadCard === true ? "trump" : "lead";
  }

  if (context.trumpSuit !== null && card.suit === context.trumpSuit) {
    return "trump";
  }

  if (card.suit === leadSuit) {
    return "lead";
  }

  return "other";
}

function comparePlayedCards(
  left: PlayedCard,
  right: PlayedCard,
  leadSuit: Suit,
  context: TrickContext,
  leftIndex: number,
  rightIndex: number
): number {
  const leftStrength = getTrickCardStrength(left.card, leadSuit, context, {
    isLeadCard: leftIndex === 0
  });
  const rightStrength = getTrickCardStrength(right.card, leadSuit, context, {
    isLeadCard: rightIndex === 0
  });

  if (leftStrength.categoryPriority !== rightStrength.categoryPriority) {
    return leftStrength.categoryPriority - rightStrength.categoryPriority;
  }

  return leftStrength.rankPriority - rightStrength.rankPriority;
}
