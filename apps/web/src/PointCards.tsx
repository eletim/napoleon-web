import type { CSSProperties } from "react";
import type { PublicStandardCard } from "@napoleon/protocol";
import { isRedSuit, suitSymbols } from "./cardSymbols";

interface PointCardsProps {
  cards: readonly PublicStandardCard[];
  fixedSlotCount?: number;
}

export function PointCards({ cards, fixedSlotCount }: PointCardsProps) {
  const emptySlotCount =
    fixedSlotCount === undefined ? 0 : Math.max(0, fixedSlotCount - cards.length);

  if (cards.length === 0) {
    return (
      <>
        <span className="muted-text point-cards-empty-label">なし</span>
        <PointCardEmptySlots count={emptySlotCount} />
      </>
    );
  }

  return (
    <>
      {cards.map((card, index) => (
        <span
          className={isRedSuit(card.suit) ? "mini-card red-text" : "mini-card black-text"}
          key={card.id}
          style={fixedSlotCount === undefined ? undefined : getFixedPointSlotStyle(index)}
        >
          {card.rank}
          {suitSymbols[card.suit]}
        </span>
      ))}
      <PointCardEmptySlots count={emptySlotCount} startIndex={cards.length} />
    </>
  );
}

function PointCardEmptySlots({ count, startIndex = 0 }: { count: number; startIndex?: number }) {
  return Array.from({ length: count }, (_, index) => (
    <span
      aria-hidden="true"
      className="point-card-empty-slot"
      key={`empty-point-card-slot-${index}`}
      style={getFixedPointSlotStyle(startIndex + index)}
    />
  ));
}

function getFixedPointSlotStyle(index: number): CSSProperties {
  const slotIndex = index % 10;
  const pageIndex = Math.floor(index / 10);
  const column = pageIndex * 5 + (slotIndex % 5) + 1;
  const row = Math.floor(slotIndex / 5) + 1;

  return {
    gridColumn: column,
    gridRow: row
  };
}
